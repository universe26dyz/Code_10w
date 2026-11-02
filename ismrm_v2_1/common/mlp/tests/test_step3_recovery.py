import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from modules.module_07_objective_training.mlp_trainer import _quantitative_regularization, build_mlp_training_model, train_mlp_reconstruction
from modules.module_07_objective_training.training_space import TrainingSpace
from modules.module_08_inference_export.export_quantitative import sample_quantitative_fields
from online_helpers import recon_config, write_functional_checkpoint, write_observations


_CLI_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_mlp_from_checkpoint.py"
_CLI_SPEC = importlib.util.spec_from_file_location("mlp_checkpoint_export_cli", _CLI_PATH)
assert _CLI_SPEC is not None and _CLI_SPEC.loader is not None
_CLI_MODULE = importlib.util.module_from_spec(_CLI_SPEC); _CLI_SPEC.loader.exec_module(_CLI_MODULE)
export_mlp_from_checkpoint = _CLI_MODULE.export_mlp_from_checkpoint


def _model(tmp_path):
    dataset = write_observations(tmp_path / "observations.npz")
    source = tmp_path / "decoder.pth"; write_functional_checkpoint(source)
    config = recon_config(source)
    model, space, _, _ = build_mlp_training_model(dataset, config, "configs/protocol_hhz_v1.yaml", torch.device("cpu"))
    return dataset, config, model, space


def test_mlp_per_field_regularization_uses_requested_points_and_detaches_amplitude(tmp_path):
    dataset, config, model, space = _model(tmp_path)
    settings = {
        "n_points": 7,
        "edge_epsilon": 1e-3,
        "t1": {"mode": "TV", "weight": 1.0},
        "t2": {"mode": "edge-preserving", "weight": 1.0},
        "b1": {"mode": "L2", "weight": 1.0},
        "amplitude_guidance": {"enabled": True, "alpha": 1.0, "t1_weight": 1.0, "t2_weight": 0.0},
    }
    values = _quantitative_regularization(model, space.local_to_train(dataset.xyz[:10]), space.spatial_scaling, config["loss"]["quantitative"], settings)
    assert all(torch.isfinite(values[key]) for key in ("t1", "t2", "b1", "amplitude_t1"))
    assert values["t1"].item() > 0 and values["amplitude_t1"].item() > 0
    assert "amplitude_b1" not in values
    assert values["amplitude_t2"].item() == 0
    values["amplitude_t1"].backward()
    amplitude_output_row = model.inr.parameter_head[2].weight.grad[3]
    torch.testing.assert_close(amplitude_output_row, torch.zeros_like(amplitude_output_row))


def test_mlp_amplitude_guidance_zero_weights_is_exactly_off(tmp_path):
    dataset, config, model, space = _model(tmp_path)
    settings = {"n_points": 7, "t1": {"mode": "TV", "weight": 1.0}, "t2": {"mode": "TV", "weight": 1.0}, "b1": {"mode": "L2", "weight": 1.0}, "amplitude_guidance": {"enabled": True, "alpha": 1.0, "t1_weight": 0.0, "t2_weight": 0.0}}
    values = _quantitative_regularization(model, space.local_to_train(dataset.xyz[:10]), space.spatial_scaling, config["loss"]["quantitative"], settings)
    assert values["amplitude_t1"].item() == 0 and values["amplitude_t2"].item() == 0


def test_mlp_all_spatial_modes_include_none_without_legacy_four_point_limit(tmp_path):
    _, config, _, _ = _model(tmp_path)

    class CountingINR(torch.nn.Module):
        def __init__(self): super().__init__(); self.last_count = 0
        def forward(self, points):
            self.last_count = points.shape[0]
            return {"t1_ms": 2500.0 * points[:, 0], "t2_ms": 5.0 + 195.0 * points[:, 1].pow(2), "b1": 0.1 + 1.1 * points[:, 2], "amplitude": points[:, 0]}

    inr = CountingINR(); model = SimpleNamespace(inr=inr); points = torch.arange(30, dtype=torch.float32).reshape(10, 3) / 10.0
    for mode in ("none", "TV", "L2", "edge-preserving"):
        settings = {"n_points": 7, "edge_epsilon": 1e-3, "t1": {"mode": mode, "weight": 1.0}, "t2": {"mode": mode, "weight": 1.0}, "b1": {"mode": mode, "weight": 1.0}, "amplitude_guidance": {"enabled": False, "t1_weight": 0.0, "t2_weight": 0.0}}
        values = _quantitative_regularization(model, points, 30.0, config["loss"]["quantitative"], settings)
        if mode != "none": assert inr.last_count == 7
        assert all(torch.isfinite(value) for value in values.values())
        if mode == "none": assert all(value.item() == 0 for value in values.values())


def test_mlp_training_log_separates_regularization_terms(tmp_path):
    dataset, config, _, _ = _model(tmp_path)
    config["training"]["stage_a_iterations"] = 1; config["training"]["stage_b_iterations"] = 0
    config["spatial_regularization"] = {
        "n_points": 7,
        "t1": {"mode": "TV", "weight": 1e-3}, "t2": {"mode": "TV", "weight": 1e-3}, "b1": {"mode": "TV", "weight": 1e-3},
        "amplitude_guidance": {"enabled": True, "alpha": 1.0, "t1_weight": 1e-3, "t2_weight": 0.0},
    }
    output = tmp_path / "outputs"; train_mlp_reconstruction(dataset, config, "configs/protocol_hhz_v1.yaml", output)
    with (output / "training_log.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert {"reg_t1", "reg_t2", "reg_b1", "amplitude_reg_t1", "amplitude_reg_t2"}.issubset(row)
    assert float(row["reg_t1"]) > 0 and float(row["reg_t2"]) > 0 and float(row["reg_b1"]) > 0
    assert float(row["amplitude_reg_t2"]) == 0.0


def test_checkpoint_only_export_restores_existing_state_without_training(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared" / "CYJ" / "sax"; prepared.mkdir(parents=True)
    dataset = write_observations(prepared / "observations.npz")
    source = tmp_path / "decoder.pth"; write_functional_checkpoint(source)
    config = recon_config(source); config["training"]["stage_a_iterations"] = 1; config["training"]["stage_b_iterations"] = 0
    trained = tmp_path / "trained"; train_mlp_reconstruction(dataset, config, "configs/protocol_hhz_v1.yaml", trained)
    from modules.module_05_signal_decoder.trad_teacher import trad_signal_simulator
    class UnexpectedBlochCall:
        def __init__(self, *args, **kwargs): raise AssertionError("checkpoint-only MLP export must not instantiate Bloch teacher")
    monkeypatch.setattr(trad_signal_simulator, "TradSignalSimulator", UnexpectedBlochCall)
    paths = export_mlp_from_checkpoint(tmp_path / "prepared", "CYJ", trained / "model.pt", tmp_path / "exported", output_resolution_mm=12.0, stack_names=("sax",))
    assert Path(paths["t1_ms"]).is_file() and Path(paths["support_mask"]).is_file()
    assert '"training_called": false' in Path(paths["provenance"]).read_text()
    checkpoint = torch.load(trained / "model.pt", map_location="cpu", weights_only=False)
    model, _, _, _ = build_mlp_training_model(dataset, checkpoint["resolved_config"], "configs/protocol_hhz_v1.yaml", torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state"])
    fields, _ = sample_quantitative_fields(model, TrainingSpace.from_state_dict(checkpoint["training_space"]), 12.0, 256)
    import nibabel as nib
    torch.testing.assert_close(torch.from_numpy(nib.load(paths["amplitude"]).get_fdata(dtype=np.float32)), torch.from_numpy(fields["amplitude"]))
    torch.testing.assert_close(torch.from_numpy(nib.load(paths["t1_ms"]).get_fdata(dtype=np.float32)), torch.from_numpy(fields["t1_ms"]))


def test_checkpoint_only_export_rejects_incomplete_checkpoint_before_export(tmp_path):
    dataset, config, _, _ = _model(tmp_path)
    config["training"]["stage_a_iterations"] = config["training"]["stage_b_iterations"] = 0
    trained = tmp_path / "trained"; train_mlp_reconstruction(dataset, config, "configs/protocol_hhz_v1.yaml", trained)
    payload = torch.load(trained / "model.pt", map_location="cpu", weights_only=False)
    payload.pop("intensity_normalization")
    invalid = tmp_path / "incomplete.pt"; torch.save(payload, invalid)
    with pytest.raises(ValueError, match="incomplete"):
        export_mlp_from_checkpoint(tmp_path / "missing", "CYJ", invalid, tmp_path / "exported", stack_names=("sax",))
