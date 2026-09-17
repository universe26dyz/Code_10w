import copy
import csv
import json
from pathlib import Path

import torch
import yaml
import pytest

from modules.module_07_objective_training.experiment_infrastructure import normalize_step1_config
from modules.module_07_objective_training.trad_trainer import (
    _balanced_mse,
    _data_loss,
    _optimizer,
    _quantitative_regularization,
    build_training_model,
    train_trad,
)
from tests.test_end_to_end_tiny import _config, _dataset


def test_regularization_modes_and_detached_amplitude_guidance_are_finite(tmp_path):
    dataset = _dataset(tmp_path / "observations.npz")
    config = _config(); config["loss"]["quantitative"] = {"t1": 1.0, "t2": 1.0, "b1": 1.0}
    model, space, _ = build_training_model(dataset, config, "configs/protocol_hhz_v1.yaml", torch.device("cpu"))
    points = space.local_to_train(dataset.xyz[:10])
    for mode in ("none", "TV", "L2", "edge-preserving"):
        settings = {"mode": mode, "n_points": 7, "edge_epsilon": 1e-3, "amplitude_guidance": {"enabled": True, "alpha": 1.0}}
        values = _quantitative_regularization(model, points, space.spatial_scaling, config["loss"]["quantitative"], settings)
        assert all(torch.isfinite(value) for value in values.values())
        if mode == "none":
            assert all(value.item() == 0 for value in values.values())
        else:
            sum(values.values()).backward()
            model.zero_grad(set_to_none=True)


def test_per_field_modes_and_zero_amplitude_weights_keep_guidance_exactly_off(tmp_path):
    dataset = _dataset(tmp_path / "observations.npz")
    config = _config(); config["loss"]["quantitative"] = {"t1": 1.0, "t2": 1.0, "b1": 1.0}
    model, space, _ = build_training_model(dataset, config, "configs/protocol_hhz_v1.yaml", torch.device("cpu"))
    settings = {
        "n_points": 7,
        "t1": {"mode": "TV", "weight": 1.0},
        "t2": {"mode": "edge-preserving", "weight": 1.0},
        "b1": {"mode": "L2", "weight": 1.0},
        "amplitude_guidance": {"enabled": True, "alpha": 1.0, "t1_weight": 0.0, "t2_weight": 0.0},
    }
    values = _quantitative_regularization(model, space.local_to_train(dataset.xyz[:10]), space.spatial_scaling, config["loss"]["quantitative"], settings)
    assert all(torch.isfinite(values[field]) for field in ("t1", "t2", "b1"))
    assert values["amplitude_t1"].item() == 0 and values["amplitude_t2"].item() == 0


def test_amplitude_guidance_detaches_amplitude_gradient_and_excludes_b1(tmp_path):
    dataset = _dataset(tmp_path / "observations.npz")
    config = _config(); config["loss"]["quantitative"] = {"t1": 1.0, "t2": 0.0, "b1": 0.0}
    model, space, _ = build_training_model(dataset, config, "configs/protocol_hhz_v1.yaml", torch.device("cpu"))
    settings = {"n_points": 7, "t1": {"mode": "TV", "weight": 1.0}, "t2": {"mode": "none", "weight": 0.0}, "b1": {"mode": "none", "weight": 0.0}, "amplitude_guidance": {"enabled": True, "alpha": 1.0, "t1_weight": 1.0, "t2_weight": 0.0}}
    values = _quantitative_regularization(model, space.local_to_train(dataset.xyz[:10]), space.spatial_scaling, config["loss"]["quantitative"], settings)
    values["amplitude_t1"].backward()
    amplitude_output_row = model.inr.parameter_head[2].weight.grad[3]
    torch.testing.assert_close(amplitude_output_row, torch.zeros_like(amplitude_output_row))
    assert values["amplitude_t2"].item() == 0


def test_server_baseline_enables_nonzero_tv_per_field_regularization(tmp_path):
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/server_train_example.yaml").read_text())
    regularization = config["spatial_regularization"]
    assert all(regularization[field]["mode"] == "TV" and regularization[field]["weight"] > 0 for field in ("t1", "t2", "b1"))
    dataset = _dataset(tmp_path / "observations.npz")
    model, space, _ = build_training_model(dataset, config, "configs/protocol_hhz_v1.yaml", torch.device("cpu"))
    values = _quantitative_regularization(model, space.local_to_train(dataset.xyz[:10]), space.spatial_scaling, config["loss"]["quantitative"], regularization)
    assert all(values[field].item() > 0 for field in ("t1", "t2", "b1"))


def test_variance_off_is_exact_balanced_mse_and_enabled_branch_is_trainable(tmp_path):
    dataset = _dataset(tmp_path / "observations.npz")
    config = _config()
    model, space, _ = build_training_model(dataset, config, "configs/protocol_hhz_v1.yaml", torch.device("cpu"))
    indices = torch.cat([torch.nonzero(dataset.weight_idx == weight, as_tuple=False)[0] for weight in range(10)])
    batch = {"xyz": space.local_to_train(dataset.xyz[indices]), "group_idx": dataset.group_idx[indices], "weight_idx": dataset.weight_idx[indices]}
    prediction, observed, weights = torch.arange(10, dtype=torch.float32), torch.zeros(10), torch.ones(10)
    torch.testing.assert_close(_data_loss(model, prediction, observed, batch, weights), _balanced_mse(prediction, observed, batch["weight_idx"], weights))
    assert model.variance_head is None
    assert all(group["name"] != "variance" for group in _optimizer(model, config["training"]["learning_rates"], joint=False).param_groups)
    config["variance"] = {"enabled": True, "pixel": True, "slice": True}
    config["training"]["learning_rates"]["variance"] = 0.05
    model, _, _ = build_training_model(dataset, config, "configs/protocol_hhz_v1.yaml", torch.device("cpu"))
    loss = _data_loss(model, prediction, observed, batch, weights)
    loss.backward()
    assert model.variance_head is not None and all(parameter.grad is not None for parameter in model.variance_head.parameters())
    before = [parameter.detach().clone() for parameter in model.variance_head.parameters()]
    optimizer = _optimizer(model, config["training"]["learning_rates"], joint=False)
    assert any(group["name"] == "variance" and group["lr"] == 0.05 for group in optimizer.param_groups)
    optimizer.step()
    assert any(not torch.equal(previous, current) for previous, current in zip(before, model.variance_head.parameters()))


def test_enabled_variance_requires_its_own_explicit_learning_rate():
    with pytest.raises(ValueError, match="learning_rates.variance"):
        normalize_step1_config({"training": {"psf_samples": 8, "learning_rates": {}}, "variance": {"enabled": True, "pixel": True, "slice": False}})


def test_training_log_separates_ordinary_and_amplitude_guided_regularization(tmp_path):
    source = tmp_path / "observations.npz"; dataset = _dataset(source)
    config = _config(); config["training"]["stage_a_iterations"] = 1; config["training"]["stage_b_iterations"] = 0
    config["spatial_regularization"] = {
        "n_points": 7,
        "t1": {"mode": "TV", "weight": 1e-3},
        "t2": {"mode": "TV", "weight": 1e-3},
        "b1": {"mode": "TV", "weight": 1e-3},
        "amplitude_guidance": {"enabled": True, "alpha": 1.0, "t1_weight": 1e-3, "t2_weight": 0.0},
    }
    output = tmp_path / "outputs"; train_trad(dataset, config, "configs/protocol_hhz_v1.yaml", output)
    with (output / "training_log.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert {"reg_t1", "reg_t2", "reg_b1", "amplitude_reg_t1", "amplitude_reg_t2"}.issubset(row)
    assert float(row["amplitude_reg_t2"]) == 0.0


def test_stack_initialization_is_saved_before_frozen_stage_a(tmp_path):
    source = tmp_path / "observations.npz"; dataset = _dataset(source)
    config = copy.deepcopy(_config())
    config["training"]["stage_a_iterations"] = config["training"]["stage_b_iterations"] = 0
    config["stack_initialization"] = {"enabled": True, "args_registration": {"num_levels": 1, "num_steps": 1, "max_iter": 1}}
    output = tmp_path / "outputs"
    result = train_trad(dataset, config, "configs/protocol_hhz_v1.yaml", output, prepared_inputs=[source])
    assert result["stack_initialization"] is not None
    assert torch.equal(result["stage_a_axisangle_final"], result["model"].rigid_psf.axisangle_init)
    payload = json.loads((output / "stack_initialization_poses.json").read_text())
    assert len(payload["stacks"]) == 1
