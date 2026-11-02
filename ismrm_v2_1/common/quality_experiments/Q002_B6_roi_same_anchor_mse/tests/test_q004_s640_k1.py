from pathlib import Path
import csv
import json

import numpy as np
import pytest
import torch


def _b6_training():
    return {"stage_a_iterations": 2000, "stage_b_iterations": 4000, "batch_size": 640, "psf_samples": 8, "seed": 20260911}


def test_q004_route_is_s640_mse_only_central_k1_and_historical_routes_stay_k8():
    from quality_experiments.Q002_B6_roi_same_anchor_mse.contracts import Q002S640_EXPERIMENT_ID, Q003S640_EXPERIMENT_ID, Q003S640_10K_EXPERIMENT_ID, Q004S640_K1_EXPERIMENT_ID, build_q002_route_config

    base = {"training": {}, "decoder": {}}
    q004 = build_q002_route_config(base, cropped_prepared_root="/cropped", experiment_id=Q004S640_K1_EXPERIMENT_ID, anchor_batch_size=640)
    assert q004["route"]["sampling"] == {"anchor_batch_size": 640, "weights_per_anchor": 10, "signal_residual_count": 6400, "shared_training_psf_samples": 1, "data_psf_inr_location_count": 640, "B6_scalar_batch_size": 640}
    assert "fingerprint_cosine" not in q004["route"]
    assert build_q002_route_config(base, cropped_prepared_root="/cropped", experiment_id=Q002S640_EXPERIMENT_ID, anchor_batch_size=640)["route"]["sampling"]["shared_training_psf_samples"] == 8
    assert build_q002_route_config(base, cropped_prepared_root="/cropped", experiment_id=Q003S640_EXPERIMENT_ID, anchor_batch_size=640)["route"]["sampling"]["shared_training_psf_samples"] == 8
    assert build_q002_route_config(base, cropped_prepared_root="/cropped", experiment_id=Q003S640_10K_EXPERIMENT_ID, anchor_batch_size=640)["route"]["sampling"]["shared_training_psf_samples"] == 8
    with pytest.raises(ValueError, match="fixed anchor_batch_size"):
        build_q002_route_config(base, cropped_prepared_root="/cropped", experiment_id=Q004S640_K1_EXPERIMENT_ID, anchor_batch_size=6400)


def test_q004_controls_and_sampler_account_for_k1_without_changing_b6_regularization():
    from quality_experiments.Q002_B6_roi_same_anchor_mse.training import JointAnchorSampler, q002_training_controls

    controls = q002_training_controls(_b6_training(), anchor_batch_size=640, training_psf_samples=1)
    assert controls["training_psf_samples"] == 1
    assert controls["data_psf_inr_location_count"] == 640
    assert controls["regularization_candidate_count"] == 640
    assert controls["regularization_effective_point_count"] == 256
    assert controls["regularization_application_count_per_optimizer_step"] == 1
    dataset = type("Joint", (), {"xyz": torch.zeros((2, 3)), "observed": torch.zeros((2, 10)), "group_idx": torch.zeros(2, dtype=torch.long), "stack_idx": torch.zeros(2, dtype=torch.long), "timing": torch.zeros((2, 9)), "row_col": torch.zeros((2, 2), dtype=torch.long)})()
    batch = JointAnchorSampler(dataset, seed=7, anchor_batch_size=640, training_psf_samples=1).sample()
    assert batch["observed"].shape == (640, 10)
    assert batch["signal_residual_count"] == 6400 and batch["data_psf_inr_location_count"] == 640


def test_k1_rigid_psf_is_exact_central_transform_and_never_draws_gaussian(monkeypatch):
    from trad.modules.module_06_rigid_psf.rigid_psf_forward import GroupRigidPSF

    psf = GroupRigidPSF(torch.tensor([[0.0, 0.0, 0.0, 2.0, -1.0, 0.5]]), torch.ones((1, 3)))
    local, groups = torch.tensor([[1.0, 3.0, 5.0]]), torch.tensor([0], dtype=torch.long)
    monkeypatch.setattr(torch, "randn", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("K1 must not sample Gaussian noise")))
    actual = psf.sample_local_then_transform(local, groups, n_samples=1)
    assert torch.equal(actual[:, 0, :], psf.transform_local_to_ras(local, groups))


def test_q004_k1_monitor_forward_is_repeatable_and_uses_640_spatial_locations():
    from torch import nn
    from trad.modules.module_06_rigid_psf.rigid_psf_forward import TradQuantitativeForward

    class PSF(nn.Module):
        def __init__(self): super().__init__(); self.calls = 0
        def sample_local_then_transform(self, xyz, group_idx, n_samples):
            assert n_samples == 1; self.calls += 1; return xyz[:, None, :]
    class INR(nn.Module):
        def __init__(self): super().__init__(); self.last_xyz = None
        def forward(self, xyz):
            self.last_xyz = xyz; value = xyz[:, 0]; return {"t1_ms": value, "t2_ms": value, "b1": value, "amplitude": torch.ones_like(value)}
    class Decoder(nn.Module):
        def forward(self, t1, t2, b1, timing, protocol, normalize=True): return t1[:, None].expand(-1, 10)

    psf, inr = PSF(), INR()
    forward = TradQuantitativeForward(inr, Decoder(), psf, protocol=object())
    batch = (torch.arange(640 * 3, dtype=torch.float32).reshape(640, 3), torch.zeros(640, dtype=torch.long), torch.zeros((640, 9)))
    first = forward.forward_fingerprint(*batch, n_psf_samples=1)
    second = forward.forward_fingerprint(*batch, n_psf_samples=1)
    assert first.shape == (640, 10) and torch.equal(first, second)
    assert psf.calls == 2 and inr.last_xyz.shape == (640, 3)


def test_q004_mse_only_matches_q002_s640_value_and_gradients():
    from quality_experiments.Q002_B6_roi_same_anchor_mse.training import joint_fingerprint_objective, joint_vector_mse

    prediction = torch.tensor([[1.0] * 10, [2.0] * 10], requires_grad=True)
    observed = torch.tensor([[0.0] * 10, [1.0] * 10])
    stack_idx, weights = torch.tensor([0, 1]), {0: 2.0, 1: 0.5}
    expected = joint_vector_mse(prediction, observed, stack_idx, weights); expected.backward(); gradient = prediction.grad.detach().clone(); prediction.grad = None
    q004 = joint_fingerprint_objective(prediction, observed, stack_idx, weights, cosine_weight=0.0, cosine_epsilon=1e-8)["total_data_loss"]
    q004.backward()
    assert torch.equal(q004, expected.detach()) and torch.equal(prediction.grad, gradient)


def test_q004_strict_three_way_support_requires_reference_b6_q002_and_q004_finite():
    from quality_experiments.Q002_B6_roi_same_anchor_mse.metrics import strict_q004_three_way_common_support

    values, masks = np.ones((2, 2)), np.ones((2, 2), bool)
    q004 = values.copy(); q004[1, 1] = np.nan
    assert strict_q004_three_way_common_support(values, values, values, q004, masks, masks, masks, masks).tolist() == [[True, True], [True, False]]


def test_q004_convergence_artifacts_are_post_training_and_mark_2000_6000(tmp_path):
    from quality_experiments.Q002_B6_roi_same_anchor_mse.training_curves import generate_q004_convergence_artifacts

    fields = ["stage", "iteration", "joint_signal_mse", "total", "reg_t1", "reg_t2", "reg_b1", "amplitude_reg_t1", "amplitude_reg_t2", "transformation", "lr_encoding", "lr_network", "lr_rigid"]
    training, monitor = tmp_path / "training_log.csv", tmp_path / "monitor_log.csv"
    with training.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for iteration in range(1, 6001): writer.writerow({field: 1.0 for field in fields} | {"stage": "A" if iteration <= 2000 else "B", "iteration": iteration, "lr_rigid": "" if iteration <= 2000 else 0.01})
    with monitor.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["iteration", "stage", "monitor_mse"]); writer.writeheader(); writer.writerow({"iteration": 6000, "stage": "B", "monitor_mse": 1.0})
    generate_q004_convergence_artifacts(training, monitor, tmp_path / "training_curves")
    assert (tmp_path / "training_curves/data_loss_curve.png").is_file()
    assert json.loads((tmp_path / "training_curves/convergence_plot_manifest.json").read_text())["stage_boundaries"] == {"stage_a_to_b": 2000, "final_checkpoint": 6000}


def test_q004_visualization_reuses_fixed_ranges_and_deterministic_selection(tmp_path):
    from quality_experiments.Q002_B6_roi_same_anchor_mse.visualization import deterministic_representatives, render_map_montages, visualization_display_contract

    contract = visualization_display_contract()
    assert contract["per_method_autoscaling"] is False
    assert deterministic_representatives([("sax", 2, 3, 4), ("sax", 1, 9, 9), ("sax", 2, 3, 4)], limit=2) == [("sax", 1, 9, 9), ("sax", 2, 3, 4)]
    maps = np.ones((2, 2))
    outputs = render_map_montages([{"parameter": "T1", "stack": "sax", "group_idx": 0, "reference": maps, "predictions": {"FrozenMLP_B6": maps, "Q002_S640": maps, "Q004_S640_K1": maps}, "support": np.ones((2, 2), bool)}], tmp_path / "maps", methods=("FrozenMLP_B6", "Q002_S640", "Q004_S640_K1"))
    assert all(path.is_file() for path in outputs)


def test_q004_derived_k1_comparators_write_only_under_new_evaluation_root(tmp_path, monkeypatch):
    import argparse
    import quality_experiments.Q002_B6_roi_same_anchor_mse.evaluate_q004_s640_k1 as evaluator

    calls = []
    monkeypatch.setattr(evaluator, "cropped_observation_paths", lambda _root: ["cropped.npz"])
    monkeypatch.setattr(evaluator, "QuantPointDataset", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(evaluator, "_load_model", lambda *_args, **_kwargs: (object(), object()))
    monkeypatch.setattr(evaluator, "export_native_plane_reprojections", lambda _model, _space, _inputs, root, **kwargs: calls.append((Path(root), kwargs)))
    output, historical_q002, historical_q004 = tmp_path / "evaluation", tmp_path / "q002_history", tmp_path / "q004_history"
    roots = evaluator._derive_central_k1(argparse.Namespace(cropped_prepared_root="/crop", device="cpu", b6_model="/b6.pt"), output, historical_q004 / "model.pt", historical_q002 / "model.pt")
    assert set(roots) == {"B6", "Q002_S640", "Q004_S640_K1"}
    assert all(root.is_relative_to(output / "derived_k1") and kwargs["output_psf"] == {"enabled": True, "n_samples": 1} for root, kwargs in calls)


def test_q004_cli_and_server_scripts_are_dedicated_and_protected():
    from quality_experiments.Q002_B6_roi_same_anchor_mse.evaluate_q004_s640_k1 import build_parser as evaluator_parser
    from quality_experiments.Q002_B6_roi_same_anchor_mse.run_q004_s640_k1_reconstruction import build_parser

    args = build_parser().parse_args(["--cropped-prepared-root", "/crop", "--b6-model", "/b6.pt", "--signal-simulator", "/decoder.pt", "--output", "/out"])
    assert args.experiment_id == "Q004S640_K1_B6_roi_same_anchor_mse" and args.anchor_batch_size == 640
    assert evaluator_parser().parse_args(["--run-root", "/q004", "--q002-s640-run-root", "/q002", "--b6-model", "/b6.pt", "--b6-map-root", "/maps", "--b6-d2-root", "/d2", "--cropped-prepared-root", "/crop", "--reference-root", "/ref", "--preprocessed-root", "/pre", "--output", "/out"]).run_root == "/q004"
    scripts = Path(__file__).resolve().parents[2] / "server_commands"
    run, evaluate = (scripts / "run_Q004S640_K1_CYJ.sh").read_text(), (scripts / "evaluate_Q004S640_K1_CYJ.sh").read_text()
    assert "Q004S640_K1_B6_roi_same_anchor_mse_v1" in run and "training_psf_samples_8_to_central_1_only" in run
    assert "Q002-S640 parent checkpoint SHA256 mismatch" in run and "Refusing non-empty output" in run
    assert "derived_k1" in evaluate and "Refusing non-empty evaluation" in evaluate and "rm -rf" not in run + evaluate
