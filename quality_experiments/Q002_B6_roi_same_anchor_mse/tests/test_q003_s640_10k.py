import csv
import json

import numpy as np
import pytest


def test_q003_10k_route_keeps_s640_accounting_and_primary_stage_schedule():
    from quality_experiments.Q002_B6_roi_same_anchor_mse.contracts import Q003S640_10K_EXPERIMENT_ID, build_q002_route_config
    from quality_experiments.Q002_B6_roi_same_anchor_mse.training import q003_10k_training_log_fields, q003_10k_training_plan

    route = build_q002_route_config({"training": {}, "decoder": {}}, cropped_prepared_root="/cropped", experiment_id=Q003S640_10K_EXPERIMENT_ID, anchor_batch_size=640)
    assert route["route"]["sampling"] == {"anchor_batch_size": 640, "weights_per_anchor": 10, "signal_residual_count": 6400, "shared_training_psf_samples": 8, "data_psf_inr_location_count": 5120, "B6_scalar_batch_size": 640}
    assert route["route"]["fingerprint_cosine"] == {"enabled": True, "weight": 1.0, "epsilon": 1.0e-8}
    plan = q003_10k_training_plan({"stage_a_iterations": 2000, "stage_b_iterations": 4000, "batch_size": 640, "psf_samples": 8, "seed": 20260911, "scheduler_milestones": [0.5], "scheduler_gamma": 0.1})
    assert [(stage["label"], stage["iterations"], stage["new_optimizer"]) for stage in plan["stages"]] == [("A", 2000, True), ("B", 4000, True), ("B_extension", 4000, False)]
    assert plan["primary_checkpoint_iteration"] == 6000
    assert plan["final_checkpoint_iteration"] == 10000
    assert plan["stage_b_scheduler_milestones"] == [2000]
    assert {"stage", "iteration", "anchor_batch_size", "signal_residual_count", "regularization_application_count"}.issubset(q003_10k_training_log_fields())


def test_q003_10k_rejects_6400_anchor_batch_and_preserves_original_stage_b_milestones():
    from quality_experiments.Q002_B6_roi_same_anchor_mse.contracts import Q003S640_10K_EXPERIMENT_ID, build_q002_route_config
    from quality_experiments.Q002_B6_roi_same_anchor_mse.training import q003_10k_training_plan

    with pytest.raises(ValueError, match="fixed anchor_batch_size"):
        build_q002_route_config({"training": {}, "decoder": {}}, cropped_prepared_root="/cropped", experiment_id=Q003S640_10K_EXPERIMENT_ID, anchor_batch_size=6400)
    training = {"stage_a_iterations": 2000, "stage_b_iterations": 4000, "batch_size": 640, "psf_samples": 8, "seed": 20260911, "scheduler_milestones": [0.25, 0.5, 0.75], "scheduler_gamma": 0.1}
    assert q003_10k_training_plan(training)["stage_b_scheduler_milestones"] == [1000, 2000, 3000]


def test_q003_four_way_support_requires_every_method_and_all_ten_weights():
    from quality_experiments.Q002_B6_roi_same_anchor_mse.metrics import strict_fingerprint_four_way_common_support, strict_q003_four_way_common_support

    values = np.ones((2, 2), np.float32)
    masks = np.ones((2, 2), bool)
    q003_10k = values.copy(); q003_10k[1, 1] = np.nan
    assert strict_q003_four_way_common_support(values, values, values, values, q003_10k, masks, masks, masks, masks, masks).tolist() == [[True, True], [True, False]]
    fingerprints = np.ones((10, 2, 2), np.float32)
    q003_masks = np.ones((10, 2, 2), bool); q003_masks[9, 0, 1] = False
    assert strict_fingerprint_four_way_common_support(fingerprints, fingerprints, fingerprints, fingerprints, np.ones_like(fingerprints, bool), np.ones_like(fingerprints, bool), np.ones_like(fingerprints, bool), q003_masks).tolist() == [[True, False], [True, True]]


def test_convergence_artifacts_are_post_training_with_fixed_boundaries(tmp_path):
    from quality_experiments.Q002_B6_roi_same_anchor_mse.training_curves import generate_convergence_artifacts

    training = tmp_path / "training_log.csv"
    monitor = tmp_path / "monitor_log.csv"
    fields = ["stage", "iteration", "joint_signal_mse", "weighted_fingerprint_cosine_loss", "total_data_loss", "total", "reg_t1", "reg_t2", "reg_b1", "amplitude_reg_t1", "amplitude_reg_t2", "transformation", "lr_encoding", "lr_network", "lr_rigid"]
    with training.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for iteration in range(1, 10001):
            stage = "A" if iteration <= 2000 else "B" if iteration <= 6000 else "B_extension"
            writer.writerow({field: 1.0 for field in fields} | {"stage": stage, "iteration": iteration, "lr_rigid": "" if stage == "A" else 0.01})
    with monitor.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["iteration", "stage", "monitor_joint_mse", "monitor_cosine_similarity"]); writer.writeheader()
        for iteration in (100, 6000, 10000): writer.writerow({"iteration": iteration, "stage": "B_extension" if iteration > 6000 else "B", "monitor_joint_mse": 1.0, "monitor_cosine_similarity": 0.5})
    result = generate_convergence_artifacts(training, monitor, tmp_path / "training_curves")
    assert result["summary"]["total"]["mean_5901_6000"] == 1.0
    manifest = json.loads((tmp_path / "training_curves/convergence_plot_manifest.json").read_text())
    assert manifest["stage_boundaries"] == {"stage_a_to_b": 2000, "primary_checkpoint": 6000, "final_checkpoint": 10000}
    for name in ("data_loss_curves", "total_loss_curve", "regularization_curves", "monitor_convergence", "learning_rates"):
        assert (tmp_path / f"training_curves/{name}.png").is_file()
        assert (tmp_path / f"training_curves/{name}.pdf").is_file()


def test_visualization_selection_and_display_ranges_are_deterministic_and_non_metric_driven():
    from quality_experiments.Q002_B6_roi_same_anchor_mse.visualization import deterministic_representatives, visualization_display_contract

    identities = [("sax", 2, 7, 8), ("sax", 0, 1, 2), ("2ch", 1, 3, 4), ("4ch", 3, 5, 6)]
    assert deterministic_representatives(identities, limit=3) == [("2ch", 1, 3, 4), ("4ch", 3, 5, 6), ("sax", 0, 1, 2)]
    display = visualization_display_contract()
    assert display["selection_rule"] == "lexicographic_stack_group_anchor_identity"
    assert display["per_method_autoscaling"] is False


def test_q003_10k_cli_and_server_routes_are_distinct_and_protected():
    from pathlib import Path

    from quality_experiments.Q002_B6_roi_same_anchor_mse.evaluate_q003_s640_10k import build_parser as evaluation_parser
    from quality_experiments.Q002_B6_roi_same_anchor_mse.run_q003_s640_10k_reconstruction import build_parser as reconstruction_parser

    args = reconstruction_parser().parse_args(["--cropped-prepared-root", "/crop", "--b6-model", "/b6.pt", "--signal-simulator", "/decoder.pt", "--output", "/out"])
    assert args.experiment_id == "Q003S640_B6_roi_same_anchor_mse_plus_cosine_10k" and args.anchor_batch_size == 640
    assert evaluation_parser().parse_args(["--run-root", "/q003", "--q002-s640-run-root", "/q002", "--b6-map-root", "/b6", "--b6-d2-root", "/d2", "--reference-root", "/ref", "--preprocessed-root", "/pre", "--output", "/out"]).run_root == "/q003"
    scripts = Path(__file__).resolve().parents[2] / "server_commands"
    run, evaluate = (scripts / "run_Q003S640_10K_CYJ.sh").read_text(), (scripts / "evaluate_Q003S640_10K_CYJ.sh").read_text()
    assert "Q003S640_B6_roi_same_anchor_mse_plus_cosine_10k_v1" in run
    assert "Q002-S640 parent checkpoint SHA256 mismatch" in run
    assert "Refusing non-empty output" in run and "Refusing existing log" in run
    assert "formal_q003s640_10k_four_way_common_support" in evaluate
    assert "Q002-S640 parent checkpoint SHA256 mismatch" in evaluate
    assert "rm -rf" not in run + evaluate
