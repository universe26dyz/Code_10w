import torch


class _Joint:
    def __init__(self):
        self.xyz = torch.arange(18, dtype=torch.float32).reshape(6, 3)
        self.observed = torch.arange(60, dtype=torch.float32).reshape(6, 10)
        self.group_idx = torch.arange(6, dtype=torch.long)
        self.stack_idx = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        self.timing = torch.zeros((6, 9))
        self.row_col = torch.zeros((6, 2), dtype=torch.long)


def test_anchor_sampler_uses_replacement_and_never_expands_to_640_anchors():
    from quality_experiments.Q002_B6_roi_same_anchor_mse.training import JointAnchorSampler

    sampler = JointAnchorSampler(_Joint(), seed=7)
    batch = sampler.sample(64)
    assert batch["xyz"].shape == (64, 3)
    assert batch["observed"].shape == (64, 10)
    assert batch["effective_scalar_budget"] == 640
    assert batch["anchor_idx"].unique().numel() < 64


def test_route_controlled_anchor_sampler_keeps_q002_64_and_q002_s640_distinct():
    from quality_experiments.Q002_B6_roi_same_anchor_mse.training import JointAnchorSampler

    q002_64 = JointAnchorSampler(_Joint(), seed=7, anchor_batch_size=64).sample()
    q002_s640 = JointAnchorSampler(_Joint(), seed=7, anchor_batch_size=640).sample()
    assert q002_64["xyz"].shape == (64, 3)
    assert q002_64["anchor_batch_size"] == 64
    assert q002_64["signal_residual_count"] == 640
    assert q002_s640["observed"].shape == (640, 10)
    assert q002_s640["anchor_batch_size"] == 640
    assert q002_s640["weights_per_anchor"] == 10
    assert q002_s640["signal_residual_count"] == 6400


def test_s640_control_accounting_uses_k8_and_preserves_b6_regularization_budget():
    from quality_experiments.Q002_B6_roi_same_anchor_mse.training import q002_training_controls

    controls = q002_training_controls(
        {"stage_a_iterations": 2000, "stage_b_iterations": 4000, "batch_size": 640, "psf_samples": 8, "seed": 20260911},
        anchor_batch_size=640,
    )
    assert controls["anchor_batch_size"] == 640
    assert controls["weights_per_anchor"] == 10
    assert controls["signal_residual_count"] == 6400
    assert controls["training_psf_samples"] == 8
    assert controls["data_psf_inr_location_count"] == 5120
    assert controls["B6_scalar_batch_size"] == 640
    assert controls["regularization_candidate_count"] == 640
    assert controls["regularization_effective_point_count"] == 256
    assert controls["regularization_application_count_per_optimizer_step"] == 1


def test_joint_vector_mse_matches_hand_formula_and_uses_b6_stack_weights():
    from quality_experiments.Q002_B6_roi_same_anchor_mse.training import joint_vector_mse

    prediction = torch.tensor([[1.0] * 10, [3.0] * 10])
    observed = torch.tensor([[0.0] * 10, [1.0] * 10])
    actual = joint_vector_mse(prediction, observed, torch.tensor([0, 1]), {0: 2.0, 1: 0.5})
    assert torch.allclose(actual, torch.tensor((2.0 * 1.0 + 0.5 * 4.0) / 2.0))


def test_q002_control_contract_preserves_b6_stage_and_scalar_budget():
    from quality_experiments.Q002_B6_roi_same_anchor_mse.training import q002_training_controls

    controls = q002_training_controls({"stage_a_iterations": 2000, "stage_b_iterations": 4000, "batch_size": 640, "psf_samples": 8, "seed": 20260911})
    assert controls == {"stage_a_iterations": 2000, "stage_b_iterations": 4000, "effective_scalar_budget": 640, "anchor_batch_size": 64, "weights_per_anchor": 10, "psf_samples": 8, "seed": 20260911}


def test_q002_64_route_remains_explicit_and_s640_route_is_immutable():
    from quality_experiments.Q002_B6_roi_same_anchor_mse.contracts import (
        Q002_EXPERIMENT_ID,
        Q002S640_EXPERIMENT_ID,
        build_q002_route_config,
    )

    base = {"training": {}, "decoder": {}}
    q002_64 = build_q002_route_config(base, cropped_prepared_root="/cropped", experiment_id=Q002_EXPERIMENT_ID, anchor_batch_size=64)
    s640 = build_q002_route_config(base, cropped_prepared_root="/cropped", experiment_id=Q002S640_EXPERIMENT_ID, anchor_batch_size=640)
    assert q002_64["route"]["sampling"]["anchor_batch_size"] == 64
    assert s640["route"]["sampling"] == {
        "anchor_batch_size": 640,
        "weights_per_anchor": 10,
        "signal_residual_count": 6400,
        "shared_training_psf_samples": 8,
        "data_psf_inr_location_count": 5120,
        "B6_scalar_batch_size": 640,
    }


def test_q002_runner_exposes_explicit_cropped_only_arguments():
    from quality_experiments.Q002_B6_roi_same_anchor_mse.run_q002_reconstruction import build_parser

    args = build_parser().parse_args(["--cropped-prepared-root", "/crop", "--b6-model", "/b6.pt", "--signal-simulator", "/approved.pth", "--output", "/out"])
    assert args.device == "cuda:0" and args.cropped_prepared_root == "/crop"
    s640 = build_parser().parse_args(["--cropped-prepared-root", "/crop", "--b6-model", "/b6.pt", "--signal-simulator", "/approved.pth", "--output", "/out", "--experiment-id", "Q002S640_B6_roi_same_anchor_mse", "--anchor-batch-size", "640"])
    assert s640.experiment_id == "Q002S640_B6_roi_same_anchor_mse" and s640.anchor_batch_size == 640


def test_regularization_batch_reuses_b6_scalar_sampler_with_640_candidates():
    from quality_experiments.Q002_B6_roi_same_anchor_mse.training import regularization_batch_from_b6_scalar_support

    class ScalarSampler:
        def sample(self, n):
            assert n == 640
            return {"xyz": torch.zeros((n, 3)), "group_idx": torch.zeros(n, dtype=torch.long)}
    result = regularization_batch_from_b6_scalar_support(ScalarSampler(), n_points=256)
    assert result["candidate_count"] == 640
    assert result["effective_point_count"] == 256
    assert result["sampling_source"] == "B6 scalar cropped support"
