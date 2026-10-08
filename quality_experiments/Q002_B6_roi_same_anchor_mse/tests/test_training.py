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


def test_q002_runner_exposes_explicit_cropped_only_arguments():
    from quality_experiments.Q002_B6_roi_same_anchor_mse.run_q002_reconstruction import build_parser

    args = build_parser().parse_args(["--cropped-prepared-root", "/crop", "--b6-model", "/b6.pt", "--signal-simulator", "/approved.pth", "--output", "/out"])
    assert args.device == "cuda:0" and args.cropped_prepared_root == "/crop"


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
