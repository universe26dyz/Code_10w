"""CPU contracts for the Q002S640_K256 training-sampling variant."""
from __future__ import annotations

import torch
from torch import nn
from pathlib import Path


def _b6_training() -> dict[str, int]:
    return {
        "stage_a_iterations": 2000,
        "stage_b_iterations": 4000,
        "batch_size": 640,
        "psf_samples": 8,
        "seed": 20260911,
    }


def test_k256_route_freezes_q002s640_except_training_psf_sampling() -> None:
    from quality_experiments.Q002_B6_roi_same_anchor_mse.contracts import (
        Q002S640_EXPERIMENT_ID,
        Q002S640_K256_EXPERIMENT_ID,
        build_q002_route_config,
    )

    base = {"training": {}, "decoder": {}}
    parent = build_q002_route_config(base, cropped_prepared_root="/cropped", experiment_id=Q002S640_EXPERIMENT_ID, anchor_batch_size=640)
    k256 = build_q002_route_config(base, cropped_prepared_root="/cropped", experiment_id=Q002S640_K256_EXPERIMENT_ID, anchor_batch_size=640)
    parent_sampling = dict(parent["route"]["sampling"])
    k256_sampling = dict(k256["route"]["sampling"])
    changed_for_k256 = {"shared_training_psf_samples", "data_psf_inr_location_count", "psf_forward_chunk_size"}
    assert {key: value for key, value in k256_sampling.items() if key not in changed_for_k256} == {key: value for key, value in parent_sampling.items() if key not in changed_for_k256}
    assert k256_sampling["shared_training_psf_samples"] == 256
    assert k256_sampling["data_psf_inr_location_count"] == 640 * 256
    assert k256["route"]["parent_experiment"] == Q002S640_EXPERIMENT_ID
    assert k256["route"]["scientific_change"] == "training_psf_samples_8_to_256_only"
    assert k256["route"]["loss"] == "MSE_only"
    comparison = k256["route"]["fit_first_n_samples_semantics"]
    assert comparison["comparison_label"] == "matched per-observation training PSF sample count (K=256)"
    assert comparison["matched_per_observation_training_psf_sample_count"] is True
    assert comparison["per_observation_training_psf_samples"] == {"proposed": 256, "fit_first_nesvor": 256}
    for key in ("same_observation_semantics", "same_forward_model", "same_loss", "same_batch_size", "same_total_sampled_locations_per_optimizer_step", "same_computational_budget"):
        assert comparison[key] is False


def test_k256_controls_and_sampler_record_logical_counts_without_regularization_change() -> None:
    from quality_experiments.Q002_B6_roi_same_anchor_mse.training import JointAnchorSampler, q002_training_controls

    controls = q002_training_controls(_b6_training(), anchor_batch_size=640, training_psf_samples=256)
    assert controls["training_psf_samples"] == 256
    assert controls["data_psf_inr_location_count"] == 163840
    assert controls["psf_forward_chunk_size"] == 32
    assert controls["regularization_candidate_count"] == 640
    assert controls["regularization_effective_point_count"] == 256
    assert controls["regularization_application_count_per_optimizer_step"] == 1

    dataset = type("Joint", (), {
        "xyz": torch.zeros((2, 3)), "observed": torch.zeros((2, 10)),
        "group_idx": torch.zeros(2, dtype=torch.long), "stack_idx": torch.zeros(2, dtype=torch.long),
        "timing": torch.zeros((2, 9)), "row_col": torch.zeros((2, 2), dtype=torch.long),
    })()
    batch = JointAnchorSampler(dataset, seed=7, anchor_batch_size=640, training_psf_samples=256).sample()
    assert batch["anchor_batch_size"] == 640
    assert batch["weights_per_anchor"] == 10
    assert batch["data_psf_inr_location_count"] == 163840


def test_chunked_fingerprint_forward_matches_unchunked_output_loss_and_gradients() -> None:
    from trad.modules.module_06_rigid_psf.rigid_psf_forward import TradQuantitativeForward

    class PSF(nn.Module):
        def sample_local_then_transform(self, xyz, group_idx, n_samples):
            offsets = torch.arange(n_samples, dtype=xyz.dtype, device=xyz.device).view(1, n_samples, 1)
            return xyz[:, None, :] + offsets

    class INR(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.25))

        def forward(self, xyz):
            value = self.scale * xyz[:, 0]
            return {"t1_ms": value, "t2_ms": value + 1.0, "b1": value + 2.0, "amplitude": value + 3.0}

    class Decoder(nn.Module):
        def forward(self, t1, t2, b1, timing, protocol, normalize=True):
            return torch.stack(tuple(t1 + t2 + b1 + float(weight) for weight in range(10)), dim=1)

    torch.manual_seed(7)
    xyz = torch.randn((3, 3))
    groups, timing = torch.zeros(3, dtype=torch.long), torch.zeros((3, 9))
    inr = INR()
    forward = TradQuantitativeForward(inr, Decoder(), PSF(), protocol=object())
    unchunked = forward.forward_fingerprint(xyz, groups, timing, n_psf_samples=5)
    unchunked_loss = unchunked.square().mean(); unchunked_loss.backward()
    unchunked_gradient = inr.scale.grad.detach().clone()
    inr.scale.grad = None
    chunked = forward.forward_fingerprint(xyz, groups, timing, n_psf_samples=5, psf_chunk_size=2)
    chunked_loss = chunked.square().mean(); chunked_loss.backward()
    assert torch.allclose(chunked, unchunked, rtol=1e-6, atol=1e-6)
    assert torch.allclose(chunked_loss, unchunked_loss, rtol=1e-6, atol=1e-6)
    assert torch.allclose(inr.scale.grad, unchunked_gradient, rtol=1e-6, atol=1e-6)


def test_k256_cyj_server_wrappers_are_immutable_and_limit_the_requested_cases() -> None:
    root = Path(__file__).resolve().parents[2] / "server_commands"
    phase2 = Path(__file__).resolve().parents[3] / "paper_experiments" / "joint_vs_2dfit" / "phase2_clean" / "server_commands"
    run = (root / "run_Q002S640_K256_CYJ.sh").read_text()
    aliases = (phase2 / "formalize_q002s640_k256_aliases_cyj.sh").read_text()
    evaluate = (phase2 / "evaluate_q002s640_k256_matched_case.sh").read_text()
    cases = "L100_fold0|L50_fold0|L50_fold1|L25_fold0|L25_fold1|L25_fold2|L25_fold3"
    for text in (run, aliases, evaluate):
        assert cases in text and "L33" not in text and "rm -rf" not in text
    assert "Q002S640_K256" in run and "--anchor-batch-size 640" in run
    assert "alias-q002s640-k256" in aliases and "--endpoint central" in aliases and "--endpoint psf128" in aliases
    assert "FIT_FIRST_OR_Q002S640" in evaluate and "--endpoint" in evaluate
