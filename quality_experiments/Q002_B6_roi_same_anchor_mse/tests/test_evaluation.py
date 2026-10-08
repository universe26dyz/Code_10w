import numpy as np
import pytest


def test_strict_common_support_requires_reference_b6_q002_and_finite_values():
    from quality_experiments.Q002_B6_roi_same_anchor_mse.metrics import strict_q002_common_support

    support = strict_q002_common_support(np.array([1.0, np.nan]), np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([True, True]), np.array([True, True]), np.array([True, True]))
    assert support.tolist() == [True, False]


def test_q002_evaluation_plan_keeps_cosine_read_only_and_k128_provenance():
    from quality_experiments.Q002_B6_roi_same_anchor_mse.evaluation import evaluation_plan

    plan = evaluation_plan("checkpoint")
    assert plan["central_no_map_psf"]["psf_samples"] == 1
    assert plan["signal_psf_K8"]["psf_samples"] == 8
    assert plan["formal_export_psf128"] == {"output_resolution_mm": 1.0, "output_psf_factor": 1.0, "psf_samples": 128, "seed": 20260911, "checkpoint_sha256": "checkpoint"}
    assert plan["fingerprint_cosine"]["training_objective"] is False


def test_signal_fingerprint_common_support_requires_all_ten_weights():
    from quality_experiments.Q002_B6_roi_same_anchor_mse.metrics import strict_fingerprint_common_support

    b6 = np.ones((10, 2, 2), np.float32)
    q002 = np.ones((10, 2, 2), np.float32)
    masks = np.ones((10, 2, 2), bool)
    masks[4, 0, 1] = False
    support = strict_fingerprint_common_support(b6, q002, masks, masks)
    assert support.tolist() == [[True, False], [True, True]]


def test_map_support_requires_reference_mask_and_weight_zero_group_alignment():
    from quality_experiments.Q002_B6_roi_same_anchor_mse.evaluate_q002 import _canonical_weight_zero_maps
    from quality_experiments.Q002_B6_roi_same_anchor_mse.metrics import strict_q002_common_support

    archive = {"group_idx": np.array([1, 0, 1, 0]), "weight_idx": np.array([0, 0, 1, 1]), "t1_ms": np.arange(16, dtype=np.float32).reshape(4, 2, 2), "masks": np.ones((4, 2, 2), bool)}
    maps, masks = _canonical_weight_zero_maps(archive, "t1_ms", 2)
    assert maps[:, 0, 0].tolist() == [4.0, 0.0] and masks.shape == (2, 2, 2)
    support = strict_q002_common_support(np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2), bool), np.ones((2, 2), bool), np.array([[True, False], [True, True]]))
    assert support.tolist() == [[True, False], [True, True]]


def test_signal_identity_validator_rejects_duplicate_missing_and_extra_pairs():
    from quality_experiments.Q002_B6_roi_same_anchor_mse.evaluate_q002 import _signal_identity_index
    archive = {"group_idx": np.repeat(np.arange(2), 10), "weight_idx": np.tile(np.arange(10), 2)}
    assert len(_signal_identity_index(archive, "fixture")) == 20
    duplicate = dict(archive); duplicate["weight_idx"] = duplicate["weight_idx"].copy(); duplicate["weight_idx"][1] = 0
    with pytest.raises(ValueError, match="duplicate|weights"):
        _signal_identity_index(duplicate, "fixture")
