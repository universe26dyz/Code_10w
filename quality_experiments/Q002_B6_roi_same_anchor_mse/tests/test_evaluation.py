import numpy as np


def test_strict_common_support_requires_reference_b6_q002_and_finite_values():
    from quality_experiments.Q002_B6_roi_same_anchor_mse.metrics import strict_q002_common_support

    support = strict_q002_common_support(np.array([1.0, np.nan]), np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([True, True]), np.array([True, True]))
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
