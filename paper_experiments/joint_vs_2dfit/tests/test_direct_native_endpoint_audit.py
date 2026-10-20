"""CPU-only contracts for direct-native endpoint-audit support semantics."""
from __future__ import annotations

import inspect

import numpy as np

from paper_experiments.joint_vs_2dfit.phase2_clean import direct_native_endpoint_audit as audit


def test_direct_and_volume_metrics_share_one_exact_paired_support() -> None:
    reference = np.array([[1000.0, 1000.0, np.nan]])
    support = np.array([[True, True, True]])
    l100 = np.array([[True, True, True]])
    direct = np.array([[1001.0, np.nan, 999.0]])
    volume = np.array([[1002.0, 1003.0, 998.0]])
    shared = audit.exact_shared_support(reference, support, l100, direct, volume, np.ones((1, 3), bool), np.array([[True, False, True]]))
    assert shared.tolist() == [[True, False, False]]


def test_direct_path_is_model_inr_and_volume_path_only_samples_the_central_nifti() -> None:
    source = inspect.getsource(audit.run_direct_native_endpoint_audit)
    assert "direct_native_world_grid(model, space" in source
    assert "_model_prediction(model, world, field" in source
    assert "_world_sample(central[parameter].get_fdata()" in source
    assert "native_affine" not in source
    assert "model.inr(points)" in inspect.getsource(audit._model_prediction)


def test_output_support_does_not_depend_on_a_private_method_mask() -> None:
    reference = np.ones((2, 2)); source = np.ones((2, 2), bool); l100 = np.ones((2, 2), bool)
    direct = np.ones((2, 2)); volume = np.ones((2, 2))
    direct_valid = np.array([[True, True], [False, True]])
    volume_valid = np.array([[True, False], [True, True]])
    support = audit.exact_shared_support(reference, source, l100, direct, volume, direct_valid, volume_valid)
    assert support.tolist() == [[True, False], [False, True]]
