"""Contracts for the read-only old-quality versus L100 support audit."""
from __future__ import annotations

import numpy as np
import pytest
import json

from paper_experiments.joint_vs_2dfit.phase2_clean import mask_sensitivity_audit as audit


def test_same_prediction_has_different_metrics_when_only_support_changes() -> None:
    reference = np.array([[0.0, 0.0, 0.0, 0.0]])
    prediction = np.array([[0.0, 0.0, 10.0, 10.0]])
    old_common = np.array([[True, True, False, False]])
    l100 = np.array([[False, False, True, True]])

    result = audit.compare_mask_sensitivity(reference, prediction, old_common, l100)

    assert result["old_quality_common"]["metrics"]["rmse"] == pytest.approx(0.0)
    assert result["phase4r_l100"]["metrics"]["rmse"] == pytest.approx(10.0)
    assert result["intersection"]["metrics"] is None
    assert result["old_only"]["metrics"]["support_voxels"] == 2
    assert result["l100_only"]["metrics"]["support_voxels"] == 2


def test_intersection_uses_one_fixed_finite_base_support() -> None:
    reference = np.array([[1.0, np.nan, 3.0]])
    prediction = np.array([[2.0, 20.0, np.nan]])
    mask = np.array([[True, True, True]])

    result = audit.compare_mask_sensitivity(reference, prediction, mask, mask)

    assert result["old_quality_common"]["metrics"]["support_voxels"] == 1
    assert result["phase4r_l100"]["metrics"]["support_voxels"] == 1
    assert result["intersection"]["metrics"]["support_voxels"] == 1
    assert result["mask_overlap"]["jaccard"] == pytest.approx(1.0)
    assert result["mask_overlap"]["dice"] == pytest.approx(1.0)


def test_rejects_misaligned_support_arrays() -> None:
    with pytest.raises(ValueError, match="share one shape"):
        audit.compare_mask_sensitivity(
            np.zeros((2, 2)), np.zeros((2, 2)), np.ones((2, 2), bool), np.ones((2, 3), bool),
        )


def test_published_audit_records_explicit_inputs_and_refuses_overwrite(tmp_path) -> None:
    inputs = tmp_path / "inputs"; inputs.mkdir()
    np.savez_compressed(
        inputs / "t1_2ch_group000.npz",
        reference=np.array([[0.0, 0.0]]), prediction=np.array([[0.0, 10.0]]),
        old_quality_common=np.array([[True, False]]), phase4r_l100=np.array([[False, True]]),
        source_support=np.array([[True, True]]),
    )
    output = audit.run_mask_sensitivity_audit(inputs, tmp_path / "audit")
    manifest = json.loads((output / "manifest.json").read_text())
    aggregate = json.loads((output / "aggregate_metrics.json").read_text())
    assert manifest["no_reconstruction"] is True
    assert manifest["inputs"][0]["path"].endswith("t1_2ch_group000.npz")
    assert aggregate["domains"]["old_quality_common"]["metrics"]["rmse"] == pytest.approx(0.0)
    assert aggregate["domains"]["phase4r_l100"]["metrics"]["rmse"] == pytest.approx(10.0)
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        audit.run_mask_sensitivity_audit(inputs, output)
