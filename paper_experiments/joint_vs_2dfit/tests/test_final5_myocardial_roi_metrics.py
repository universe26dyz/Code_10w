"""Contracts for the Final5-only local native-SAX myocardial ROI report."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def test_myocardial_roi_variants_keep_mm_and_one_pixel_cores_distinct() -> None:
    """Catches accidentally substituting either core definition for the other."""
    from paper_experiments.joint_vs_2dfit.aha.final5_myocardial_roi_metrics import build_myocardial_rois

    outer = np.zeros((9, 9), bool); outer[1:8, 1:8] = True
    rois = build_myocardial_rois(outer, np.zeros_like(outer), spacing_rc_mm=(0.5, 1.0))

    assert set(rois) == {"myocardium_full", "myocardium_core_legacy", "myocardium_core_1px"}
    assert int(rois["myocardium_full"].sum()) == 49
    assert int(rois["myocardium_core_legacy"].sum()) == 15
    assert int(rois["myocardium_core_1px"].sum()) == 25


def test_paired_roi_statistics_use_one_common_support_for_all_methods() -> None:
    """Catches method-private availability masks in myocardial comparisons."""
    from paper_experiments.joint_vs_2dfit.aha.final5_myocardial_roi_metrics import paired_roi_statistics

    reference = np.array([[10.0, 20.0], [30.0, 40.0]])
    proposed = np.array([[11.0, 21.0], [np.nan, 50.0]])
    fit_first = np.array([[8.0, np.nan], [28.0, 42.0]])
    result = paired_roi_statistics(reference=reference, proposed=proposed, fit_first=fit_first,
                                   proposed_valid=np.ones((2, 2), bool),
                                   fit_first_valid=np.ones((2, 2), bool), roi=np.ones((2, 2), bool))

    assert result["common_support_voxels"] == 2
    assert result["reference"] == {"n": 2, "mean": 25.0, "sample_sd": pytest.approx(21.2132034356)}
    assert result["proposed"]["n"] == result["fit_first"]["n"] == 2
    assert result["proposed"]["bias"] == pytest.approx(5.5)
    assert result["proposed"]["mae"] == pytest.approx(5.5)
    assert result["fit_first"]["bias"] == pytest.approx(0.0)
    assert result["fit_first"]["rmse"] == pytest.approx(2.0)


def test_mapped_roi_plane_preserves_declared_group_identity_and_discrete_orientation() -> None:
    """Catches ignored group IDs or an interpolated/incorrect ROI orientation."""
    from paper_experiments.joint_vs_2dfit.aha.final5_myocardial_roi_metrics import mapped_roi_plane

    outer = np.zeros((2, 3, 1), np.uint8); outer[:, :, 0] = [[1, 0, 0], [1, 1, 0]]
    inner = np.zeros_like(outer)
    entry = {"level": "basal", "native_group_idx": 7, "original_group_idx": 3,
             "roi_slice_index": 0, "roi_transform": "transpose"}
    result = mapped_roi_plane(SimpleNamespace(get_fdata=lambda: outer), SimpleNamespace(get_fdata=lambda: inner),
                              entry=entry, expected_shape=(3, 2), spacing_rc_mm=(1.0, 1.0))

    assert result["native_group_idx"] == 7
    assert result["original_group_idx"] == 3
    assert result["roi_transform"] == "transpose"
    assert np.array_equal(result["rois"]["myocardium_full"], np.array([[True, True], [False, True], [False, False]]))


def test_roi_output_must_be_new_and_outside_the_final5_root(tmp_path) -> None:
    """Catches writing a local ROI report into or over a formal Final5 tree."""
    from paper_experiments.joint_vs_2dfit.aha.final5_myocardial_roi_metrics import require_new_external_output

    final_root = tmp_path / "final5"; final_root.mkdir()
    with pytest.raises(ValueError, match="outside"):
        require_new_external_output(final_root, final_root / "downstream_roi")
    output = tmp_path / "local_reports" / "CYJ" / "L100"
    assert require_new_external_output(final_root, output) == output
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        require_new_external_output(final_root, output)


def test_dyl_2_without_its_own_roi_is_pending_not_fallback(tmp_path) -> None:
    """Catches any fallback from DYL_2 to DYL's manual ROI."""
    from paper_experiments.joint_vs_2dfit.aha.final5_myocardial_roi_metrics import assess_final5_myocardial_roi_readiness

    readiness = assess_final5_myocardial_roi_readiness(
        tmp_path / "final5", "DYL_2", "L100", tmp_path / "subjects")

    assert readiness["status"] == "ROI_PENDING"
