"""Contract tests for the Phase-4R Joint-vs-fit-first renderer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.phase2_clean import visualization as viz
from trad.evaluation.scmr.quality_control import load_legacy_scmr_colorbars


def test_exact_scmr_display_and_residual_ranges() -> None:
    assert viz.DISPLAY_RANGES_MS == {"T1": (0.0, 2500.0), "T2": (0.0, 200.0)}
    assert viz.RESIDUAL_RANGES_MS == {"T1": (0.0, 300.0), "T2": (0.0, 30.0)}


def test_exact_vendored_luts_are_reused() -> None:
    expected = load_legacy_scmr_colorbars()
    actual = viz.colorbars()
    assert actual.metadata["mapping_colormaps"] == expected.metadata["mapping_colormaps"]
    assert np.array_equal(actual.mapping["T1"].get_bad(), (0.0, 0.0, 0.0, 1.0))
    assert np.array_equal(actual.mapping["T2"].get_under(), (0.0, 0.0, 0.0, 1.0))


def test_one_crop_and_support_are_used_for_every_panel() -> None:
    support = np.zeros((9, 10), dtype=bool)
    support[3:6, 4:7] = True
    panels, crop = viz.prepare_native_comparison_panels(
        np.arange(90, dtype=float).reshape(9, 10),
        np.full((9, 10), 1000.0),
        np.full((9, 10), 1010.0),
        support,
    )
    assert crop == (0, 9, 1, 10)
    assert {panel.mask.shape for panel in panels.values()} == {(9, 9)}
    assert all(np.array_equal(panel.mask, panels["native"].mask) for panel in panels.values())


def test_representative_group_selection_is_deterministic_and_central() -> None:
    groups = [
        {"group_index": 1, "common_support_voxels": 9},
        {"group_index": 2, "common_support_voxels": 12},
        {"group_index": 3, "common_support_voxels": 12},
        {"group_index": 4, "common_support_voxels": 9},
    ]
    assert viz.select_representative_group(groups) == 2
    assert viz.select_representative_group(list(reversed(groups))) == 2


def test_figure_one_uses_one_world_plane_for_both_methods() -> None:
    native = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)
    proposed = native + 1
    fit_first = native + 2
    result = viz.sample_same_world_through_plane(native, np.eye(4), proposed, np.eye(4), fit_first, np.eye(4), cut_column=2)
    assert np.array_equal(result["world"], result["proposed_world"])
    assert np.array_equal(result["world"], result["fit_first_world"])
    assert result["interpolation"] == {"native": "nearest", "reconstructions": "linear"}


def test_renderer_contains_no_psf_matched_language() -> None:
    source = Path(viz.__file__).read_text(encoding="utf-8").lower()
    assert "psf-matched" not in source
    assert "k=8 matched" not in source


def _fit_layout(root: Path, *, mean: float, scale: float) -> tuple[Path, Path]:
    import nibabel as nib

    normalized, ms = root / "volume_normalized.nii.gz", root / "volume_ms.nii.gz"
    root.mkdir()
    nib.save(nib.Nifti1Image(np.full((2, 2, 2), 0.4), np.eye(4)), normalized)
    nib.save(nib.Nifti1Image(np.full((2, 2, 2), 0.4 * scale), np.eye(4)), ms)
    (root / "result.json").write_text(json.dumps({"output_intensity_mean": mean}))
    (root / "model.pt").write_bytes(b"model")
    return normalized, ms


def test_stale_fit_first_mean_700_is_rejected(tmp_path: Path) -> None:
    normalized, ms = _fit_layout(tmp_path / "fit", mean=700, scale=2500)
    with pytest.raises(ValueError, match="mean-700"):
        viz.validate_fit_first_scale(tmp_path / "fit", "T1", normalized, ms)


@pytest.mark.parametrize(("parameter", "scale"), (("T1", 2500.0), ("T2", 200.0)))
def test_current_fit_first_scale_is_accepted(tmp_path: Path, parameter: str, scale: float) -> None:
    normalized, ms = _fit_layout(tmp_path / parameter, mean=0, scale=scale)
    report = viz.validate_fit_first_scale(tmp_path / parameter, parameter, normalized, ms)
    assert report["scale_ms"] == scale


def test_sax_myocardium_is_not_allowed_for_long_axis() -> None:
    with pytest.raises(ValueError, match="SAX myocardium"):
        viz.paired_common_support("2ch", np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2), bool), myocardium_mask=np.ones((2, 2), bool))


def test_heldout_support_matches_evaluator_support() -> None:
    reference = np.array([[1.0, 2.0], [np.nan, 4.0]])
    proposed = np.array([[1.0, 2.0], [3.0, 4.0]])
    fit_first = np.array([[1.0, 2.0], [3.0, np.nan]])
    source = np.ones((2, 2), dtype=bool)
    valid = np.ones((2, 2), dtype=bool)
    expected = source & valid & np.isfinite(reference) & np.isfinite(proposed) & np.isfinite(fit_first)
    actual = viz.paired_common_support("2ch", reference, proposed, fit_first, source, reprojection_valid=valid)
    assert np.array_equal(actual, expected)
