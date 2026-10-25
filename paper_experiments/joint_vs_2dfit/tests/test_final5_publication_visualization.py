"""Contracts for the read-only Final5 single-element publication renderer."""
from __future__ import annotations

import numpy as np
import pytest


def test_single_element_errors_reuse_the_parameter_map_lut_and_range(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.final5_publication_visualization import render_single_element_plane
    from trad.evaluation.scmr.quality_control import DISPLAY_RANGES_MS

    captured: list[tuple[object, tuple[float, float]]] = []
    monkeypatch.setattr(
        "paper_experiments.joint_vs_2dfit.phase2_clean.final5_publication_visualization._save_panel",
        lambda _path, _values, _support, *, cmap, limits: captured.append((cmap, limits)),
    )
    values = np.arange(9, dtype=float).reshape(3, 3)
    render_single_element_plane(
        tmp_path, values, values + 1, values + 2, np.ones_like(values, bool), parameter="T1",
        bounds={"row_start": 0, "row_stop": 3, "column_start": 0, "column_stop": 3},
    )

    assert len(captured) == 5
    assert all(limits == DISPLAY_RANGES_MS["T1"] for _cmap, limits in captured)
    assert len({id(cmap) for cmap, _limits in captured}) == 1


def test_oblique_native_plane_reprojection_uses_endpoint_world_geometry() -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.evaluation import reproject_volume_to_native_plane
    from paper_experiments.joint_vs_2dfit.phase2_clean.final_native_grid_metrics import native_ras_affine

    native_ras = np.array([
        [0.0, -2.0, 0.0, 13.0],
        [1.5, 0.0, 0.0, -7.0],
        [0.0, 0.0, 3.0, 11.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    native_lps_rc = lps_to_ras @ native_ras
    endpoint = np.fromfunction(lambda row, column, _slice: 100.0 * row + column, (4, 5, 1))

    mapped, valid = reproject_volume_to_native_plane(endpoint, native_ras, native_ras_affine(native_lps_rc), (4, 5))

    assert valid.all()
    assert np.allclose(mapped, endpoint[:, :, 0], atol=1e-12, rtol=0)


def test_manifest_group_roles_keep_sax_and_do_not_invent_l100_heldout() -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.final5_publication_visualization import publication_group_roles

    source = {"sax": [0, 1], "2ch": [0, 1], "4ch": [0, 1]}
    sparse = {"case_id": "L50_fold0", "stacks": {"sax": {}, "2ch": {"retained_original_group_ids": [0], "heldout_original_group_ids": [1]}, "4ch": {"retained_original_group_ids": [1], "heldout_original_group_ids": [0]}}}
    assert publication_group_roles(sparse, source) == {("sax", 0): "observed", ("sax", 1): "observed", ("2ch", 0): "retained", ("2ch", 1): "heldout", ("4ch", 0): "heldout", ("4ch", 1): "retained"}
    assert set(publication_group_roles({"case_id": "L100", "stacks": {}}, source).values()) == {"observed"}


def test_single_element_panels_share_one_crop_and_display_contract(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.final5_publication_visualization import (
        crop_bounds, display_support, render_single_element_plane,
    )

    reference = np.arange(36, dtype=float).reshape(6, 6)
    proposed, fit_first = reference + 1, reference + 2
    foreground = np.zeros((6, 6), bool); foreground[1:5, 2:5] = True
    support = display_support(foreground, reference, proposed, fit_first, np.ones_like(foreground), np.ones_like(foreground))
    bounds = crop_bounds(support, pad=0)
    written = render_single_element_plane(tmp_path, reference, proposed, fit_first, support, parameter="T1", bounds=bounds)
    assert set(written) == {"native_reference", "proposed", "fit_first", "proposed_absolute_error", "fit_first_absolute_error"}
    assert all(path.is_file() and path.with_suffix(".pdf").is_file() for path in written.values())
    assert bounds == {"row_start": 1, "row_stop": 5, "column_start": 2, "column_stop": 5}


def test_display_support_has_no_component_filter_and_requires_both_predictions() -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.final5_publication_visualization import display_support

    foreground = np.array([[True, False, True]])
    values = np.ones((1, 3))
    support = display_support(foreground, values, values, values, np.ones_like(foreground), np.ones_like(foreground))
    assert np.array_equal(support, foreground)
    assert not display_support(foreground, values, values, np.array([[1.0, 1.0, np.nan]]), np.ones_like(foreground), np.ones_like(foreground))[0, 2]


def test_rejects_existing_publication_output(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.final5_publication_visualization import require_new_publication_output

    output = tmp_path / "publication"; output.mkdir()
    with pytest.raises(FileExistsError):
        require_new_publication_output(output)


def test_dry_run_reports_missing_input_without_creating_output(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.final5_publication_visualization import publication_readiness

    output = tmp_path / "publication"
    result = publication_readiness(final5_root=tmp_path / "missing-final5", subject="DYL_2", case_id="L100", output=output)
    assert result["status"] == "MISSING_INPUT"
    assert not output.exists()
