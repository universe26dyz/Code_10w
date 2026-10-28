"""Contract tests for the Phase-4R Joint-vs-fit-first renderer."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

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


def test_final5_paired_support_requires_primary_mask_and_both_native_bundle_validities() -> None:
    reference = np.array([[1000.0, 1000.0], [np.nan, 1000.0]])
    proposed = np.array([[1001.0, np.nan], [1001.0, 1001.0]])
    fit_first = np.full((2, 2), 999.0)
    support = viz.final5_paired_display_support(
        reference, proposed, fit_first, np.ones((2, 2), bool),
        np.array([[True, True], [True, False]]),
        np.array([[True, True], [True, True]]),
        np.array([[True, True], [False, True]]),
    )
    assert np.array_equal(support, np.array([[True, False], [False, False]]))
    panels, _ = viz.prepare_native_comparison_panels(reference, proposed, fit_first, support)
    assert all(np.array_equal(panel.mask, support) for panel in panels.values())


def test_display_support_keeps_valid_foreground_outside_myocardium() -> None:
    reference = np.full((5, 5), 1000.0)
    proposed, fit_first = reference + 10.0, reference - 10.0
    myocardium = np.zeros((5, 5), bool); myocardium[2, 2] = True
    display = viz.native_display_support(reference, proposed, fit_first, np.ones((5, 5), bool), np.ones((5, 5), bool))
    assert display[0, 0]
    assert display.sum() > myocardium.sum()
    panels, crop = viz.prepare_native_comparison_panels(reference, proposed, fit_first, display)
    contour = viz.crop_myocardium_contour(myocardium, crop)
    assert contour.sum() == 1 and panels["native"].mask[0, 0]


@pytest.mark.parametrize(("parameter", "range_"), (("T1", (0.0, 2500.0)), ("T2", (0.0, 200.0))))
def test_native_and_error_panels_share_parameter_lut_and_range(parameter: str, range_: tuple[float, float]) -> None:
    specs = viz.native_panel_specs(parameter)
    assert len(specs) == 5
    assert all(spec["limits"] == range_ for spec in specs)
    assert all(spec["cmap"].name == viz.colorbars().mapping[parameter].name for spec in specs)


def test_native_plane_figure_has_one_colorbar_and_accepts_roi_contour(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import matplotlib.figure

    calls: list[object] = []
    monkeypatch.setattr(matplotlib.figure.Figure, "colorbar", lambda self, *args, **kwargs: calls.append((args, kwargs)))
    support = np.ones((3, 3), bool)
    panels, _ = viz.prepare_native_comparison_panels(np.full((3, 3), 1000.0), np.full((3, 3), 1010.0), np.full((3, 3), 990.0), support)
    written = viz.render_native_plane_comparison("T1", panels, tmp_path / "figure" / "panel.png", title="test", metadata={}, myocardium_contour=np.eye(3, dtype=bool))
    assert len(calls) == 1 and Path(written["png"]).is_file()


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
    assert result["grid_spacing_mm"] == {"row": 1.0, "slice": 1.0}


def test_orthogonal_centre_row_cut_uses_column_and_slice_world_axes() -> None:
    native = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)
    result = viz.sample_same_world_through_plane(native, np.eye(4), native, np.eye(4), native, np.eye(4), cut_row=2)
    assert result["cut_orientation"] == "row"
    assert result["cut_row"] == 2 and result["cut_column"] is None
    assert result["grid_spacing_mm"] == {"column": 1.0, "slice": 1.0}
    assert result["native"].shape == (5, 6)


def test_common_native_plane_samples_both_reconstructions_on_one_native_world_grid() -> None:
    from paper_experiments.joint_vs_2dfit.aha.common_native import sample_common_native_plane

    reference = np.arange(12, dtype=float).reshape(3, 4)
    plane = sample_common_native_plane(
        reference=reference, native_affine_ras=np.eye(4),
        proposed_volume=reference[:, :, None] + 10, proposed_affine_ras=np.eye(4),
        fit_first_volume=reference[:, :, None] + 20, fit_first_affine_ras=np.eye(4),
    )
    assert np.array_equal(plane["proposed"], reference + 10)
    assert np.array_equal(plane["fit_first"], reference + 20)
    assert plane["proposed_valid"].all() and plane["fit_first_valid"].all()


def test_common_native_roi_sets_select_only_the_current_sax_plane() -> None:
    from paper_experiments.joint_vs_2dfit.aha.common_native_metrics_qc import _sax_roi_plane_sets

    outer = np.ones((2, 3, 4), bool)
    myocardium = {"myocardium_core_1px": np.zeros((2, 3, 4), bool)}
    myocardium["myocardium_core_1px"][1] = True
    rois = _sax_roi_plane_sets(outer, myocardium, index=1)
    assert {name: value.shape for name, value in rois.items()} == {
        "lv_outer_filled": (3, 4), "myocardium_core_1px": (3, 4),
    }
    assert rois["myocardium_core_1px"].all()


def test_sax_mask_stack_projects_as_nearest_neighbor_3d_volume() -> None:
    from paper_experiments.joint_vs_2dfit.aha.common_native_all_stack_roi_metrics import (
        project_sax_mask_volume, stack_sax_masks_as_volume,
    )
    affines = np.repeat(np.eye(4)[None], 2, axis=0)
    affines[1, 2, 3] = 1.0
    masks = {"lv_outer_filled": np.zeros((2, 3, 4), bool)}
    masks["lv_outer_filled"][1, 1, 2] = True
    volumes, affine = stack_sax_masks_as_volume({"affine_lps_rc": affines}, masks)
    assert volumes["lv_outer_filled"].shape == (3, 4, 2)
    assert project_sax_mask_volume(volumes["lv_outer_filled"], affine, affine, (3, 4), slice_index=1)[1, 2]


def test_blood_pool_mask_uses_filled_endocardium_clipped_to_outer_contour() -> None:
    from paper_experiments.joint_vs_2dfit.aha.common_native_all_stack_roi_metrics import legacy_blood_pool_mask

    inner = np.array([[True, True, False], [False, True, False]])
    outer = np.array([[True, False, False], [False, True, True]])
    assert np.array_equal(legacy_blood_pool_mask(inner, outer), np.array([[True, False, False], [False, True, False]]))


def test_fit_first_zero_audit_keeps_finite_zeroes_visible_in_roi_counts() -> None:
    from paper_experiments.joint_vs_2dfit.aha.common_native_all_stack_roi_metrics import fit_first_zero_audit

    roi = np.array([[True, True], [True, False]])
    fit_first = np.array([[0.0, 1200.0], [0.0, np.nan]])
    valid = np.array([[True, True], [False, True]])
    common = np.array([[True, True], [False, False]])
    audit = fit_first_zero_audit(roi=roi, fit_first=fit_first, fit_first_valid=valid, common_support=common)
    assert audit == {"fit_first_zero_valid_roi_voxels": 1, "fit_first_zero_common_support_voxels": 1}


def test_through_plane_grid_is_physical_for_anisotropic_geometry() -> None:
    native = np.arange(4 * 5 * 6, dtype=float).reshape(4, 5, 6)
    affine = np.diag([2.0, 3.0, 8.0, 1.0])
    result = viz.sample_same_world_through_plane(native, affine, native, affine, native, affine, cut_column=2)
    assert result["native"].shape == (7, 41)
    assert result["physical_extent_mm"]["row"] == [0.0, 6.0]
    assert result["physical_extent_mm"]["slice"] == [0.0, 40.0]
    assert result["world"].shape == (4, 7, 41)
    assert np.array_equal(result["world"], result["proposed_world"])
    assert result["physical_extent_mm"]["slice"][1] / result["physical_extent_mm"]["row"][1] > 5


def test_invalid_through_plane_geometry_fails_instead_of_index_fallback() -> None:
    native = np.ones((3, 3, 3))
    invalid = np.eye(4); invalid[:3, 2] = invalid[:3, 0]
    with pytest.raises(ValueError, match="physical"):
        viz.sample_same_world_through_plane(native, invalid, native, np.eye(4), native, np.eye(4))


def test_renderer_contains_no_psf_matched_language() -> None:
    source = Path(viz.__file__).read_text(encoding="utf-8").lower()
    assert "psf-matched" not in source
    assert "k=8 matched" not in source


def _fit_layout(root: Path, *, mean: float, scale: float) -> tuple[Path, Path]:
    import nibabel as nib

    normalized, ms = root / "volume_normalized.nii.gz", root / "volume_ms.nii.gz"
    root.mkdir(parents=True)
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


def test_r1_cache_selection_uses_heldout_report_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sparse_manifest = tmp_path / "sparse_manifest.json"
    sparse_manifest.write_text(json.dumps({"status": "ELIGIBLE"}))
    cache_root = tmp_path / "r0_native_cache"; cache_root.mkdir()
    report = {
        "provenance": {
            "sparse_manifest": {"path": str(sparse_manifest), "sha256": viz._sha256(sparse_manifest)},
            "native_reference_cache_root": str(cache_root),
        },
        "stacks": {"2ch": {"heldout_original_group_ids": [4, 6]}},
    }
    calls: list[tuple[Path, Path, str, list[int]]] = []

    def select(cache: str | Path, manifest: str | Path, stack: str, groups: list[int]) -> dict[str, object]:
        calls.append((Path(cache), Path(manifest), stack, groups))
        return {"group_idx": np.asarray(groups), "source_support": np.ones((2, 2, 2), bool)}

    monkeypatch.setattr(viz, "select_verified_native_cache_groups", select)
    selected = viz.select_verified_heldout_native_cache(report, "2ch")
    assert selected["group_idx"].tolist() == [4, 6]
    assert calls == [(cache_root, sparse_manifest, "2ch", [4, 6])]


def test_r1_cache_selection_rejects_report_manifest_hash_mismatch(tmp_path: Path) -> None:
    sparse_manifest = tmp_path / "sparse_manifest.json"
    sparse_manifest.write_text(json.dumps({"status": "ELIGIBLE"}))
    report = {
        "provenance": {"sparse_manifest": {"path": str(sparse_manifest), "sha256": "wrong"}, "native_reference_cache_root": str(tmp_path)},
        "stacks": {"2ch": {"heldout_original_group_ids": [4]}},
    }
    with pytest.raises(ValueError, match="SHA256"):
        viz.select_verified_heldout_native_cache(report, "2ch")


def _r0_inputs(root: Path, *, mean: float = 0, t1_scale: float = 2500.0, t2_scale: float = 200.0) -> tuple[Path, Path, Path]:
    proposed, fit, data = root / "proposed", root / "fit", root / "data"
    for subject in ("CYJ", "DYZ", "HHZ", "HJL"):
        subject_proposed = proposed / subject
        subject_proposed.mkdir(parents=True)
        (subject_proposed / "T1_3D.nii.gz").write_bytes(b"proposed")
        (subject_proposed / "T2_3D.nii.gz").write_bytes(b"proposed")
        for parameter, scale in (("t1", t1_scale), ("t2", t2_scale)):
            normalized, ms = _fit_layout(fit / subject / "nesvor" / parameter, mean=mean, scale=scale)
            assert normalized.is_file() and ms.is_file()
        reference = data / "reference_data" / "phase2_clean" / subject / "native_reference_2d"
        reference.mkdir(parents=True)
        for name in ("sax.npz", "sax_T1_stack_ms.nii.gz", "sax_T2_stack_ms.nii.gz", "native_reference_manifest.json"):
            (reference / name).write_text(name)
        masks = data / "reference_data" / "phase2_clean" / subject / "myocardium_masks_v1"
        masks.mkdir()
        (masks / "sax_myocardium_masks.npz").write_text("masks")
        (masks / "manifest.json").write_text("mask manifest")
    return proposed, fit, data


def test_r0_preflight_rejects_mean_700_before_any_output(tmp_path: Path) -> None:
    proposed, fit, data = _r0_inputs(tmp_path, mean=700)
    output = tmp_path / "formal_output"
    with pytest.raises(ValueError, match="mean-700"):
        viz.preflight_r0_inputs(proposed, fit, data)
    assert not output.exists()


def test_direct_r0_evaluation_script_rejects_mean_700_before_output_creation(tmp_path: Path) -> None:
    proposed, fit, data = _r0_inputs(tmp_path, mean=700)
    script = Path(__file__).resolve().parents[1] / "phase2_clean" / "server_commands" / "evaluate_real_full.sh"
    output = tmp_path / "formal_evaluation"
    completed = subprocess.run(["bash", str(script), str(proposed), str(fit), str(output)], env={**os.environ, "REPO": str(Path(__file__).resolve().parents[3]), "DATA": str(data), "PHASE4R_PYTHON": os.sys.executable}, capture_output=True, text=True)
    assert completed.returncode != 0
    assert "mean-700" in completed.stderr
    assert not output.exists()


@pytest.mark.parametrize(("parameter", "scale"), (("t1", 2499.0), ("t2", 199.0)))
def test_r0_preflight_rejects_wrong_scale_before_any_output(tmp_path: Path, parameter: str, scale: float) -> None:
    proposed, fit, data = _r0_inputs(tmp_path, **{f"{parameter}_scale": scale})
    with pytest.raises(ValueError, match="does not satisfy"):
        viz.preflight_r0_inputs(proposed, fit, data)
    assert not (tmp_path / "formal_output").exists()


def test_r0_preflight_accepts_all_four_valid_subjects(tmp_path: Path) -> None:
    proposed, fit, data = _r0_inputs(tmp_path)
    report = viz.preflight_r0_inputs(proposed, fit, data)
    assert list(report["subjects"]) == ["CYJ", "DYZ", "HHZ", "HJL"]


@pytest.mark.parametrize(("relative", "expected"), (("proposed/CYJ/T1_3D.nii.gz", "proposed"), ("data/reference_data/phase2_clean/CYJ/native_reference_2d/sax.npz", "reference"), ("data/reference_data/phase2_clean/CYJ/myocardium_masks_v1/manifest.json", "mask")))
def test_r0_preflight_rejects_missing_required_input(tmp_path: Path, relative: str, expected: str) -> None:
    proposed, fit, data = _r0_inputs(tmp_path)
    (tmp_path / relative).unlink()
    with pytest.raises(FileNotFoundError, match=expected):
        viz.preflight_r0_inputs(proposed, fit, data)


def test_r0_provenance_names_manifests_and_npz_hashes_correctly(tmp_path: Path) -> None:
    proposed, fit, data = _r0_inputs(tmp_path)
    report = viz.preflight_r0_inputs(proposed, fit, data)
    subject = report["subjects"]["CYJ"]
    provenance = viz.r0_reference_mask_provenance(Path(subject["native_reference"]), Path(subject["myocardium_bundle"]))
    assert provenance["native_reference_manifest_sha256"] == viz._sha256(provenance["native_reference_manifest_path"])
    assert provenance["native_reference_sax_npz_sha256"] == viz._sha256(provenance["native_reference_sax_npz_path"])
    assert provenance["myocardium_manifest_sha256"] == viz._sha256(provenance["myocardium_manifest_path"])
    assert provenance["myocardium_masks_npz_sha256"] == viz._sha256(provenance["myocardium_masks_npz_path"])


def test_combined_wrapper_cleans_staging_and_never_publishes_on_late_failure(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "phase2_clean" / "server_commands" / "evaluate_and_visualize_real_full.sh"
    repo = tmp_path / "repo"
    target = repo / "paper_experiments" / "joint_vs_2dfit" / "phase2_clean" / "server_commands"
    target.mkdir(parents=True)
    wrapper = target / source.name
    shutil.copy2(source, wrapper)
    evaluation = target / "evaluate_real_full.sh"
    evaluation.write_text("#!/usr/bin/env bash\nset -eu\nmkdir -p \"$3\"\ntouch \"$3/CYJ_paired.json\"\n")
    evaluation.chmod(0o755)
    fake_python = tmp_path / "fake_python.sh"
    fake_python.write_text("#!/usr/bin/env bash\ncase \"$*\" in *r0-preflight*) exit 0;; *) exit 9;; esac\n")
    fake_python.chmod(0o755)
    final_evaluation, final_figures = tmp_path / "evaluation", tmp_path / "figures"
    completed = subprocess.run(["bash", str(wrapper), "proposed", "fit", str(final_evaluation), str(final_figures)], env={**os.environ, "REPO": str(repo), "PHASE4R_PYTHON": str(fake_python)}, capture_output=True, text=True)
    assert completed.returncode == 9
    assert not final_evaluation.exists() and not final_figures.exists()
    assert not list(tmp_path.glob(".evaluation.staging-*"))
    assert not list(tmp_path.glob(".figures.staging-*"))
