"""CPU contracts for final CYJ matched summaries and paper figures."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


def _metrics(value: float) -> dict[str, object]:
    return {"rmse": value, "mae": value / 2, "bias": -value / 10, "voxel_count": 10, "coverage_fraction": 1.0}


def _result(root: Path, *, condition: str, endpoint: str, proposed: float, fit_first: float) -> None:
    target = root / condition / endpoint; target.mkdir(parents=True)
    regions = {parameter: {"all_lax": {"l100_full": {"proposed": _metrics(proposed), "fit_first": _metrics(fit_first)}}} for parameter in ("t1", "t2")}
    (target / "manifest.json").write_text(json.dumps({"subject": "CYJ", "condition": condition, "endpoint": endpoint}), encoding="utf-8")
    (target / "aggregate_metrics.json").write_text(json.dumps({"regions": regions}), encoding="utf-8")


def test_final_summary_uses_sample_sd_keeps_methods_endpoints_and_missing_folds_separate(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.final_cyj_reporting import aggregate_final_cyj

    q002, q004 = tmp_path / "q002", tmp_path / "q004"
    _result(q002, condition="L100_fold0", endpoint="psf128", proposed=1.0, fit_first=2.0)
    _result(q002, condition="L50_fold0", endpoint="psf128", proposed=3.0, fit_first=2.0)
    _result(q002, condition="L50_fold1", endpoint="psf128", proposed=5.0, fit_first=6.0)
    _result(q004, condition="L25_fold0", endpoint="central", proposed=6.0, fit_first=7.0)
    output = aggregate_final_cyj({"Q002S640": q002, "Q004S640": q004}, tmp_path / "summary", fit_first_source_method="Q002S640")
    payload = json.loads((output / "summary.json").read_text())
    q002_l50 = next(row for row in payload["level_summaries"] if row["method"] == "Q002S640" and row["endpoint"] == "psf128" and row["level"] == "L50" and row["parameter"] == "T1")
    q004_l25 = next(row for row in payload["level_summaries"] if row["method"] == "Q004S640" and row["endpoint"] == "central" and row["level"] == "L25" and row["parameter"] == "T1")
    assert q002_l50["rmse_mean"] == pytest.approx(4.0)
    assert q002_l50["rmse_sample_sd"] == pytest.approx(2**0.5)
    assert q002_l50["proposed_rmse_win_count"] == 1
    assert q004_l25["missing_folds"] == [1, 2, 3]
    assert q004_l25["status"] == "INCOMPLETE_MISSING_FOLDS"
    assert {row["endpoint"] for row in payload["level_summaries"]} == {"central", "psf128"}
    assert (output / "paper_table.csv").is_file() and (output / "summary.md").is_file()
    assert "Q002S640" in (output / "summary.md").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        aggregate_final_cyj({"Q002S640": q002}, output, fit_first_source_method="Q002S640")


def test_longitudinal_plane_has_one_world_extent_and_final_error_uses_map_lut_range() -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.final_cyj_reporting import final_panel_specifications, longitudinal_plane, meaningful_longitudinal_index

    mask = np.ones((3, 4, 5), bool)
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    plane = longitudinal_plane(mask, affine, axis=1, index=2)
    assert plane["world"].shape == (4, 3, 5)
    assert plane["extent_mm"] == [0.0, 4.0, 0.0, 16.0]
    specs = final_panel_specifications("T1", ("Q002S640", "Fit-first"))
    assert all(spec["cmap"].name == specs[0]["cmap"].name and spec["range_ms"] == (0.0, 2500.0) for spec in specs)
    assert {spec["displayed_quantity"] for spec in specs if spec["kind"] == "absolute_error_ms"} == {"absolute_error_ms"}
    sparse_support = np.zeros((5, 4, 3), bool); sparse_support[1] = True; sparse_support[3] = True
    assert meaningful_longitudinal_index(sparse_support, axis=0) == (1, "support_median")
    assert meaningful_longitudinal_index(sparse_support, axis=0, index=3) == (3, "configured_index")


def test_native_preview_mode_selects_single_or_all_authoritative_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_cyj_reporting as reporting

    sparse = {"condition": "L50", "stacks": {"2ch": {"heldout_original_group_ids": [7, 9]}}}
    monkeypatch.setattr(reporting, "_heldout_ids", lambda manifest, stack: [7, 9])
    assert reporting.native_preview_groups({"stack": "2ch", "original_group_idx": 9}, sparse) == ("single", [9])
    assert reporting.native_preview_groups({"mode": "all_groups", "stack": "2ch"}, sparse) == ("all_groups", [7, 9])

    l100 = {"condition": "L100", "stacks": {"2ch": {}}}
    monkeypatch.setattr(reporting, "_heldout_ids", lambda manifest, stack: [2, 4, 6])
    assert reporting.native_preview_groups({"mode": "all_groups", "stack": "2ch"}, l100) == ("all_groups", [2, 4, 6])
    stacks = reporting.native_preview_stack_configs({"mode": "all_groups", "sparse_manifest": "sparse.json", "r0_native_maps_root": "cache", "stacks": ["2ch", {"stack": "4ch"}]})
    assert [item["stack"] for item in stacks] == ["2ch", "4ch"]
    assert all(item["mode"] == "all_groups" for item in stacks)


def test_native_mosaic_uses_one_shared_parameter_colorbar_axis(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import matplotlib.figure
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_cyj_reporting as reporting

    calls: list[object] = []
    original = matplotlib.figure.Figure.colorbar

    def capture(self: object, mappable: object, *args: object, **kwargs: object) -> object:
        calls.append(kwargs.get("cax"))
        return original(self, mappable, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "colorbar", capture)
    values = np.full((3, 4), 1000.0)
    panel = lambda label: {"label": label, "values": values, "support": np.ones_like(values, bool), "cmap": reporting.colorbars().mapping["T1"], "range_ms": (0.0, 2500.0)}
    reporting._plot_native_mosaic(tmp_path / "mosaic.png", [(5, [panel("Native reference"), panel("Fit-first"), panel("|Fit-first − Native|")])], title="test")
    assert len(calls) == 1
    assert calls[0] is not None


def test_all_groups_native_render_uses_every_heldout_group_and_records_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_cyj_reporting as reporting

    sparse_manifest = tmp_path / "sparse.json"
    sparse_manifest.write_text(json.dumps({"condition": "L25", "stacks": {"2ch": {"heldout_original_group_ids": [3, 8]}}}), encoding="utf-8")
    mask = _nifti(tmp_path / "mask.nii.gz", np.ones((2, 3, 2)), np.eye(4))
    maps = {method: {parameter: str(_nifti(tmp_path / method / f"{parameter}.nii.gz", np.ones((2, 3, 2)), np.eye(4))) for parameter in ("t1", "t2")} for method in ("Fit-first", "Q002S640", "Q004S640")}
    cache_files = [tmp_path / name for name in ("native_maps.npz", "geometry.npz", "native_manifest.json")]
    for path in cache_files:
        path.write_bytes(b"cache")
    selected = {
        "group_idx": np.array([3, 8]), "source_support": np.ones((2, 2, 3), bool),
        "affine_lps_rc": np.repeat(np.eye(4)[None], 2, axis=0), "pixel_spacing_rc_mm": np.ones((2, 2)),
        "t1_ms": np.full((2, 2, 3), 1000.0), "t2_ms": np.full((2, 2, 3), 80.0),
        "provenance": dict(zip(("native_maps_path", "geometry_path", "manifest_path"), map(str, cache_files))),
    }
    requested: list[list[int]] = []
    monkeypatch.setattr(reporting, "_heldout_ids", lambda sparse, stack: [3, 8])
    monkeypatch.setattr(reporting, "select_verified_native_cache_groups", lambda root, manifest, stack, groups: requested.append(groups) or selected)
    monkeypatch.setattr(reporting, "reproject_binary_mask_to_native_plane", lambda mask, affine, plane, shape: np.ones(shape, bool))
    monkeypatch.setattr(reporting, "reproject_volume_to_native_plane", lambda values, affine, plane, shape: (np.full(shape, 900.0), np.ones(shape, bool)))
    rendered: list[object] = []
    original_plot = reporting._plot_native_mosaic
    monkeypatch.setattr(reporting, "_plot_native_mosaic", lambda path, rows, *, title: (rendered.append(rows), original_plot(path, rows, title=title))[1])
    config = {"subject": "CYJ", "condition": "L25_fold0", "endpoint": "central", "methods": maps, "l100_masks": {"t1": str(mask), "t2": str(mask)}, "native": {"mode": "all_groups", "stack": "2ch", "sparse_manifest": str(sparse_manifest), "r0_native_maps_root": str(tmp_path / "cache")}}
    records = reporting._render_native(config, tmp_path, {})
    assert requested == [[3, 8]]
    assert all(record["mode"] == "all_groups" and record["original_group_ids"] == [3, 8] for record in records)
    expected_columns = ["Native reference", "Fit-first", "Q002S640", "Q004S640", "|Fit-first − Native|", "|Q002S640 − Native|", "|Q004S640 − Native|"]
    assert [panel["label"] for panel in rendered[0][0]["panels"]] == expected_columns
    assert all(panel["cmap"].name == rendered[0][0]["panels"][0]["cmap"].name and panel["range_ms"] == rendered[0][0]["panels"][0]["range_ms"] for panel in rendered[0][0]["panels"])
    assert all((tmp_path / record["file"]).is_file() for record in records)


def test_native_mosaic_reads_proposed_direct_native_bundle_without_volume_reprojection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_cyj_reporting as reporting

    sparse_manifest = tmp_path / "sparse.json"
    sparse_manifest.write_text(json.dumps({"condition": "L100", "stacks": {"2ch": {}}}), encoding="utf-8")
    mask = _nifti(tmp_path / "mask.nii.gz", np.ones((2, 3, 2)), np.eye(4))
    bundle = tmp_path / "q004_bundle"
    (bundle / "planes" / "2ch").mkdir(parents=True)
    (bundle / "manifest.json").write_text(json.dumps({"schema": "direct_native_prediction_bundle/v1", "endpoint": "central"}), encoding="utf-8")
    np.savez_compressed(bundle / "planes" / "2ch" / "group003.npz", t1_ms=np.full((2, 3), 800.0), t2_ms=np.full((2, 3), 70.0), global_query_index=np.arange(6).reshape(2, 3))
    fit = {parameter: str(_nifti(tmp_path / "fit" / f"{parameter}.nii.gz", np.ones((2, 3, 2)), np.eye(4))) for parameter in ("t1", "t2")}
    cache_files = [tmp_path / name for name in ("native_maps.npz", "geometry.npz", "native_manifest.json")]
    for path in cache_files:
        path.write_bytes(b"cache")
    selected = {"group_idx": np.array([3]), "source_support": np.ones((1, 2, 3), bool), "affine_lps_rc": np.repeat(np.eye(4)[None], 1, axis=0), "pixel_spacing_rc_mm": np.ones((1, 2)), "t1_ms": np.full((1, 2, 3), 1000.0), "t2_ms": np.full((1, 2, 3), 80.0), "provenance": dict(zip(("native_maps_path", "geometry_path", "manifest_path"), map(str, cache_files)))}
    monkeypatch.setattr(reporting, "_heldout_ids", lambda sparse, stack: [3])
    monkeypatch.setattr(reporting, "select_verified_native_cache_groups", lambda *args: selected)
    monkeypatch.setattr(reporting, "reproject_binary_mask_to_native_plane", lambda *args: np.ones((2, 3), bool))
    calls: list[str] = []
    monkeypatch.setattr(reporting, "reproject_volume_to_native_plane", lambda *args: (calls.append("fit") or np.full((2, 3), 900.0), np.ones((2, 3), bool)))
    rendered: list[object] = []
    monkeypatch.setattr(reporting, "_plot_native_mosaic", lambda path, rows, *, title: rendered.append(rows))
    config = {"subject": "CYJ", "condition": "L100_fold0", "endpoint": "central", "methods": {"Fit-first": fit, "Q004S640": {"native_bundle": str(bundle)}}, "l100_masks": {"t1": str(mask), "t2": str(mask)}, "native": {"mode": "all_groups", "stack": "2ch", "sparse_manifest": str(sparse_manifest), "r0_native_maps_root": str(tmp_path)}}
    reporting._render_native(config, tmp_path, {})
    q004 = next(panel for panel in rendered[0][0]["panels"] if panel["label"] == "Q004S640")
    assert np.all(q004["values"] == 800.0)
    assert calls == ["fit", "fit"]


def test_native_mosaic_reads_fit_first_final_pose_bundle_without_r0_reprojection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_cyj_reporting as reporting

    sparse_manifest = tmp_path / "sparse.json"
    sparse_manifest.write_text(json.dumps({"condition": "L100", "stacks": {"2ch": {}}}), encoding="utf-8")
    mask = _nifti(tmp_path / "mask.nii.gz", np.ones((2, 3, 2)), np.eye(4))
    fit = tmp_path / "fit_bundle"
    for parameter, value in (("t1", 777.0), ("t2", 66.0)):
        target = fit / "planes" / parameter / "2ch"; target.mkdir(parents=True)
        np.savez_compressed(target / "group003.npz", prediction_ms=np.full((2, 3), value), valid=np.ones((2, 3), bool))
    (fit / "manifest.json").write_text(json.dumps({"schema": "fit_first_native_prediction_bundle/v1"}), encoding="utf-8")
    cache_files = [tmp_path / name for name in ("native_maps.npz", "geometry.npz", "native_manifest.json")]
    for path in cache_files: path.write_bytes(b"cache")
    selected = {"group_idx": np.array([3]), "source_support": np.ones((1, 2, 3), bool), "affine_lps_rc": np.eye(4)[None], "pixel_spacing_rc_mm": np.ones((1, 2)), "t1_ms": np.full((1, 2, 3), 1000.0), "t2_ms": np.full((1, 2, 3), 80.0), "provenance": dict(zip(("native_maps_path", "geometry_path", "manifest_path"), map(str, cache_files)))}
    monkeypatch.setattr(reporting, "_heldout_ids", lambda sparse, stack: [3]); monkeypatch.setattr(reporting, "select_verified_native_cache_groups", lambda *args: selected)
    monkeypatch.setattr(reporting, "reproject_binary_mask_to_native_plane", lambda *args: np.ones((2, 3), bool))
    monkeypatch.setattr(reporting, "reproject_volume_to_native_plane", lambda *args: pytest.fail("formal Fit-first endpoint must not reproject R0 affine"))
    rendered: list[object] = []; monkeypatch.setattr(reporting, "_plot_native_mosaic", lambda path, rows, *, title: rendered.append(rows))
    config = {"subject": "CYJ", "condition": "L100_fold0", "endpoint": "central", "methods": {"Fit-first": {"fit_first_native_bundle": str(fit)}}, "l100_masks": {"t1": str(mask), "t2": str(mask)}, "native": {"mode": "all_groups", "stack": "2ch", "sparse_manifest": str(sparse_manifest), "r0_native_maps_root": str(tmp_path)}}
    reporting._render_native(config, tmp_path, {})
    assert next(panel for panel in rendered[0][0]["panels"] if panel["label"] == "Fit-first")["values"][0, 0] == 777.0


def _nifti(path: Path, values: np.ndarray, affine: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(values.astype(np.float32), affine), path)
    return path


def test_longitudinal_render_uses_one_full_world_extent_for_every_method_without_auto_crop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_cyj_reporting as reporting

    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    mask = _nifti(tmp_path / "mask.nii.gz", np.ones((3, 4, 5)), affine)
    methods = {
        method: {parameter: str(_nifti(tmp_path / method / f"{parameter}.nii.gz", np.full((3, 4, 5), value), affine)) for parameter in ("t1", "t2")}
        for method, value in (("Q002S640", 1000), ("Fit-first", 1100))
    }
    config = {"subject": "CYJ", "condition": "L50_fold0", "endpoint": "psf128", "methods": methods,
              "l100_masks": {"t1": str(mask), "t2": str(mask)}, "longitudinal": {"axis": 1, "index": 2}}
    captured: list[tuple[list[float], int]] = []
    monkeypatch.setattr(reporting, "_plot_panels", lambda path, panels, *, extent, title, contour=None: captured.append((extent, len(panels))))
    records = reporting._render_longitudinal(config, tmp_path / "out", {})
    assert [item[0] for item in captured] == [[0.0, 4.0, 0.0, 16.0], [0.0, 4.0, 0.0, 16.0]]
    assert [item[1] for item in captured] == [2, 2]
    assert all(record["no_auto_crop"] is True for record in records)


def test_final_figure_manifest_is_complete_and_immutable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_cyj_reporting as reporting

    config = tmp_path / "figure_config.json"
    config.write_text(json.dumps({"subject": "CYJ", "condition": "L25_fold0", "endpoint": "psf128",
                                  "methods": {"Q002S640": {"t1": "unused", "t2": "unused"}},
                                  "l100_masks": {"t1": "unused", "t2": "unused"},
                                  "longitudinal": {"axis": 1, "index": 2}, "native": {}}), encoding="utf-8")
    monkeypatch.setattr(reporting, "_render_longitudinal", lambda config, output, inputs: [{"plane": {"axis": 1, "index": 2}, "interpolation": {"map": "linear", "mask": "nearest-neighbor"}, "frozen_l100_source": {"path": "mask", "sha256": "x"}}])
    monkeypatch.setattr(reporting, "_render_native", lambda config, output, inputs: [{"mode": "all_groups", "original_group_ids": [2, 4], "colorbars": {"shared_parameter": "dedicated_axis"}, "display_support": "frozen L100", "interpolation": {"map": "linear", "mask": "nearest-neighbor"}, "frozen_l100_source": {"path": "mask", "sha256": "x"}}])
    output = reporting.render_final_figures(config, tmp_path / "figures")
    manifest = json.loads((output / "figure_manifest.json").read_text())
    assert {"subject", "condition", "endpoint", "methods", "inputs", "longitudinal_figures", "native_heldout_figures", "shared_parameter_lut_ranges_ms", "error_display_contract", "code_git_sha"}.issubset(manifest)
    assert manifest["endpoint"] == "psf128" and "same parameter map LUT/range" in manifest["error_display_contract"] and "one shared parameter colorbar" in manifest["error_display_contract"]
    assert manifest["native_heldout_figures"][0]["mode"] == "all_groups"
    with pytest.raises(FileExistsError):
        reporting.render_final_figures(config, output)


def test_final_figure_cli_dispatches_to_renderer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_cyj_reporting as reporting

    target = tmp_path / "figures"
    monkeypatch.setattr(reporting, "render_final_figures", lambda config, output: target)
    monkeypatch.setattr(sys, "argv", ["final_cyj_reporting", "figures", "--config", "figure.json", "--output", str(target)])
    reporting.main()
    assert capsys.readouterr().out.strip() == str(target)


def test_final_figures_reject_out_of_scope_condition(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_cyj_reporting as reporting

    config = tmp_path / "bad_condition.json"
    config.write_text(json.dumps({"subject": "CYJ", "condition": "L33_fold0", "endpoint": "psf128",
                                  "methods": {"Q002S640": {"t1": "unused", "t2": "unused"}},
                                  "l100_masks": {"t1": "unused", "t2": "unused"},
                                  "longitudinal": {"axis": 1, "index": 2}, "native": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported CYJ condition"):
        reporting.render_final_figures(config, tmp_path / "figures")


def test_final_reporting_server_wrapper_exposes_aggregate_and_figures() -> None:
    scripts = Path(__file__).resolve().parents[3] / "paper_experiments" / "joint_vs_2dfit" / "phase2_clean" / "server_commands"
    command = (scripts / "finalize_cyj_reporting.sh").read_text(encoding="utf-8")
    assert "aggregate" in command and "figures" in command
    assert "phase2_clean.final_cyj_reporting" in command and "--method-root" in command and "--config" in command


def test_scmr_compatibility_wrapper_is_auxiliary_and_explicit() -> None:
    scripts = Path(__file__).resolve().parents[3] / "paper_experiments" / "joint_vs_2dfit" / "phase2_clean" / "server_commands"
    command = (scripts / "run_scmr_compatibility_eval.sh").read_text(encoding="utf-8")
    assert "scmr_compatibility_eval" in command
    assert "COMPATIBILITY_CONFIG_JSON" in command and "NEW_COMPATIBILITY_OUTPUT_ROOT" in command
    assert "final_cyj_reporting" not in command


def test_all_source_groups_marks_sparse_heldout_and_observed_groups() -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_cyj_reporting as reporting

    sparse = {"condition": "L50", "stacks": {"sax": {"heldout_original_group_ids": [3, 7]}}}
    mode, groups = reporting.native_preview_groups(
        {"mode": "all_source_groups", "stack": "sax"}, sparse, source_groups=[1, 3, 5, 7],
    )
    assert (mode, groups) == ("all_source_groups", [1, 3, 5, 7])
    assert reporting.native_group_statuses(groups, sparse, "sax") == {
        1: "observed", 3: "held-out", 5: "observed", 7: "held-out",
    }
    assert reporting.native_group_label(3, "held-out") == "group 03 [held-out]"
    assert reporting.native_group_statuses([1, 3], {"condition": "L100"}, "sax") == {1: "source", 3: "source"}


def test_sax_centerline_selection_is_deterministic_and_has_row_and_column_cuts() -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_cyj_reporting as reporting

    selected, rule = reporting.select_sax_reference_group([9, 2, 7, 5], {})
    assert (selected, rule) == (5, "sorted_source_group_lower_median")
    assert reporting.select_sax_reference_group([2, 5, 7], {"sax_reference_group_idx": 7}) == (7, "configured_sax_reference_group_idx")
    assert reporting.sax_centerline_cuts((6, 8)) == (("center_row", 2), ("center_column", 3))
    assert reporting.scmr_longitudinal_panel_labels() == (
        "SAX source + centerline overlay", "Original SAX-stack longitudinal reformatting", "Fit-first", "Q002S640", "Q004S640",
    )


def test_scmr_compatibility_outputs_are_separate_from_primary_summary(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.scmr_compatibility_eval import run_scmr_compatibility_analysis

    affine = np.eye(4)
    reference = _nifti(tmp_path / "reference.nii.gz", np.array([[[1000.0]], [[1200.0]]]), affine)
    fit_first = _nifti(tmp_path / "fit_first.nii.gz", np.array([[[1100.0]], [[1100.0]]]), affine)
    legacy_mask = _nifti(tmp_path / "legacy_mask.nii.gz", np.ones((2, 1, 1)), affine)
    config = {
        "subject": "CYJ", "condition": "L100_fold0", "endpoint": "psf128",
        "reference_maps": {"t1": str(reference), "t2": str(reference)},
        "method_maps": {"Fit-first": {"t1": str(fit_first), "t2": str(fit_first)}},
        "legacy_masks": {"t1": str(legacy_mask), "t2": str(legacy_mask)},
    }
    config_path = tmp_path / "compatibility_config.json"; config_path.write_text(json.dumps(config), encoding="utf-8")
    output = run_scmr_compatibility_analysis(config_path, tmp_path / "scmr_compatibility")
    manifest = json.loads((output / "compatibility_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "phase4r_scmr_compatibility/v1"
    assert "primary_formal_results_unmodified" in manifest["scope"]
    assert (output / "compatibility_metrics.json").is_file()
    assert not (output / "summary.json").exists()


def test_scmr_longitudinal_uses_sax_source_for_two_centreline_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_cyj_reporting as reporting

    affine = np.eye(4)
    cache_files = [tmp_path / name for name in ("native_maps.npz", "geometry.npz", "native_manifest.json")]
    for path in cache_files:
        path.write_bytes(b"cache")
    sax = {
        "group_idx": np.array([2, 5, 8]), "source_support": np.ones((3, 4, 6), bool),
        "affine_lps_rc": np.repeat(np.eye(4)[None], 3, axis=0), "pixel_spacing_rc_mm": np.ones((3, 2)), "slice_thickness_mm": np.ones(3),
        "t1_ms": np.full((3, 4, 6), 1000.0), "t2_ms": np.full((3, 4, 6), 80.0),
        "provenance": dict(zip(("native_maps_path", "geometry_path", "manifest_path"), map(str, cache_files))),
    }
    sax["affine_lps_rc"][1, 2, 3] = 8.0; sax["affine_lps_rc"][2, 2, 3] = 16.0
    monkeypatch.setattr(reporting, "_load_all_verified_native_cache_groups", lambda *args: sax)
    methods = {method: {parameter: str(_nifti(tmp_path / method / f"{parameter}.nii.gz", np.full((4, 6, 3), value), affine)) for parameter in ("t1", "t2")} for method, value in (("Fit-first", 1100), ("Q002S640", 1000), ("Q004S640", 900))}
    captured: list[list[dict[str, object]]] = []
    original_plot = reporting._plot_scmr_longitudinal
    monkeypatch.setattr(reporting, "_plot_scmr_longitudinal", lambda path, rows, **kwargs: (captured.append(rows), original_plot(path, rows, **kwargs))[1])
    config = {"subject": "CYJ", "condition": "L100_fold0", "endpoint": "psf128", "methods": methods,
              "native": {"r0_native_maps_root": str(tmp_path), "sparse_manifest": str(tmp_path / "sparse.json")}, "scmr_longitudinal": {}}
    records = reporting._render_scmr_style_longitudinal(config, tmp_path, {})
    assert len(captured) == 2 and all(len(rows) == 2 for rows in captured)
    assert [[panel["label"] for panel in row["longitudinal"]] for row in captured[0]] == [
        ["Original SAX-stack longitudinal reformatting", "Fit-first", "Q002S640", "Q004S640"],
        ["Original SAX-stack longitudinal reformatting", "Fit-first", "Q002S640", "Q004S640"],
    ]
    assert all(record["layout"] == {"rows": 2, "columns": 5, "column_labels": list(reporting.scmr_longitudinal_panel_labels())} for record in records)
    assert all(record["sampling"]["source"] == "nearest-neighbor on source-stack grid" for record in records)
    assert all(record["sampling"]["reconstruction"] == "linear on shared approximately-1-mm physical plane" for record in records)
    assert all(record["sampling"]["reconstruction_support"] == "in_volume AND finite only" for record in records)
    assert all(record["sampling"]["same_physical_plane_for_source_reformat_and_all_reconstructions"] is True for record in records)
    assert all(record["cuts"][0]["reconstruction_plane"]["actual_spacing_mm"][1] <= 1.0 for record in records)
    assert {(tmp_path / record["file"]).name for record in records} == {"longitudinal_sax_centerline_T1.png", "longitudinal_sax_centerline_T2.png"}
    assert all((tmp_path / record["file"]).is_file() for record in records)


def test_all_source_native_render_records_displayed_statuses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_cyj_reporting as reporting

    sparse_manifest = tmp_path / "sparse.json"; sparse_manifest.write_text(json.dumps({"condition": "L50", "stacks": {"sax": {"heldout_original_group_ids": [2]}}}), encoding="utf-8")
    mask = _nifti(tmp_path / "mask.nii.gz", np.ones((2, 3, 2)), np.eye(4))
    maps = {method: {parameter: str(_nifti(tmp_path / method / f"{parameter}.nii.gz", np.ones((2, 3, 2)), np.eye(4))) for parameter in ("t1", "t2")} for method in ("Fit-first", "Q002S640", "Q004S640")}
    cache_files = [tmp_path / name for name in ("native_maps.npz", "geometry.npz", "native_manifest.json")]
    for path in cache_files:
        path.write_bytes(b"cache")
    selected = {"group_idx": np.array([1, 2, 3]), "source_support": np.ones((3, 2, 3), bool), "affine_lps_rc": np.repeat(np.eye(4)[None], 3, axis=0), "pixel_spacing_rc_mm": np.ones((3, 2)), "t1_ms": np.full((3, 2, 3), 1000.0), "t2_ms": np.full((3, 2, 3), 80.0), "provenance": dict(zip(("native_maps_path", "geometry_path", "manifest_path"), map(str, cache_files)))}
    monkeypatch.setattr(reporting, "_load_all_verified_native_cache_groups", lambda *args: selected)
    monkeypatch.setattr(reporting, "_heldout_ids", lambda sparse, stack: [2])
    monkeypatch.setattr(reporting, "reproject_binary_mask_to_native_plane", lambda *args: np.ones((2, 3), bool))
    monkeypatch.setattr(reporting, "reproject_volume_to_native_plane", lambda values, affine, plane, shape: (np.full(shape, 900.0), np.ones(shape, bool)))
    config = {"subject": "CYJ", "condition": "L50_fold0", "endpoint": "central", "methods": maps, "l100_masks": {"t1": str(mask), "t2": str(mask)}, "native": {"mode": "all_source_groups", "stack": "sax", "sparse_manifest": str(sparse_manifest), "r0_native_maps_root": str(tmp_path)}}
    records = reporting._render_native(config, tmp_path, {})
    assert all(record["original_group_ids"] == [1, 2, 3] for record in records)
    assert all(record["displayed_groups"] == [{"group_idx": 1, "status": "observed", "label": "group 01 [observed]"}, {"group_idx": 2, "status": "held-out", "label": "group 02 [held-out]"}, {"group_idx": 3, "status": "observed", "label": "group 03 [observed]"}] for record in records)
