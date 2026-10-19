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
    from paper_experiments.joint_vs_2dfit.phase2_clean.final_cyj_reporting import longitudinal_plane, final_panel_specifications

    mask = np.ones((3, 4, 5), bool)
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    plane = longitudinal_plane(mask, affine, axis=1, index=2)
    assert plane["world"].shape == (4, 3, 5)
    assert plane["extent_mm"] == [0.0, 4.0, 0.0, 16.0]
    specs = final_panel_specifications("T1", ("Q002S640", "Fit-first"))
    assert all(spec["cmap"].name == specs[0]["cmap"].name and spec["range_ms"] == (0.0, 2500.0) for spec in specs)
    assert {spec["displayed_quantity"] for spec in specs if spec["kind"] == "absolute_error_ms"} == {"absolute_error_ms"}


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
    monkeypatch.setattr(reporting, "_render_native", lambda config, output, inputs: [{"display_support": "frozen L100", "interpolation": {"map": "linear", "mask": "nearest-neighbor"}, "frozen_l100_source": {"path": "mask", "sha256": "x"}}])
    output = reporting.render_final_figures(config, tmp_path / "figures")
    manifest = json.loads((output / "figure_manifest.json").read_text())
    assert {"subject", "condition", "endpoint", "methods", "inputs", "longitudinal_figures", "native_heldout_figures", "shared_parameter_lut_ranges_ms", "error_display_contract", "code_git_sha"}.issubset(manifest)
    assert manifest["endpoint"] == "psf128" and "same parameter map LUT/range" in manifest["error_display_contract"]
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
