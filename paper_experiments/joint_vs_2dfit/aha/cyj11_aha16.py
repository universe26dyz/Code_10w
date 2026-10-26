"""CYJ1.1 immutable AHA16 output on the unified-root native SAX lattice."""
from __future__ import annotations

import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..phase2_clean.contracts import sha256, write_immutable_json
from .cyj11_adapter import cyj11_native_stack_root, require_cyj11_mapping
from .cyj11_legacy_roi import load_cyj11_roi_plane
from .cyj11_runner import load_cyj11_run_inputs, native_sax_plane
from .legacy_aha16 import (LEVELS, aha16_labels, build_filled_masks, render_scmr_figure3,
                           render_single_bullseye, segment_statistics, validate_aha16_statistics)


SUPPORT_EQUATION = "manual anatomical ROI AND finite(reference) AND proposed-valid AND fit-first-valid AND finite(proposed, fit-first)"


def build_paired_aha16_plane(*, level: str, reference: np.ndarray, proposed: np.ndarray, fit_first: np.ndarray,
                             proposed_valid: np.ndarray, fit_first_valid: np.ndarray, myocardium_core: np.ndarray,
                             anterior_xy: tuple[float, float], inferior_xy: tuple[float, float]) -> dict[str, Any]:
    if level not in LEVELS:
        raise ValueError("AHA level must be basal, mid, or apical")
    reference, proposed, fit_first = (np.asarray(value, float) for value in (reference, proposed, fit_first))
    proposed_valid, fit_first_valid, myocardium_core = (np.asarray(value, bool) for value in (proposed_valid, fit_first_valid, myocardium_core))
    if not (reference.shape == proposed.shape == fit_first.shape == proposed_valid.shape == fit_first_valid.shape == myocardium_core.shape):
        raise ValueError("CYJ1.1 AHA inputs require one native plane shape")
    labels = aha16_labels({level: myocardium_core, **{name: myocardium_core for name in LEVELS if name != level}},
                          {level: (anterior_xy, inferior_xy), **{name: (anterior_xy, inferior_xy) for name in LEVELS if name != level}})[level]
    support = myocardium_core & np.isfinite(reference) & proposed_valid & fit_first_valid & np.isfinite(proposed) & np.isfinite(fit_first)
    segments = range(LEVELS[level][1], LEVELS[level][1] + LEVELS[level][0])
    return {"labels": labels, "proposed_support": support, "fit_first_support": support,
            "proposed_statistics": segment_statistics(reference, proposed, labels, support, segments=segments),
            "fit_first_statistics": segment_statistics(reference, fit_first, labels, support, segments=segments),
            "support_equation": SUPPORT_EQUATION}


def _record(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _landmark(points: np.ndarray, labels: tuple[int, ...], name: str) -> tuple[float, float]:
    selected = np.isin(points, labels)
    if not selected.any():
        raise ValueError(f"CYJ1.1 manual point labels lack {name} landmark labels {list(labels)}")
    row, column = np.where(selected)
    return float(column.mean()), float(row.mean())


def _validate_mapping_landmarks(points: np.ndarray, entry: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    anterior, inferior = _landmark(points, (1, 3, 5), "anterior"), _landmark(points, (2, 4, 6), "inferior")
    if not (np.allclose(anterior, entry["anterior_xy"], atol=.5, rtol=0)
            and np.allclose(inferior, entry["inferior_xy"], atol=.5, rtol=0)):
        raise ValueError("CYJ1.1 reviewed mapping landmark coordinates disagree with manual point labels")
    return anterior, inferior


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing overwrite: {path}")
    fieldnames = tuple(rows[0]) if rows else ("subject", "case", "method", "parameter", "level", "segment", "n")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)


def build_cyj11_aha16(*, analysis_root: str | Path, subject: str, case: str,
                      mapping_path: str | Path, roi_bundle: str | Path, output: str | Path) -> Path:
    """Write a fresh downstream AHA16 bundle without changing central endpoints."""
    output, mapping_path, roi_bundle = Path(output), Path(mapping_path), Path(roi_bundle)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    mapping = require_cyj11_mapping(mapping_path, subject=subject, case=case,
                                    prepared_sax_path=Path(analysis_root) / "prepared" / subject / "sax" / "observations.npz",
                                    roi_bundle_path=roi_bundle / "roi_derivation_manifest.json")
    entries = mapping["planes"]
    if {entry["level"] for entry in entries} != set(LEVELS) or len(entries) != 3:
        raise ValueError("CYJ1.1 AHA16 requires exactly one explicit basal, mid, and apical mapping plane")
    native, endpoint_images = load_cyj11_run_inputs(analysis_root, subject, case)
    indices = {int(group): index for index, group in enumerate(native["group_idx"])}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        rows: list[dict[str, Any]] = []
        figures = {method: {parameter: {} for parameter in ("t1", "t2")} for method in ("proposed", "fit_first")}
        reference_figures = {parameter: {} for parameter in ("t1", "t2")}
        reports: list[dict[str, Any]] = []
        for entry in entries:
            group = int(entry["native_group_idx"])
            if group not in indices:
                raise ValueError(f"CYJ1.1 native SAX group {group} declared in mapping is absent")
            index = indices[group]
            shape = tuple(map(int, native["t1_ms"][index].shape))
            masks = load_cyj11_roi_plane(roi_bundle, group=group, expected_shape=shape)
            anatomical = build_filled_masks(masks["outer"], masks["inner"],
                                            tuple(map(float, native["pixel_spacing_rc_mm"][index][:2])))
            anterior, inferior = _validate_mapping_landmarks(masks["points"], entry)
            report: dict[str, Any] = {key: entry[key] for key in ("level", "native_group_idx", "roi_slice_index", "roi_transform")}
            report["parameters"] = {}
            for parameter in ("t1", "t2"):
                plane = native_sax_plane(native=native, group=group, endpoint_images=endpoint_images, parameter=parameter)
                result = build_paired_aha16_plane(level=entry["level"], myocardium_core=anatomical["myocardium_core"],
                                                  anterior_xy=anterior, inferior_xy=inferior, **plane)
                report["parameters"][parameter] = {key: value for key, value in result.items()
                                                    if key not in ("labels", "proposed_support", "fit_first_support")}
                for method in ("proposed", "fit_first"):
                    for segment, values in result[f"{method}_statistics"].items():
                        figures[method][parameter][segment] = values["prediction_mean"]
                        reference_figures[parameter][segment] = values["reference_mean"]
                        rows.append({"subject": subject, "case": case, "method": method, "parameter": parameter,
                                     "level": entry["level"], "segment": segment, **values})
            reports.append(report)
        for method in ("proposed", "fit_first"):
            for parameter in ("t1", "t2"):
                render_single_bullseye(staging / "figures" / subject / case / method / f"{parameter}_aha16.png",
                                       figures[method][parameter], parameter=parameter,
                                       title=f"{subject} {case} {method.replace('_', '-')} {parameter.upper()} AHA16")
        _write_csv(rows, staging / "aha16_segment_statistics.csv")
        validation = validate_aha16_statistics(rows)
        write_immutable_json(staging / "aha16_validation.json", validation)
        if validation["status"] == "PASS":
            for method in ("proposed", "fit_first"):
                render_scmr_figure3(staging / "figures" / subject / case / method / "Figure3_bullseye_T1_T2.png",
                                    reference=reference_figures, prediction=figures[method], subject=subject,
                                    case=case, method=method)
        root = Path(analysis_root)
        write_immutable_json(staging / "manifest.json", {
            "schema": "cyj11_aha16_downstream/v1", "subject": subject, "case": case,
            "mapping": _record(mapping_path), "roi_bundle": _record(roi_bundle / "roi_derivation_manifest.json"),
            "prepared_sax": _record(root / "prepared" / subject / "sax" / "observations.npz"),
            "native_sax": {name: _record(cyj11_native_stack_root(root, subject) / name)
                           for name in ("native_maps.npz", "geometry.npz", "manifest.json")},
            "endpoint_files": {parameter: {method: _record(image.get_filename()) for method, image in images.items()}
                               for parameter, images in endpoint_images.items()},
            "planes": reports, "support_equation": SUPPORT_EQUATION, "aha16_validation": validation,
            "algorithm": "historical AHA16 basal6/mid6/apical4; no segment17; historical orientation and sample SD",
            "downstream_only": True, "no_reconstruction": True, "no_formal_metric": True, "immutable_output": True,
        })
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output
