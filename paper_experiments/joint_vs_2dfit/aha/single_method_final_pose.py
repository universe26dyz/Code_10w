"""Immutable single-method final-pose AHA16 and myocardial diagnostics.

This module deliberately evaluates exactly one method against the supplied
current native cache.  It is an additive QA product: it never substitutes an
endpoint volume, never reads a historical analysis root, and never represents
itself as a paired Proposed-vs-Fit-first result.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..phase2_clean.contracts import sha256, write_immutable_json
from ..phase2_clean.evaluation import PSNR_DATA_RANGES_MS, _metrics, masked_ssim


FIT_FIRST_SCHEMAS = {
    "fit_first_native_prediction_bundle/v2",
    "fit_first_native_prediction_bundle/v4",
    "fit_first_native_prediction_bundle/v6",
    "cyj11_r3_fit_first_final_pose_native_bundle/v1",
}
PROPOSED_SCHEMAS = {
    "direct_native_prediction_bundle/v1",
    "direct_sparse_native_prediction_bundle/v1",
    "cyj11_r3_direct_final_pose_native_bundle/v1",
}
SUPPORT_EQUATION = "manual anatomical ROI AND finite(reference) AND method-final-pose-valid AND finite(prediction)"
SCHEMA = "single_method_final_pose_diagnostic/v1"


def _record(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _manifest(bundle_root: Path, method: str) -> dict[str, Any]:
    path = bundle_root / "manifest.json"
    if method not in ("fit_first", "proposed"):
        raise ValueError("method must be fit_first or proposed")
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = FIT_FIRST_SCHEMAS if method == "fit_first" else PROPOSED_SCHEMAS
    if payload.get("schema") not in expected or payload.get("endpoint") != "central":
        raise ValueError(f"{method} final-pose native bundle schema/endpoint mismatch")
    return payload


def _native_reference(native: Mapping[str, np.ndarray], group: int, parameter: str) -> np.ndarray:
    if parameter not in ("t1", "t2"):
        raise ValueError("parameter must be t1 or t2")
    groups = [int(value) for value in np.asarray(native["group_idx"], int)]
    if group not in groups:
        raise ValueError(f"native SAX group {group} is absent")
    return np.asarray(native[f"{parameter}_ms"][groups.index(group)], float)


def load_single_method_plane(*, native: Mapping[str, np.ndarray], bundle_root: str | Path,
                             method: str, group: int, parameter: str) -> dict[str, np.ndarray]:
    """Load one current final-pose SAX plane in the native pixel ordering."""
    root = Path(bundle_root)
    manifest = _manifest(root, method)
    reference = _native_reference(native, group, parameter)
    if method == "fit_first":
        path = root / "planes" / parameter / "sax" / f"group{group:03d}.npz"
        if not path.is_file():
            raise ValueError(f"Fit-first final-pose SAX group {group}/{parameter} is unavailable")
        with np.load(path, allow_pickle=False) as data:
            required = {"prediction_ms", "volume_geometry_valid", "query_effective_mask", "valid"}
            if not required.issubset(data.files):
                raise ValueError("Fit-first final-pose plane arrays are incomplete")
            prediction = np.asarray(data["prediction_ms"], float)
            derived = (np.asarray(data["volume_geometry_valid"], bool)
                       & np.asarray(data["query_effective_mask"], bool) & np.isfinite(prediction))
            valid = np.asarray(data["valid"], bool)
            if valid.shape != derived.shape or not np.array_equal(valid, derived):
                raise ValueError("Fit-first final-pose valid mask violates bundle contract")
    else:
        record = next((item for item in manifest.get("planes", [])
                       if isinstance(item, dict) and item.get("stack") == "sax"
                       and item.get("original_group_idx") == int(group)), None)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError(f"Proposed final-pose SAX group {group} is unavailable")
        path = root / record["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            key = f"{parameter}_ms"
            if key not in data.files:
                raise ValueError("Proposed final-pose plane arrays are incomplete")
            prediction = np.asarray(data[key], float)
        valid = np.isfinite(prediction)
    if not (reference.shape == prediction.shape == valid.shape):
        raise ValueError("final-pose native bundle/native geometry mismatch")
    return {"reference": reference, "prediction": prediction, "valid": valid}


def _distribution(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, float)
    if not values.size:
        return {"n": 0, "mean": None, "sample_sd": None}
    return {"n": int(values.size), "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)) if values.size > 1 else 0.0}


def single_method_statistics(*, reference: np.ndarray, prediction: np.ndarray, valid: np.ndarray,
                             roi: np.ndarray, parameter: str) -> dict[str, Any]:
    """Evaluate one method on the exact ROI/final-pose common support."""
    reference, prediction = np.asarray(reference, float), np.asarray(prediction, float)
    valid, roi = np.asarray(valid, bool), np.asarray(roi, bool)
    if not (reference.shape == prediction.shape == valid.shape == roi.shape):
        raise ValueError("single-method ROI statistics require equal-shape arrays")
    support = roi & np.isfinite(reference) & valid & np.isfinite(prediction)
    result = {"support": support, "support_voxels": int(support.sum()),
              "reference": _distribution(reference[support]), "prediction": _distribution(prediction[support])}
    if not support.any():
        result["metrics"] = None
        return result
    ssim, status = masked_ssim(reference, prediction, support,
                               data_range=PSNR_DATA_RANGES_MS[parameter])
    result["metrics"] = _metrics(reference, prediction, support,
                                 psnr_data_range=PSNR_DATA_RANGES_MS[parameter],
                                 ssim_value=ssim, ssim_status=status)
    return result


def _landmark(points: np.ndarray, labels: tuple[int, ...], name: str) -> tuple[float, float]:
    selected = np.isin(points, labels)
    if not selected.any():
        raise ValueError(f"manual point labels lack {name} landmarks")
    row, column = np.where(selected)
    return float(column.mean()), float(row.mean())


def _load_scmr_masks(root: Path, groups: np.ndarray, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    from .final5_myocardial_roi_metrics import ROI_NAMES

    manifest, archive = root / "manifest.json", root / "sax_myocardium_masks.npz"
    if not manifest.is_file() or not archive.is_file():
        raise FileNotFoundError("SCMR myocardium mask bundle is incomplete")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema") != "exact_native_myocardium_bundle/v2" or payload.get("status") != "PASS":
        raise ValueError("SCMR myocardium mask bundle is not PASS exact_native_myocardium_bundle/v2")
    with np.load(archive, allow_pickle=False) as data:
        if any(name not in data.files for name in ("group_idx", *ROI_NAMES)):
            raise ValueError("SCMR myocardium mask bundle arrays are incomplete")
        if not np.array_equal(np.asarray(data["group_idx"], int), groups):
            raise ValueError("SCMR myocardium mask bundle group identity mismatch")
        result = {name: np.asarray(data[name], bool) for name in ROI_NAMES}
    expected = (len(groups), *shape)
    if any(value.shape != expected for value in result.values()):
        raise ValueError("SCMR myocardium mask bundle/native geometry mismatch")
    return result


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ("subject", "case", "method", "parameter", "level", "segment", "n",
              "reference_mean", "reference_sd", "prediction_mean", "prediction_sd",
              "signed_difference", "absolute_difference")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def build_single_method_final_pose(*, subject: str, case: str, method: str,
                                   native_sax_root: str | Path, prepared_sax: str | Path,
                                   mapping_path: str | Path, roi_bundle: str | Path,
                                   native_bundle: str | Path, output: str | Path) -> Path:
    """Write immutable AHA16 and myocardial diagnostics from explicit current inputs."""
    native_root, prepared, mapping_path, roi_root, bundle_root, output = (
        Path(value) for value in (native_sax_root, prepared_sax, mapping_path, roi_bundle, native_bundle, output))
    from .cyj11_adapter import load_cyj11_native_sax, require_cyj11_mapping
    from .cyj11_legacy_roi import load_cyj11_roi_plane
    from .final5_myocardial_roi_metrics import ROI_NAMES
    from .legacy_aha16 import (LEVELS, aha16_labels, build_filled_masks,
                               render_scmr_figure3, render_single_bullseye,
                               segment_statistics, validate_aha16_statistics)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    mapping = require_cyj11_mapping(mapping_path, subject=subject, case=case,
                                    prepared_sax_path=prepared,
                                    roi_bundle_path=roi_root / "roi_derivation_manifest.json")
    if {item["level"] for item in mapping["planes"]} != set(LEVELS):
        raise ValueError("AHA16 requires exactly basal, mid, and apical mapping planes")
    native = load_cyj11_native_sax(native_root, prepared)
    bundle_manifest = _manifest(bundle_root, method)
    groups = np.asarray(native["group_idx"], int)
    group_indices = {int(value): index for index, value in enumerate(groups)}
    masks = _load_scmr_masks(roi_root, groups, tuple(map(int, native["t1_ms"].shape[1:])))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        aha_rows: list[dict[str, Any]] = []
        per_plane: list[dict[str, Any]] = []
        figures = {parameter: {} for parameter in ("t1", "t2")}
        references = {parameter: {} for parameter in ("t1", "t2")}
        for entry in mapping["planes"]:
            group = int(entry["native_group_idx"])
            if group not in group_indices:
                raise ValueError(f"mapping group {group} is absent from current native SAX")
            index = group_indices[group]
            shape = tuple(map(int, native["t1_ms"][index].shape))
            roi = load_cyj11_roi_plane(roi_root, group=group, expected_shape=shape)
            anatomical = build_filled_masks(roi["outer"], roi["inner"],
                                            tuple(map(float, native["pixel_spacing_rc_mm"][index][:2])))
            anterior = _landmark(roi["points"], (1, 3, 5), "anterior")
            inferior = _landmark(roi["points"], (2, 4, 6), "inferior")
            if not (np.allclose(anterior, entry["anterior_xy"], atol=.5, rtol=0)
                    and np.allclose(inferior, entry["inferior_xy"], atol=.5, rtol=0)):
                raise ValueError("mapping landmark coordinates disagree with current ROI bundle")
            labels = aha16_labels({entry["level"]: anatomical["myocardium_core"],
                                   **{level: anatomical["myocardium_core"] for level in LEVELS if level != entry["level"]}},
                                  {entry["level"]: (anterior, inferior),
                                   **{level: (anterior, inferior) for level in LEVELS if level != entry["level"]}})[entry["level"]]
            report: dict[str, Any] = {key: entry[key] for key in ("level", "native_group_idx", "roi_slice_index", "roi_transform")}
            report["parameters"] = {}
            segment_count, first_segment = LEVELS[entry["level"]]
            segments = range(first_segment, first_segment + segment_count)
            for parameter in ("t1", "t2"):
                plane = load_single_method_plane(native=native, bundle_root=bundle_root, method=method,
                                                 group=group, parameter=parameter)
                stats = single_method_statistics(reference=plane["reference"], prediction=plane["prediction"],
                                                 valid=plane["valid"], roi=anatomical["myocardium_core"], parameter=parameter)
                segments_stats = segment_statistics(plane["reference"], plane["prediction"], labels, stats["support"], segments=segments)
                report["parameters"][parameter] = {"support_voxels": stats["support_voxels"],
                                                     "statistics": {key: value for key, value in stats.items() if key != "support"},
                                                     "segments": segments_stats}
                for segment, values in segments_stats.items():
                    aha_rows.append({"subject": subject, "case": case, "method": method, "parameter": parameter,
                                     "level": entry["level"], "segment": segment, **values})
                    figures[parameter][segment] = values["prediction_mean"]
                    references[parameter][segment] = values["reference_mean"]
            per_plane.append(report)
        aha_root = staging / "aha16"
        for parameter in ("t1", "t2"):
            render_single_bullseye(aha_root / "figures" / subject / case / "reference" / f"{parameter}_aha16.png",
                                   references[parameter], parameter=parameter,
                                   title=f"{subject} {case} native {parameter.upper()} AHA16")
            render_single_bullseye(aha_root / "figures" / subject / case / method / f"{parameter}_aha16.png",
                                   figures[parameter], parameter=parameter,
                                   title=f"{subject} {case} {method.replace('_', '-')} {parameter.upper()} AHA16")
        _write_csv(aha_rows, aha_root / "aha16_segment_statistics.csv")
        validation = validate_aha16_statistics(aha_rows)
        write_immutable_json(aha_root / "aha16_validation.json", validation)
        if validation["status"] == "PASS":
            render_scmr_figure3(aha_root / "figures" / subject / case / method / "Figure3_bullseye_T1_T2.png",
                                reference=references, prediction=figures, subject=subject, case=case, method=method)

        myocardial_root = staging / "myocardial_roi"
        reports: list[dict[str, Any]] = []
        pooled = {parameter: {name: {"reference": [], "prediction": [], "ssim": []} for name in ROI_NAMES}
                  for parameter in ("t1", "t2")}
        for index, group_value in enumerate(groups):
            group = int(group_value); report: dict[str, Any] = {"native_group_idx": group, "parameters": {}}
            for parameter in ("t1", "t2"):
                plane = load_single_method_plane(native=native, bundle_root=bundle_root, method=method,
                                                 group=group, parameter=parameter)
                parameter_report: dict[str, Any] = {}
                for name in ROI_NAMES:
                    stats = single_method_statistics(reference=plane["reference"], prediction=plane["prediction"],
                                                     valid=plane["valid"], roi=masks[name][index], parameter=parameter)
                    parameter_report[name] = {"roi_voxels": int(masks[name][index].sum()),
                                              "coverage": float(stats["support_voxels"] / masks[name][index].sum()) if masks[name][index].any() else None,
                                              "statistics": {key: value for key, value in stats.items() if key != "support"}}
                    if stats["support"].any():
                        pooled[parameter][name]["reference"].append(plane["reference"][stats["support"]])
                        pooled[parameter][name]["prediction"].append(plane["prediction"][stats["support"]])
                        ssim = stats["metrics"].get("ssim") if stats["metrics"] else None
                        if ssim is not None:
                            pooled[parameter][name]["ssim"].append(float(ssim))
                report["parameters"][parameter] = parameter_report
            reports.append(report)
        pooled_report: dict[str, Any] = {}
        for parameter, by_roi in pooled.items():
            pooled_report[parameter] = {}
            for name, values in by_roi.items():
                if not values["reference"]:
                    pooled_report[parameter][name] = {"status": "EMPTY_SUPPORT", "support_voxels": 0}
                    continue
                reference, prediction = np.concatenate(values["reference"]), np.concatenate(values["prediction"])
                metric = _metrics(reference, prediction, np.ones(reference.shape, bool),
                                  psnr_data_range=PSNR_DATA_RANGES_MS[parameter],
                                  ssim_value=(float(np.mean(values["ssim"])) if values["ssim"] else None),
                                  ssim_status=("mean_of_per_plane_common_support_ssim" if values["ssim"] else "not_available_2d_support"))
                pooled_report[parameter][name] = {"reference": _distribution(reference),
                                                   "prediction": _distribution(prediction), "metrics": metric}
        write_immutable_json(myocardial_root / "per_plane_statistics.json", {"schema": SCHEMA, "subject": subject, "case": case, "method": method, "sampling": "final_pose", "support_equation": SUPPORT_EQUATION, "planes": reports})
        write_immutable_json(myocardial_root / "pooled_sax_statistics.json", {"schema": SCHEMA, "subject": subject, "case": case, "method": method, "sampling": "final_pose", "support_equation": SUPPORT_EQUATION, "pooled_sax": pooled_report})
        write_immutable_json(staging / "manifest.json", {
            "schema": SCHEMA, "subject": subject, "case": case, "method": method,
            "diagnostic_only": True, "comparison": "native_reference_vs_single_method",
            "not_a_proposed_vs_fit_first_result": True, "sampling": "final_pose",
            "support_equation": SUPPORT_EQUATION, "mapping": _record(mapping_path),
            "roi_bundle": _record(roi_root / "roi_derivation_manifest.json"),
            "prepared_sax": _record(prepared),
            "native_sax": {name: _record(native_root / name) for name in ("native_maps.npz", "geometry.npz", "manifest.json")},
            "method_final_pose_bundle": _record(bundle_root / "manifest.json"),
            "method_bundle_schema": bundle_manifest["schema"], "aha16_validation": validation,
            "downstream_only": True, "no_reconstruction": True, "no_training": True, "immutable_output": True,
        })
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True); parser.add_argument("--case", required=True)
    parser.add_argument("--method", choices=("fit_first", "proposed"), required=True)
    parser.add_argument("--native-sax-root", required=True); parser.add_argument("--prepared-sax", required=True)
    parser.add_argument("--mapping", required=True); parser.add_argument("--roi-bundle", required=True)
    parser.add_argument("--native-bundle", required=True); parser.add_argument("--output", required=True)
    print(build_single_method_final_pose(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
