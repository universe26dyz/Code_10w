"""Immutable Final5 AHA16 execution on current central endpoints only."""
from __future__ import annotations

import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..phase2_clean.contracts import sha256, write_immutable_json
from ..phase2_clean.final_sparse_native_metrics import validate_sparse_endpoint_roots
from .final5_adapter import ROI_TRANSFORMS, require_exact_native_roi_mapping, resolve_subject_roi
from .legacy_aha16 import LEVELS, build_filled_masks, render_single_bullseye, sectorize, segment_statistics


def require_new_aha_output(output: str | Path) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    return output


def build_paired_aha16_plane(*, level: str, reference: np.ndarray, proposed: np.ndarray, fit_first: np.ndarray,
                             proposed_valid: np.ndarray, fit_first_valid: np.ndarray, myocardium_core: np.ndarray,
                             anterior_xy: tuple[float, float], inferior_xy: tuple[float, float],
                             parameter: str | None = None) -> dict[str, Any]:
    """Apply one AHA partition and one paired-valid support to both methods."""
    if level not in LEVELS:
        raise ValueError(f"unknown AHA16 level: {level}")
    reference, proposed, fit_first = (np.asarray(value, float) for value in (reference, proposed, fit_first))
    proposed_valid, fit_first_valid, myocardium_core = (np.asarray(value, bool) for value in (proposed_valid, fit_first_valid, myocardium_core))
    if not (reference.shape == proposed.shape == fit_first.shape == proposed_valid.shape == fit_first_valid.shape == myocardium_core.shape) or reference.ndim != 2:
        raise ValueError("paired AHA16 input requires equal 2-D native-plane arrays")
    n_segments, first_segment = LEVELS[level]
    labels = sectorize(myocardium_core, anterior_xy, inferior_xy, n_segments, first_segment)
    support = myocardium_core & np.isfinite(reference) & proposed_valid & fit_first_valid & np.isfinite(proposed) & np.isfinite(fit_first)
    segments = range(first_segment, first_segment + n_segments)
    return {
        "labels": labels, "paired_support": support,
        # Named twice to make the invariant conspicuous in serialized/tested
        # method records: neither method receives a private support.
        "proposed_support": support, "fit_first_support": support,
        "proposed_statistics": segment_statistics(reference, proposed, labels, support, segments=segments, parameter=parameter),
        "fit_first_statistics": segment_statistics(reference, fit_first, labels, support, segments=segments, parameter=parameter),
        "paired_support_equation": "manual_myocardium_core AND finite(reference) AND proposed_reprojection_valid AND fit_first_reprojection_valid AND finite(proposed, fit_first)",
    }


def _record(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _native_sax(native_sax_root: str | Path) -> dict[str, np.ndarray]:
    root = Path(native_sax_root)
    maps_path, geometry_path, manifest_path = root / "native_maps.npz", root / "geometry.npz", root / "manifest.json"
    if not all(path.is_file() for path in (maps_path, geometry_path, manifest_path)):
        raise FileNotFoundError("Final5 AHA16 requires explicit sax/native_maps.npz, geometry.npz, and manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("native_maps_sha256") != sha256(maps_path) or manifest.get("geometry_sha256") != sha256(geometry_path):
        raise ValueError("native SAX map/geometry provenance SHA256 mismatch")
    with np.load(maps_path, allow_pickle=False) as maps, np.load(geometry_path, allow_pickle=False) as geometry:
        required = ("group_idx", "t1_ms", "t2_ms")
        if any(name not in maps for name in required) or "affine_lps_rc" not in geometry or "pixel_spacing_rc_mm" not in geometry:
            raise ValueError("native SAX artifacts lack required map/geometry arrays")
        result = {name: np.asarray(maps[name]) for name in required}
        result.update({name: np.asarray(geometry[name]) for name in ("affine_lps_rc", "pixel_spacing_rc_mm")})
    count = len(result["group_idx"])
    if len(set(map(int, result["group_idx"]))) != count or any(value.shape[0] != count for value in result.values()):
        raise ValueError("native SAX group identity or dimensions are inconsistent")
    return result


def _roi_slice(image: Any, index: int, transform: str, expected_shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(image.get_fdata())
    if values.ndim != 3 or not isinstance(index, int) or not 0 <= index < values.shape[2]:
        raise ValueError("manual ROI slice index is outside the explicit ROI volume")
    plane = values[:, :, index]
    transforms = {
        "identity": lambda value: value,
        "transpose": lambda value: value.T,
        "flipud": np.flipud,
        "fliplr": np.fliplr,
        "transpose_flipud": lambda value: np.flipud(value.T),
        "transpose_fliplr": lambda value: np.fliplr(value.T),
    }
    if transform not in ROI_TRANSFORMS:
        raise ValueError("manual ROI mapping uses an unsupported explicit orientation")
    plane = transforms[transform](plane)
    if plane.shape != expected_shape:
        raise ValueError("manual ROI/native SAX plane shape mismatch; no resampling is permitted")
    return np.asarray(plane)


def _landmark_from_point_labels(points: np.ndarray, labels: tuple[int, ...], name: str) -> tuple[float, float]:
    selected = np.isin(points, labels)
    if not selected.any():
        raise ValueError(f"manual point-label ROI lacks {name} landmark labels {list(labels)}")
    row, column = np.where(selected)
    return (float(column.mean()), float(row.mean()))


def _write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    keys = tuple(rows[0]) if rows else ("subject", "case", "method", "parameter", "level", "segment", "n")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def build_final5_aha16(*, final_root: str | Path, subject: str, case: str, roi_root: str | Path, mapping_path: str | Path,
                      native_sax_root: str | Path, output: str | Path) -> Path:
    """Build one immutable AHA16 downstream bundle; it never alters Final5 artifacts."""
    import nibabel as nib
    from .final5_adapter import final5_endpoint_paths
    from ..phase2_clean.final5_partitioned_native_metrics import _bundle_plane, _validate_bundles

    output = require_new_aha_output(output)
    roi = resolve_subject_roi(roi_root, subject)
    if roi is None:
        raise ValueError("ROI_PENDING: subject-specific audited manual ROI is unavailable")
    mapping = require_exact_native_roi_mapping(mapping_path, subject=subject, case=case, roi_root=roi)
    entries = mapping["planes"]
    if {entry["level"] for entry in entries} != set(LEVELS) or len(entries) != 3:
        raise ValueError("AHA16 requires exactly one explicit SAX plane for basal, mid, and apical")
    endpoints = final5_endpoint_paths(final_root, subject, case)
    _volume_paths, endpoint_provenance = validate_sparse_endpoint_roots(endpoints["proposed"], endpoints["fit_first"], case, subject=subject)
    central = Path(final_root) / "05_formal_endpoints" / "central"
    direct_root, fit_root = central / "proposed_native_bundle" / subject / case, central / "fit_first_native_bundle" / subject / case
    direct_bundle, fit_bundle, _fit_endpoint_source_provenance = _validate_bundles(
        direct_root=direct_root, fit_root=fit_root, subject=subject, case=case,
        fit_endpoint=endpoints["fit_first"])
    outer, inner, points, composite = (nib.load(str(roi / name)) for name in (
        "Segmentation_1-out-label.nii.gz", "Segmentation_2-in-label.nii.gz", "Segmentation_3-label_point.nii.gz", "sax_composite_reference.nii.gz"))
    if any(image.shape != outer.shape or not np.allclose(image.affine, outer.affine, atol=1e-6, rtol=0) for image in (inner, points, composite)):
        raise ValueError("manual outer/inner/point/composite ROI geometry differs")
    native = _native_sax(native_sax_root)
    by_group = {int(group): index for index, group in enumerate(native["group_idx"])}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        rows: list[dict[str, Any]] = []
        figure_values = {method: {parameter: {} for parameter in ("t1", "t2")} for method in ("proposed", "fit_first")}
        planes: list[dict[str, Any]] = []
        for entry in entries:
            group = entry["native_group_idx"]
            if group not in by_group:
                raise ValueError(f"native SAX group {group} declared in mapping is absent")
            index = by_group[group]
            spacing = tuple(map(float, native["pixel_spacing_rc_mm"][index][:2]))
            reference_shape = tuple(map(int, native["t1_ms"][index].shape))
            masks = build_filled_masks(_roi_slice(outer, entry["roi_slice_index"], entry["roi_transform"], reference_shape) > 0,
                                       _roi_slice(inner, entry["roi_slice_index"], entry["roi_transform"], reference_shape) > 0,
                                       spacing)
            point_plane = _roi_slice(points, entry["roi_slice_index"], entry["roi_transform"], reference_shape)
            anterior = _landmark_from_point_labels(point_plane, (1, 3, 5), "anterior")
            inferior = _landmark_from_point_labels(point_plane, (2, 4, 6), "inferior")
            if not (np.allclose(anterior, entry["anterior_xy"], atol=.5, rtol=0) and np.allclose(inferior, entry["inferior_xy"], atol=.5, rtol=0)):
                raise ValueError("reviewed mapping landmark coordinates disagree with the subject's manual point labels")
            plane_report: dict[str, Any] = {"level": entry["level"], "native_group_idx": group, "roi_slice_index": entry["roi_slice_index"], "roi_transform": entry["roi_transform"], "parameters": {}}
            for parameter in ("t1", "t2"):
                reference = np.asarray(native[f"{parameter}_ms"][index], float)
                proposed, proposed_valid = _bundle_plane(direct_bundle, direct_root, parameter=parameter, stack="sax", group=int(group), method="proposed")
                fit_first, fit_first_valid = _bundle_plane(fit_bundle, fit_root, parameter=parameter, stack="sax", group=int(group), method="fit_first")
                if proposed.shape != reference.shape or fit_first.shape != reference.shape:
                    raise ValueError("final-pose AHA native bundle plane shape mismatch")
                result = build_paired_aha16_plane(level=entry["level"], reference=reference,
                                                  proposed=proposed, fit_first=fit_first,
                                                  proposed_valid=proposed_valid, fit_first_valid=fit_first_valid,
                                                  myocardium_core=masks["myocardium_core"], anterior_xy=anterior, inferior_xy=inferior,
                                                  parameter=parameter)
                plane_report["parameters"][parameter] = {key: value for key, value in result.items() if key not in ("labels", "paired_support", "proposed_support", "fit_first_support")}
                for method in ("proposed", "fit_first"):
                    stats = result[f"{method}_statistics"]
                    for segment, values in stats.items():
                        figure_values[method][parameter][segment] = values["prediction_mean"]
                        rows.append({"subject": subject, "case": case, "method": method, "parameter": parameter, "level": entry["level"], "segment": segment, **values})
            planes.append(plane_report)
        for method in ("proposed", "fit_first"):
            for parameter in ("t1", "t2"):
                render_single_bullseye(staging / "figures" / subject / case / method / f"{parameter}_aha16.png", figure_values[method][parameter], parameter=parameter, title=f"{subject} {case} {method.replace('_', '-')} {parameter.upper()} AHA16")
        _write_csv(rows, staging / "aha16_segment_statistics.csv")
        write_immutable_json(staging / "manifest.json", {
            "schema": "final5_aha16_downstream/v1", "subject": subject, "case": case,
            "endpoint_provenance": endpoint_provenance, "native_sax": {"root": str(Path(native_sax_root).resolve()), "maps": _record(Path(native_sax_root) / "native_maps.npz"), "geometry": _record(Path(native_sax_root) / "geometry.npz"), "manifest": _record(Path(native_sax_root) / "manifest.json")},
            "manual_roi": {name: _record(roi / name) for name in ("Segmentation_1-out-label.nii.gz", "Segmentation_2-in-label.nii.gz", "Segmentation_3-label_point.nii.gz", "sax_composite_reference.nii.gz")},
            "roi_derivation": _record(roi / "roi_derivation_manifest.json") if (roi / "roi_derivation_manifest.json").is_file() else None,
            "mapping": _record(Path(mapping_path)), "planes": planes,
            "native_sampling": "SAX is observed in every Final5 sparse profile; Proposed and Fit-first use their final learned-pose native bundles, never endpoint-to-R0 reprojection",
            "proposed_native_bundle": _record(direct_root / "manifest.json"), "fit_first_native_bundle": _record(fit_root / "manifest.json"),
            "support_equation": "manual_myocardium_core AND finite(reference) AND proposed_final_pose_valid AND fit_first_final_pose_valid AND finite(proposed, fit_first)",
            "metric_contract": "RMSE, NRMSE, bias, MAE, Pearson CC, NCC, fixed-range PSNR, and mean masked 2-D SSIM per AHA segment",
            "psnr_data_ranges_ms": {"t1": 2500.0, "t2": 200.0},
            "algorithm": "historical AHA16 basal6/mid6/apical4; no segment17; historical anterior/inferior orientation and sample SD", "downstream_only": True,
        })
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output
