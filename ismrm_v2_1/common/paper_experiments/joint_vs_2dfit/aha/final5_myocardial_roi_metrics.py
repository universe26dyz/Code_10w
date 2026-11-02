"""Immutable Final5-only local CPU myocardial ROI statistics on native SAX planes.

This downstream report reads only explicit Final5 inputs.  It never changes a
formal endpoint, reconstruction, training input, or formal metric artifact.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..phase2_clean.contracts import sha256, write_immutable_json
from ..phase2_clean.evaluation import PSNR_DATA_RANGES_MS, _metrics, masked_ssim
from ..phase2_clean.final5_partitioned_native_metrics import _bundle_plane, _validate_bundles
from ..phase2_clean.final_sparse_native_metrics import validate_sparse_endpoint_roots
from .final5_adapter import (assess_final5_aha_readiness, final5_endpoint_paths,
                             require_exact_native_roi_mapping,
                             resolve_subject_roi)
from .final5_aha16 import _native_sax, _roi_slice
from .legacy_aha16 import build_filled_masks


SCHEMA = "final5_local_myocardial_roi_metrics/v2"
ROI_NAMES = ("myocardium_full", "myocardium_core_legacy", "myocardium_core_1px")
PRIMARY_ROI = "myocardium_core_1px"
SUPPORT_EQUATION = "selected_named_myocardial_roi (primary=myocardium_core_1px) AND finite(reference) AND proposed_final_pose_valid AND fit_first_final_pose_valid AND finite(proposed, fit_first)"


def _record(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def require_new_external_output(final_root: str | Path, output: str | Path) -> Path:
    """Require a fresh downstream directory that cannot alter Final5 artifacts."""
    root, output = Path(final_root).resolve(), Path(output).resolve()
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("Final5 local myocardial ROI output must be outside the Final5 root")
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    return output


def build_myocardial_rois(outer: np.ndarray, inner: np.ndarray, *,
                          spacing_rc_mm: tuple[float, float]) -> dict[str, np.ndarray]:
    """Create separately named full, historical-mm-core, and 1-pixel-core ROIs."""
    from scipy.ndimage import binary_erosion

    historical = build_filled_masks(outer, inner, spacing_rc_mm)
    full = np.asarray(historical["myocardium"], bool)
    return {
        "myocardium_full": full,
        "myocardium_core_legacy": np.asarray(historical["myocardium_core"], bool),
        "myocardium_core_1px": np.asarray(
            binary_erosion(full, structure=np.ones((3, 3), bool), border_value=0), bool),
    }


def _distribution(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, float)
    if not values.size:
        return {"n": 0, "mean": None, "sample_sd": None}
    return {
        "n": int(values.size), "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
    }


def paired_roi_statistics(*, reference: np.ndarray, proposed: np.ndarray, fit_first: np.ndarray,
                          proposed_valid: np.ndarray, fit_first_valid: np.ndarray,
                          roi: np.ndarray, parameter: str | None = None) -> dict[str, Any]:
    """Summarize all three maps on one common native-pixel support."""
    reference, proposed, fit_first = (np.asarray(value, float) for value in (reference, proposed, fit_first))
    proposed_valid, fit_first_valid, roi = (np.asarray(value, bool) for value in (proposed_valid, fit_first_valid, roi))
    if not (reference.shape == proposed.shape == fit_first.shape == proposed_valid.shape == fit_first_valid.shape == roi.shape):
        raise ValueError("paired myocardial ROI statistics require equal-shape arrays")
    support = roi & np.isfinite(reference) & proposed_valid & fit_first_valid & np.isfinite(proposed) & np.isfinite(fit_first)
    values = {"reference": reference[support], "proposed": proposed[support], "fit_first": fit_first[support]}
    result = {name: _distribution(value) for name, value in values.items()}
    data_range = PSNR_DATA_RANGES_MS.get(str(parameter).lower()) if parameter is not None else None
    for method in ("proposed", "fit_first"):
        if not values[method].size:
            result[method].update({"bias": None, "mae": None, "rmse": None, "nrmse": None,
                                   "relative_nrmse": None, "pearson_r": None, "cc": None,
                                   "ncc": None, "psnr": None, "ssim": None,
                                   "ssim_status": "empty_common_support"})
            continue
        ssim_value, ssim_status = masked_ssim(reference, np.asarray({"proposed": proposed, "fit_first": fit_first}[method]), support,
                                               data_range=float(data_range or 1.0))
        metric = _metrics(reference, {"proposed": proposed, "fit_first": fit_first}[method], support,
                          psnr_data_range=data_range, ssim_value=ssim_value, ssim_status=ssim_status)
        result[method].update({key: metric[key] for key in (
            "bias", "mae", "rmse", "nrmse", "relative_nrmse", "pearson_r", "cc", "ncc",
            "psnr", "psnr_data_range_ms", "psnr_status", "ssim", "ssim_status",
        )})
    return {"common_support": support, "common_support_voxels": int(support.sum()), **result}


def mapped_roi_plane(outer_image: Any, inner_image: Any, *, entry: Mapping[str, Any],
                     expected_shape: tuple[int, int], spacing_rc_mm: tuple[float, float]) -> dict[str, Any]:
    """Map one audited ROI slice by its declared discrete orientation only."""
    outer = _roi_slice(outer_image, int(entry["roi_slice_index"]), str(entry["roi_transform"]), expected_shape) > 0
    inner = _roi_slice(inner_image, int(entry["roi_slice_index"]), str(entry["roi_transform"]), expected_shape) > 0
    return {
        "level": entry["level"], "native_group_idx": int(entry["native_group_idx"]),
        "original_group_idx": int(entry["original_group_idx"]),
        "roi_slice_index": int(entry["roi_slice_index"]), "roi_transform": str(entry["roi_transform"]),
        "rois": build_myocardial_rois(outer, inner, spacing_rc_mm=spacing_rc_mm),
    }


def assess_final5_myocardial_roi_readiness(final_root: str | Path, subject: str, case: str,
                                            roi_root: str | Path, *, mapping_path: str | Path | None = None,
                                            standardization_manifest_path: str | Path | None = None) -> dict[str, Any]:
    """Expose the established Final5 ROI-pending state without substitutions."""
    readiness = assess_final5_aha_readiness(final_root, subject, case, roi_root, mapping_path=None)
    if mapping_path is None or readiness["status"] != "NATIVE_MAPPING_PENDING":
        return readiness
    try:
        require_exact_native_roi_mapping(mapping_path, subject=subject, case=case, roi_root=readiness["roi_root"],
                                         standardization_manifest_path=standardization_manifest_path)
    except (FileNotFoundError, ValueError) as exc:
        return {**readiness, "status": "NATIVE_MAPPING_INVALID", "reason": str(exc)}
    return {**readiness, "mapping": json.loads(Path(mapping_path).read_text(encoding="utf-8"))}


def _pooled_statistics(values: dict[str, list[np.ndarray]], *, parameter: str | None = None,
                       ssim_values: dict[str, list[float]] | None = None) -> dict[str, Any]:
    reference = np.concatenate(values["reference"]) if values["reference"] else np.asarray([], float)
    proposed = np.concatenate(values["proposed"]) if values["proposed"] else np.asarray([], float)
    fit_first = np.concatenate(values["fit_first"]) if values["fit_first"] else np.asarray([], float)
    result = paired_roi_statistics(reference=reference, proposed=proposed, fit_first=fit_first,
                                   proposed_valid=np.ones(reference.shape, bool), fit_first_valid=np.ones(reference.shape, bool),
                                   roi=np.ones(reference.shape, bool), parameter=parameter)
    result.pop("common_support", None)
    if ssim_values is not None:
        for method in ("proposed", "fit_first"):
            values_for_method = ssim_values.get(method, [])
            result[method]["ssim"] = float(np.mean(values_for_method)) if values_for_method else None
            result[method]["ssim_status"] = "mean_of_per_plane_common_support_ssim" if values_for_method else "not_available_2d_support"
    return result


def build_final5_myocardial_roi_metrics(*, final_root: str | Path, subject: str, case: str,
                                         roi_root: str | Path, mapping_path: str | Path,
                                         native_sax_root: str | Path, output: str | Path,
                                         standardization_manifest_path: str | Path | None = None) -> Path:
    """Write one immutable, local-only Final5 native-SAX myocardial report."""
    import nibabel as nib

    final_root = Path(final_root)
    output = require_new_external_output(final_root, output)
    roi_root = Path(roi_root)
    roi = resolve_subject_roi(roi_root, subject)
    if roi is None:
        raise ValueError("ROI_PENDING: subject-specific audited manual ROI is unavailable")
    mapping_path = Path(mapping_path)
    mapping = require_exact_native_roi_mapping(
        mapping_path, subject=subject, case=case, roi_root=roi,
        standardization_manifest_path=standardization_manifest_path)
    endpoints = final5_endpoint_paths(final_root, subject, case)
    _volumes, endpoint_provenance = validate_sparse_endpoint_roots(
        endpoints["proposed"], endpoints["fit_first"], case, subject=subject)
    central = final_root / "05_formal_endpoints" / "central"
    direct_root = central / "proposed_native_bundle" / subject / case
    fit_root = central / "fit_first_native_bundle" / subject / case
    direct_bundle, fit_bundle, fit_endpoint_source_provenance = _validate_bundles(
        direct_root=direct_root, fit_root=fit_root, subject=subject, case=case,
        fit_endpoint=endpoints["fit_first"])
    outer_path, inner_path = roi / "Segmentation_1-out-label.nii.gz", roi / "Segmentation_2-in-label.nii.gz"
    outer, inner = nib.load(str(outer_path)), nib.load(str(inner_path))
    if outer.shape != inner.shape or not np.allclose(outer.affine, inner.affine, atol=1e-6, rtol=0):
        raise ValueError("manual outer/inner ROI geometry differs")
    native_root = Path(native_sax_root)
    native = _native_sax(native_root)
    by_group = {int(group): index for index, group in enumerate(native["group_idx"])}
    input_provenance = {
        "endpoint_provenance": endpoint_provenance,
        "fit_first_endpoint_source_manifest_provenance": fit_endpoint_source_provenance,
        "mapping": _record(mapping_path),
        "standardization_manifest": _record(
            Path(standardization_manifest_path) if standardization_manifest_path is not None
            else Path(str(mapping["standardization_manifest"]["path"]))),
        "mapping_declared_standardization_manifest": mapping["standardization_manifest"],
        "manual_roi": {"outer": _record(outer_path), "inner": _record(inner_path)},
        "roi_manifest": _record(roi / "roi_derivation_manifest.json") if (roi / "roi_derivation_manifest.json").is_file()
        else _record(roi / "final_aha_roi_provenance.json"),
        "native_sax": {"root": str(native_root.resolve()), "maps": _record(native_root / "native_maps.npz"),
                       "geometry": _record(native_root / "geometry.npz"), "manifest": _record(native_root / "manifest.json")},
        "proposed_native_bundle": _record(direct_root / "manifest.json"),
        "fit_first_native_bundle": _record(fit_root / "manifest.json"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        per_plane: list[dict[str, Any]] = []
        pooled = {parameter: {roi_name: {method: [] for method in ("reference", "proposed", "fit_first")}
                              for roi_name in ROI_NAMES} for parameter in ("t1", "t2")}
        pooled_ssim = {parameter: {roi_name: {method: [] for method in ("proposed", "fit_first")}
                                   for roi_name in ROI_NAMES} for parameter in ("t1", "t2")}
        for entry in mapping["planes"]:
            group = int(entry["native_group_idx"])
            if group not in by_group:
                raise ValueError(f"native SAX group {group} declared in mapping is absent")
            index = by_group[group]
            reference_shape = tuple(map(int, native["t1_ms"][index].shape))
            plane = mapped_roi_plane(outer, inner, entry=entry, expected_shape=reference_shape,
                                     spacing_rc_mm=tuple(map(float, native["pixel_spacing_rc_mm"][index][:2])))
            plane_report: dict[str, Any] = {key: plane[key] for key in ("level", "native_group_idx", "original_group_idx", "roi_slice_index", "roi_transform")}
            plane_report["roi_voxels"] = {name: int(mask.sum()) for name, mask in plane["rois"].items()}
            plane_report["input_provenance"] = input_provenance
            plane_report["parameters"] = {}
            for parameter in ("t1", "t2"):
                reference = np.asarray(native[f"{parameter}_ms"][index], float)
                proposed, proposed_valid = _bundle_plane(direct_bundle, direct_root, parameter=parameter, stack="sax", group=group, method="proposed")
                fit_first, fit_first_valid = _bundle_plane(fit_bundle, fit_root, parameter=parameter, stack="sax", group=group, method="fit_first")
                if not (reference.shape == proposed.shape == fit_first.shape == reference_shape):
                    raise ValueError("final-pose native bundle plane shape mismatch")
                parameter_report: dict[str, Any] = {}
                for roi_name, roi_mask in plane["rois"].items():
                    statistics = paired_roi_statistics(reference=reference, proposed=proposed, fit_first=fit_first,
                                                       proposed_valid=proposed_valid, fit_first_valid=fit_first_valid, roi=roi_mask,
                                                       parameter=parameter)
                    count = int(roi_mask.sum())
                    parameter_report[roi_name] = {
                        "roi_voxels": count, "common_support_voxels": statistics["common_support_voxels"],
                        "coverage": None if count == 0 else statistics["common_support_voxels"] / count,
                        "statistics": {key: value for key, value in statistics.items() if key != "common_support"},
                    }
                    support = statistics["common_support"]
                    for method, array in (("reference", reference), ("proposed", proposed), ("fit_first", fit_first)):
                        if support.any():
                            pooled[parameter][roi_name][method].append(array[support])
                    for method in ("proposed", "fit_first"):
                        value = statistics[method].get("ssim")
                        if value is not None:
                            pooled_ssim[parameter][roi_name][method].append(float(value))
                plane_report["parameters"][parameter] = parameter_report
            per_plane.append(plane_report)
        pooled_report = {parameter: {roi_name: _pooled_statistics(values, parameter=parameter,
                                                                  ssim_values=pooled_ssim[parameter][roi_name])
                                     for roi_name, values in by_roi.items()}
                         for parameter, by_roi in pooled.items()}
        write_immutable_json(staging / "per_plane_statistics.json", {
            "schema": SCHEMA, "subject": subject, "case": case, "primary_roi": PRIMARY_ROI, "support_equation": SUPPORT_EQUATION,
            "metric_contract": "RMSE, NRMSE, bias, MAE, Pearson CC, NCC, fixed-range PSNR, and mean masked 2-D SSIM",
            "psnr_data_ranges_ms": PSNR_DATA_RANGES_MS,
            "planes": per_plane,
        })
        write_immutable_json(staging / "pooled_sax_statistics.json", {
            "schema": SCHEMA, "subject": subject, "case": case, "primary_roi": PRIMARY_ROI, "support_equation": SUPPORT_EQUATION,
            "metric_contract": "RMSE, NRMSE, bias, MAE, Pearson CC, NCC, fixed-range PSNR, and mean masked 2-D SSIM",
            "psnr_data_ranges_ms": PSNR_DATA_RANGES_MS,
            "pooled_sax": pooled_report,
            "pooled_sax_primary": {parameter: pooled_report[parameter][PRIMARY_ROI] for parameter in pooled_report},
        })
        write_immutable_json(staging / "manifest.json", {
            "schema": SCHEMA, "subject": subject, "case": case,
            "roi_names": list(ROI_NAMES), "primary_roi": PRIMARY_ROI, "support_equation": SUPPORT_EQUATION,
            "metric_contract": "RMSE, NRMSE, bias, MAE, Pearson CC, NCC, fixed-range PSNR, and mean masked 2-D SSIM",
            "psnr_data_ranges_ms": PSNR_DATA_RANGES_MS,
            "input_provenance": input_provenance,
            "native_sampling": "only explicit mapped SAX planes; Proposed and Fit-first use final-pose native bundles",
            "roi_policy": {
                "myocardium_full": "outer AND NOT inner",
                "myocardium_core_legacy": "myocardium_full AND EDT(myocardium_full, spacing_rc_mm) > 1.0 mm",
                "myocardium_core_1px": "one in-plane 3x3 binary erosion of myocardium_full",
                "transform": "audited discrete roi_transform only; no interpolation",
            },
            "downstream_only": True, "no_reconstruction": True, "no_training": True,
            "no_formal_endpoints_modified": True, "immutable_output": True,
        })
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output
