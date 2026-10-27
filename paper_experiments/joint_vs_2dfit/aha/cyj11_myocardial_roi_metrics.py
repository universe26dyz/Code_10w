"""CYJ1.1 myocardial ROI statistics using the established pure ROI algorithms."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..phase2_clean.contracts import sha256, write_immutable_json
from .cyj11_adapter import cyj11_native_stack_root, require_cyj11_mapping
from .cyj11_runner import (final_pose_sax_plane, load_cyj11_final_pose_bundles,
                           load_cyj11_native_inputs, load_cyj11_run_inputs,
                           native_sax_plane)

from .final5_myocardial_roi_metrics import paired_roi_statistics

SUPPORT_EQUATION = "manual anatomical ROI AND finite(reference) AND proposed-valid AND fit-first-valid AND finite(proposed, fit-first)"

ROI_NAMES = ("myocardium_full", "myocardium_core_legacy", "myocardium_core_1px")


def _record(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.is_file(): raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _pooled(values: dict[str, list[np.ndarray]]) -> dict[str, Any]:
    reference = np.concatenate(values["reference"]) if values["reference"] else np.asarray([], float)
    proposed = np.concatenate(values["proposed"]) if values["proposed"] else np.asarray([], float)
    fit_first = np.concatenate(values["fit_first"]) if values["fit_first"] else np.asarray([], float)
    result = paired_roi_statistics(reference=reference, proposed=proposed, fit_first=fit_first,
                                   proposed_valid=np.ones(reference.shape, bool), fit_first_valid=np.ones(reference.shape, bool),
                                   roi=np.ones(reference.shape, bool))
    return {key: value for key, value in result.items() if key != "common_support"}


def _load_scmr_masks(root: Path, *, expected_groups: np.ndarray, expected_shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Load the immutable all-SAX mask bundle used by SCMR ROI measurement."""
    manifest_path, archive_path = root / "manifest.json", root / "sax_myocardium_masks.npz"
    if not manifest_path.is_file() or not archive_path.is_file():
        raise FileNotFoundError("CYJ1.1 myocardial metrics require SCMR manifest.json and sax_myocardium_masks.npz")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "exact_native_myocardium_bundle/v2" or manifest.get("status") != "PASS":
        raise ValueError("CYJ1.1 myocardial metrics require a PASS exact_native_myocardium_bundle/v2 bundle")
    with np.load(archive_path, allow_pickle=False) as archive:
        if any(name not in archive for name in ("group_idx", *ROI_NAMES)):
            raise ValueError("CYJ1.1 SCMR myocardium bundle arrays are incomplete")
        groups = np.asarray(archive["group_idx"], int)
        masks = {name: np.asarray(archive[name], bool) for name in ROI_NAMES}
    if not np.array_equal(groups, expected_groups):
        raise ValueError("CYJ1.1 SCMR myocardium bundle group identity mismatches native SAX")
    expected = (len(expected_groups), *expected_shape)
    if any(mask.shape != expected for mask in masks.values()):
        raise ValueError("CYJ1.1 SCMR myocardium bundle shape mismatches native SAX")
    return masks


def build_cyj11_myocardial_roi_metrics(*, analysis_root: str | Path, subject: str, case: str,
                                        mapping_path: str | Path, roi_bundle: str | Path, output: str | Path,
                                        sampling: str = "native_endpoint",
                                        proposed_native_bundle: str | Path | None = None,
                                        fit_first_native_bundle: str | Path | None = None) -> Path:
    """Write SCMR-style all-SAX per-plane and pooled paired ROI statistics."""
    output, mapping_path, roi_bundle = Path(output), Path(mapping_path), Path(roi_bundle)
    if output.exists(): raise FileExistsError(f"refusing overwrite: {output}")
    root = Path(analysis_root)
    mapping = require_cyj11_mapping(mapping_path, subject=subject, case=case,
                                    prepared_sax_path=root / "prepared" / subject / "sax" / "observations.npz",
                                    roi_bundle_path=roi_bundle / "roi_derivation_manifest.json")
    if sampling not in ("native_endpoint", "final_pose"):
        raise ValueError("CYJ1.1 myocardial sampling must be native_endpoint or final_pose")
    if sampling == "final_pose":
        if proposed_native_bundle is None or fit_first_native_bundle is None:
            raise ValueError("FINAL_POSE_AHA_PENDING: both current final-pose bundle roots are required")
        native = load_cyj11_native_inputs(root, subject)
        final_pose = load_cyj11_final_pose_bundles(proposed_native_bundle, fit_first_native_bundle,
                                                    subject=subject, case=case)
        endpoint_images = None
    else:
        native, endpoint_images = load_cyj11_run_inputs(root, subject, case)
        final_pose = None
    indices = {int(group): index for index, group in enumerate(native["group_idx"])}
    masks = _load_scmr_masks(roi_bundle, expected_groups=np.asarray(native["group_idx"], int),
                             expected_shape=tuple(map(int, native["t1_ms"].shape[1:])))
    aha_levels = {int(entry["native_group_idx"]): entry["level"] for entry in mapping["planes"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        reports: list[dict[str, Any]] = []
        pooled = {parameter: {name: {method: [] for method in ("reference", "proposed", "fit_first")}
                              for name in ROI_NAMES} for parameter in ("t1", "t2")}
        for group in map(int, native["group_idx"]):
            index = indices[group]; shape = tuple(map(int, native["t1_ms"][index].shape))
            if shape != tuple(map(int, masks["myocardium_full"][index].shape)):
                raise ValueError("CYJ1.1 SCMR myocardium mask/native plane shape mismatch")
            rois = {name: masks[name][index] for name in ROI_NAMES}
            report: dict[str, Any] = {"native_group_idx": group, "aha_level": aha_levels.get(group)}
            report["roi_voxels"] = {name: int(mask.sum()) for name, mask in rois.items()}; report["parameters"] = {}
            for parameter in ("t1", "t2"):
                plane = (final_pose_sax_plane(native=native, group=group, parameter=parameter, **final_pose)
                         if final_pose is not None else
                         native_sax_plane(native=native, group=group, endpoint_images=endpoint_images, parameter=parameter))
                parameter_report: dict[str, Any] = {}
                for name, roi in rois.items():
                    stats = paired_roi_statistics(roi=roi, **plane)
                    parameter_report[name] = {"roi_voxels": int(roi.sum()), "common_support_voxels": stats["common_support_voxels"],
                                              "coverage": None if not roi.any() else stats["common_support_voxels"] / int(roi.sum()),
                                              "statistics": {key: value for key, value in stats.items() if key != "common_support"}}
                    support = stats["common_support"]
                    for method in ("reference", "proposed", "fit_first"):
                        if support.any(): pooled[parameter][name][method].append(plane[method][support])
                report["parameters"][parameter] = parameter_report
            reports.append(report)
        pooled_report = {parameter: {name: _pooled(values) for name, values in items.items()}
                         for parameter, items in pooled.items()}
        write_immutable_json(staging / "per_plane_statistics.json", {"schema": "cyj11_local_myocardial_roi_metrics/v2", "subject": subject, "case": case, "sampling": sampling, "support_equation": SUPPORT_EQUATION, "planes": reports})
        write_immutable_json(staging / "pooled_sax_statistics.json", {"schema": "cyj11_local_myocardial_roi_metrics/v2", "subject": subject, "case": case, "sampling": sampling, "support_equation": SUPPORT_EQUATION, "pooled_sax": pooled_report})
        write_immutable_json(staging / "manifest.json", {
            "schema": "cyj11_local_myocardial_roi_metrics/v2", "subject": subject, "case": case,
            "mapping": _record(mapping_path), "roi_bundle": _record(roi_bundle / "roi_derivation_manifest.json"),
            "scmr_myocardium_bundle": {"manifest": _record(roi_bundle / "manifest.json"),
                                         "archive": _record(roi_bundle / "sax_myocardium_masks.npz")},
            "prepared_sax": _record(root / "prepared" / subject / "sax" / "observations.npz"),
            "native_sax": {name: _record(cyj11_native_stack_root(root, subject) / name) for name in ("native_maps.npz", "geometry.npz", "manifest.json")},
            "sampling": sampling,
            "endpoint_files": None if endpoint_images is None else {parameter: {method: _record(image.get_filename()) for method, image in images.items()} for parameter, images in endpoint_images.items()},
            "current_final_pose_bundles": None if final_pose is None else {
                "proposed_manifest": _record(final_pose["direct_root"] / "manifest.json"),
                "fit_first_manifest": _record(final_pose["fit_root"] / "manifest.json"),
                "policy": "current reconstruction final learned pose (Proposed) and parameter-specific final NeSVoR output-slice poses (Fit-first)",
            },
            "roi_names": list(ROI_NAMES), "support_equation": SUPPORT_EQUATION,
            "roi_policy": {"myocardium_full": "SCMR bundle outer AND NOT inner", "myocardium_core_legacy": "SCMR bundle EDT(...) > 1.0 mm", "myocardium_core_1px": "SCMR bundle one in-plane 3x3 erosion", "transform": "derived native lattice; no interpolation"},
            "downstream_only": True, "no_reconstruction": True, "no_formal_metric": True, "immutable_output": True,
        })
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True); raise
    return output


__all__ = ("SUPPORT_EQUATION", "ROI_NAMES", "build_myocardial_rois", "paired_roi_statistics", "build_cyj11_myocardial_roi_metrics")
