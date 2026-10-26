"""CYJ1.1 myocardial ROI statistics using the established pure ROI algorithms."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..phase2_clean.contracts import sha256, write_immutable_json
from .cyj11_adapter import require_cyj11_mapping
from .cyj11_legacy_roi import load_cyj11_roi_plane
from .cyj11_runner import load_cyj11_run_inputs, native_sax_plane

from .final5_myocardial_roi_metrics import build_myocardial_rois, paired_roi_statistics

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


def build_cyj11_myocardial_roi_metrics(*, analysis_root: str | Path, subject: str, case: str,
                                        mapping_path: str | Path, roi_bundle: str | Path, output: str | Path) -> Path:
    """Write immutable per-plane and pooled three-ROI statistics on paired support."""
    output, mapping_path, roi_bundle = Path(output), Path(mapping_path), Path(roi_bundle)
    if output.exists(): raise FileExistsError(f"refusing overwrite: {output}")
    root = Path(analysis_root)
    mapping = require_cyj11_mapping(mapping_path, subject=subject, case=case,
                                    prepared_sax_path=root / "prepared" / subject / "sax" / "observations.npz",
                                    roi_bundle_path=roi_bundle / "roi_derivation_manifest.json")
    native, endpoint_images = load_cyj11_run_inputs(root, subject, case)
    indices = {int(group): index for index, group in enumerate(native["group_idx"])}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        reports: list[dict[str, Any]] = []
        pooled = {parameter: {name: {method: [] for method in ("reference", "proposed", "fit_first")}
                              for name in ROI_NAMES} for parameter in ("t1", "t2")}
        for entry in mapping["planes"]:
            group = int(entry["native_group_idx"])
            if group not in indices: raise ValueError(f"CYJ1.1 native SAX group {group} declared in mapping is absent")
            index = indices[group]; shape = tuple(map(int, native["t1_ms"][index].shape))
            masks = load_cyj11_roi_plane(roi_bundle, group=group, expected_shape=shape)
            rois = build_myocardial_rois(masks["outer"], masks["inner"],
                                         spacing_rc_mm=tuple(map(float, native["pixel_spacing_rc_mm"][index][:2])))
            report: dict[str, Any] = {key: entry[key] for key in ("level", "native_group_idx", "roi_slice_index", "roi_transform")}
            report["roi_voxels"] = {name: int(mask.sum()) for name, mask in rois.items()}; report["parameters"] = {}
            for parameter in ("t1", "t2"):
                plane = native_sax_plane(native=native, group=group, endpoint_images=endpoint_images, parameter=parameter)
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
        write_immutable_json(staging / "per_plane_statistics.json", {"schema": "cyj11_local_myocardial_roi_metrics/v1", "subject": subject, "case": case, "support_equation": SUPPORT_EQUATION, "planes": reports})
        write_immutable_json(staging / "pooled_sax_statistics.json", {"schema": "cyj11_local_myocardial_roi_metrics/v1", "subject": subject, "case": case, "support_equation": SUPPORT_EQUATION, "pooled_sax": pooled_report})
        write_immutable_json(staging / "manifest.json", {
            "schema": "cyj11_local_myocardial_roi_metrics/v1", "subject": subject, "case": case,
            "mapping": _record(mapping_path), "roi_bundle": _record(roi_bundle / "roi_derivation_manifest.json"),
            "prepared_sax": _record(root / "prepared" / subject / "sax" / "observations.npz"),
            "native_sax": {name: _record(root / "native_cache" / subject / "sax" / name) for name in ("native_maps.npz", "geometry.npz", "manifest.json")},
            "endpoint_files": {parameter: {method: _record(image.get_filename()) for method, image in images.items()} for parameter, images in endpoint_images.items()},
            "roi_names": list(ROI_NAMES), "support_equation": SUPPORT_EQUATION,
            "roi_policy": {"myocardium_full": "outer AND NOT inner", "myocardium_core_legacy": "myocardium_full AND EDT(...) > 1.0 mm", "myocardium_core_1px": "one in-plane 3x3 erosion", "transform": "derived native lattice; no interpolation"},
            "downstream_only": True, "no_reconstruction": True, "no_formal_metric": True, "immutable_output": True,
        })
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True); raise
    return output


__all__ = ("SUPPORT_EQUATION", "ROI_NAMES", "build_myocardial_rois", "paired_roi_statistics", "build_cyj11_myocardial_roi_metrics")
