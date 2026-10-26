"""Fail-closed CYJ1.1 unified-root adapter; it never accepts Final5 layout."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..phase2_clean.contracts import sha256, write_immutable_json


MAPPING_SCHEMA = "cyj11_aha_native_roi_mapping/v1"
ROI_BUNDLE_SCHEMA = "cyj11_native_roi_bundle/v1"
ROI_TRANSFORMS = ("identity", "transpose", "flipud", "fliplr", "transpose_flipud", "transpose_fliplr")
LEVELS = ("basal", "mid", "apical")


def cyj11_endpoint_paths(analysis_root: str | Path, subject: str, case: str) -> dict[str, Path]:
    root = Path(analysis_root)
    return {name: root / "endpoints" / "central" / name / subject / case
            for name in ("proposed", "fit_first")}


def _verified_record(record: Any, label: str, override: str | Path | None = None) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"{label} provenance record is required")
    path = Path(override) if override is not None else Path(str(record.get("path", "")))
    if not path.is_file() or record.get("sha256") != sha256(path):
        raise ValueError(f"{label} path/SHA256 mismatch")
    return path


def require_cyj11_mapping(path: str | Path, *, subject: str, case: str,
                          prepared_sax_path: str | Path | None = None,
                          roi_bundle_path: str | Path | None = None) -> dict[str, Any]:
    """Validate an explicit three-level mapping against immutable SAX observations."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != MAPPING_SCHEMA or payload.get("subject") != subject or payload.get("case") != case:
        raise ValueError("CYJ1.1 mapping schema or subject/case identity mismatch")
    prepared = _verified_record(payload.get("prepared_sax"), "prepared SAX", prepared_sax_path)
    with np.load(prepared, allow_pickle=False) as observations:
        groups = {int(value) for value in np.unique(observations["group_idx"])}
    bundle = _verified_record(payload.get("roi_bundle_manifest"), "ROI bundle", roi_bundle_path)
    bundle_payload = json.loads(bundle.read_text(encoding="utf-8"))
    if bundle_payload.get("schema") != ROI_BUNDLE_SCHEMA or bundle_payload.get("subject") != subject:
        raise ValueError("CYJ1.1 ROI bundle schema or subject identity mismatch")
    bundle_planes = {
        int(item["native_group_idx"]): int(item["roi_slice_index"])
        for item in bundle_payload.get("planes", [])
        if isinstance(item, dict) and isinstance(item.get("native_group_idx"), int)
        and isinstance(item.get("roi_slice_index"), int)
    }
    planes = payload.get("planes")
    if not isinstance(planes, list) or len(planes) != 3:
        raise ValueError("CYJ1.1 mapping requires exactly basal, mid, and apical planes")
    seen: set[str] = set()
    for plane in planes:
        if not isinstance(plane, dict) or plane.get("level") not in LEVELS or plane["level"] in seen:
            raise ValueError("CYJ1.1 mapping requires unique basal/mid/apical levels")
        seen.add(plane["level"])
        group = plane.get("native_group_idx")
        if not isinstance(group, int) or group not in groups:
            raise ValueError("CYJ1.1 mapping native SAX group is absent from prepared observations")
        if not isinstance(plane.get("roi_slice_index"), int) or plane["roi_slice_index"] < 0:
            raise ValueError("CYJ1.1 mapping requires a non-negative ROI slice index")
        if bundle_planes.get(group) != plane["roi_slice_index"]:
            raise ValueError("CYJ1.1 mapping ROI bundle plane identity disagrees with native group")
        if plane.get("roi_transform") not in ROI_TRANSFORMS:
            raise ValueError("CYJ1.1 mapping requires a declared discrete ROI transform")
        if plane["roi_transform"] != "identity":
            raise ValueError("CYJ1.1 derived ROI planes already occupy the native lattice; mapping transform must be identity")
        for key in ("anterior_xy", "inferior_xy"):
            point = np.asarray(plane.get(key), dtype=float)
            if point.shape != (2,) or not np.isfinite(point).all():
                raise ValueError(f"CYJ1.1 mapping requires finite {key}")
    return payload


def write_cyj11_mapping(*, template: str | Path, output: str | Path, subject: str, case: str) -> Path:
    """Materialize one reviewed explicit mapping template as a fresh immutable record."""
    template, output = Path(template), Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    payload = require_cyj11_mapping(template, subject=subject, case=case)
    write_immutable_json(output, payload)
    return output


def apply_roi_transform(array: np.ndarray, transform: str, expected_shape: tuple[int, int]) -> np.ndarray:
    """Apply only audited integer-grid orientation operations; never resample."""
    value = np.asarray(array)
    operations = {
        "identity": lambda x: x,
        "transpose": lambda x: x.T,
        "flipud": np.flipud,
        "fliplr": np.fliplr,
        "transpose_flipud": lambda x: np.flipud(x.T),
        "transpose_fliplr": lambda x: np.fliplr(x.T),
    }
    if transform not in operations:
        raise ValueError("CYJ1.1 ROI transform must be discrete")
    value = operations[transform](value)
    if value.shape != tuple(expected_shape):
        raise ValueError("CYJ1.1 ROI transform does not match native SAX plane shape")
    return value


def load_cyj11_native_sax(cache_root: str | Path, prepared_sax_path: str | Path) -> dict[str, np.ndarray]:
    """Load only an r3/native-cache SAX artifact bound to the selected prepared input."""
    root, prepared = Path(cache_root), Path(prepared_sax_path)
    maps_path, geometry_path, manifest_path = root / "native_maps.npz", root / "geometry.npz", root / "manifest.json"
    if not all(path.is_file() for path in (maps_path, geometry_path, manifest_path, prepared)):
        raise FileNotFoundError("CYJ1.1 native SAX cache is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest.get("source_observations", {})
    if source.get("sha256") != sha256(prepared):
        raise ValueError("CYJ1.1 native SAX cache is not bound to the selected prepared input")
    if manifest.get("native_maps_sha256") != sha256(maps_path) or manifest.get("geometry_sha256") != sha256(geometry_path):
        raise ValueError("CYJ1.1 native SAX cache manifest SHA256 mismatch")
    with np.load(prepared, allow_pickle=False) as observations, np.load(maps_path, allow_pickle=False) as maps, np.load(geometry_path, allow_pickle=False) as geometry:
        required_maps = ("group_idx", "t1_ms", "t2_ms", "b1")
        required_geometry = ("affine_lps_rc", "pixel_spacing_rc_mm", "slice_thickness_mm")
        if any(name not in maps for name in required_maps) or any(name not in geometry for name in required_geometry):
            raise ValueError("CYJ1.1 native SAX cache arrays are incomplete")
        result = {name: np.asarray(maps[name]) for name in required_maps}
        result.update({name: np.asarray(geometry[name]) for name in required_geometry})
        groups = np.unique(observations["group_idx"])
        expected_shape = (len(groups), *observations["images"].shape[-2:])
    if not np.array_equal(result["group_idx"], groups) or any(result[name].shape != expected_shape for name in ("t1_ms", "t2_ms", "b1")):
        raise ValueError("CYJ1.1 native SAX cache group identity or shape mismatches prepared input")
    if any(result[name].shape[0] != len(groups) for name in required_geometry):
        raise ValueError("CYJ1.1 native SAX cache geometry group count mismatch")
    return result
