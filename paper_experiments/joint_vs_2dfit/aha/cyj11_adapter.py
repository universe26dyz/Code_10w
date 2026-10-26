"""Fail-closed CYJ1.1 unified-root adapter; it never accepts Final5 layout."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..phase2_clean.contracts import sha256, write_immutable_json
from ..phase2_clean.fit_first import NATIVE_FIT_METHOD


MAPPING_SCHEMA = "cyj11_aha_native_roi_mapping/v1"
ROI_BUNDLE_SCHEMA = "cyj11_native_roi_bundle/v1"
ROI_TRANSFORMS = ("identity", "transpose", "flipud", "fliplr", "transpose_flipud", "transpose_fliplr")
LEVELS = ("basal", "mid", "apical")
HISTORICAL_AHA_LABELS = {"basal": (5, 6), "mid": (1, 2), "apical": (3, 4)}


def cyj11_endpoint_paths(analysis_root: str | Path, subject: str, case: str) -> dict[str, Path]:
    root = Path(analysis_root)
    return {name: root / "endpoints" / "central" / name / subject / case
            for name in ("proposed", "fit_first")}


def cyj11_native_stack_root(analysis_root: str | Path, subject: str, stack: str = "sax") -> Path:
    """Return the fixed unified-root R0 cache layout used by all imported subjects."""
    if stack not in ("sax", "2ch", "4ch"):
        raise ValueError(f"unknown CYJ1.1 native cache stack: {stack}")
    return Path(analysis_root) / "native_cache" / subject / "native_maps" / stack


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


def derive_cyj11_historical_mapping(*, subject: str, case: str, prepared_sax: str | Path,
                                    roi_bundle: str | Path, output: str | Path) -> Path:
    """Create the historical SCMR basal/mid/apical selection from labels 5/6, 1/2, and 3/4."""
    prepared_sax, roi_bundle, output = Path(prepared_sax), Path(roi_bundle), Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    derivation_path = roi_bundle / "roi_derivation_manifest.json"
    standard_path, masks_path = roi_bundle / "manifest.json", roi_bundle / "sax_myocardium_masks.npz"
    if not prepared_sax.is_file() or not derivation_path.is_file() or not standard_path.is_file() or not masks_path.is_file():
        raise FileNotFoundError("CYJ1.1 historical mapping requires prepared SAX plus immutable SCMR ROI bundle")
    derivation = json.loads(derivation_path.read_text(encoding="utf-8"))
    standard = json.loads(standard_path.read_text(encoding="utf-8"))
    if derivation.get("schema") != ROI_BUNDLE_SCHEMA or derivation.get("subject") != subject:
        raise ValueError("CYJ1.1 ROI derivation bundle schema or subject identity mismatch")
    if standard.get("schema") != "exact_native_myocardium_bundle/v2" or standard.get("status") != "PASS":
        raise ValueError("CYJ1.1 historical mapping requires a PASS SCMR myocardium bundle")
    slices = {int(item["native_group_idx"]): int(item["roi_slice_index"])
              for item in derivation.get("planes", []) if isinstance(item, dict)
              and isinstance(item.get("native_group_idx"), int) and isinstance(item.get("roi_slice_index"), int)}
    with np.load(prepared_sax, allow_pickle=False) as observations, np.load(masks_path, allow_pickle=False) as masks:
        groups = np.asarray(masks["group_idx"], int)
        points = np.asarray(masks["landmark_labels"])
        prepared_groups = {int(value) for value in np.unique(observations["group_idx"])}
    if points.ndim != 3 or points.shape[0] != len(groups) or set(groups.tolist()) != prepared_groups:
        raise ValueError("CYJ1.1 landmark bundle does not match prepared SAX groups")
    planes: list[dict[str, Any]] = []
    for level in LEVELS:
        anterior_label, inferior_label = HISTORICAL_AHA_LABELS[level]
        candidates = np.flatnonzero(np.any(points == anterior_label, axis=(1, 2))
                                   & np.any(points == inferior_label, axis=(1, 2)))
        if len(candidates) != 1:
            raise ValueError(f"CYJ1.1 historical {level} AHA labels {anterior_label}/{inferior_label} require exactly one native SAX plane")
        index = int(candidates[0]); group = int(groups[index])
        if group not in slices:
            raise ValueError("CYJ1.1 historical AHA landmark group lacks an immutable ROI derivation plane")
        anterior_rc, inferior_rc = np.where(points[index] == anterior_label), np.where(points[index] == inferior_label)
        planes.append({"level": level, "native_group_idx": group, "roi_slice_index": slices[group], "roi_transform": "identity",
                       "anterior_xy": [float(anterior_rc[1].mean()), float(anterior_rc[0].mean())],
                       "inferior_xy": [float(inferior_rc[1].mean()), float(inferior_rc[0].mean())]})
    payload = {"schema": MAPPING_SCHEMA, "subject": subject, "case": case,
               "prepared_sax": {"path": str(prepared_sax.resolve()), "sha256": sha256(prepared_sax)},
               "roi_bundle_manifest": {"path": str(derivation_path.resolve()), "sha256": sha256(derivation_path)},
               "landmark_selection": {"source": str(masks_path.resolve()), "sha256": sha256(masks_path),
                                      "historical_labels": {key: list(value) for key, value in HISTORICAL_AHA_LABELS.items()}},
               "planes": planes}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)) / output.name
    try:
        write_immutable_json(staging, payload)
        require_cyj11_mapping(staging, subject=subject, case=case)
        staging.rename(output)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(staging.parent, ignore_errors=True)
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
    maps_hash, geometry_hash = manifest.get("native_maps_sha256"), manifest.get("geometry_sha256")
    if maps_hash is not None or geometry_hash is not None:
        if (not isinstance(maps_hash, str) or not isinstance(geometry_hash, str)
                or maps_hash != sha256(maps_path) or geometry_hash != sha256(geometry_path)):
            raise ValueError("CYJ1.1 native SAX cache manifest SHA256 mismatch")
    elif not (
        manifest.get("method") == NATIVE_FIT_METHOD
        and isinstance(manifest.get("dictionary_cache_entries"), int)
        and not isinstance(manifest.get("dictionary_cache_entries"), bool)
        and manifest["dictionary_cache_entries"] > 0
        and isinstance(manifest.get("baseline_pose_policy"), str)
        and isinstance(manifest.get("next_formal_step"), str)
    ):
        raise ValueError("CYJ1.1 native SAX cache lacks sealed SHA256 provenance")
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
