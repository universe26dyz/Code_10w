"""Explicit, fail-closed data adapter for Final5 AHA16 downstream output."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..phase2_clean.contracts import sha256
from ..phase2_clean.evaluation import reproject_volume_to_native_plane
from ..phase2_clean.final_native_grid_metrics import native_ras_affine
from ..phase2_clean.final_sparse_native_metrics import validate_sparse_endpoint_roots


ROI_FILENAMES = (
    "Segmentation_1-out-label.nii.gz", "Segmentation_2-in-label.nii.gz",
    "Segmentation_3-label_point.nii.gz", "sax_composite_reference.nii.gz",
)
MAPPING_SCHEMA = "final5_aha_native_roi_mapping/v1"
FINAL5_SUBJECTS = ("CYJ", "DYL_2", "DYZ", "HHZ", "HJL")
FINAL5_CROPPED_ROI_VERSION = "20261025_final5_cropped_roi_v1"
FINAL5_DIRECT_ROI_VERSION = "20261025_final5_aha_v1"
DERIVED_ROI_SCHEMA = "final5_cropped_roi_bundle/v1"
DIRECT_ROI_SCHEMA = "dyl_2_final5_aha_manual_roi/v1"
ROI_TRANSFORMS = ("identity", "transpose", "flipud", "fliplr", "transpose_flipud", "transpose_fliplr")


def final5_endpoint_paths(final_root: str | Path, subject: str, case: str) -> dict[str, Path]:
    """Return the only accepted deterministic Final5 endpoint paths."""
    root = Path(final_root)
    return {
        "proposed": root / "05_formal_endpoints" / "central" / "proposed" / subject / case,
        "fit_first": root / "05_formal_endpoints" / "central" / "fit_first" / subject / case,
    }


def _is_final5_roi_bundle(folder: Path, subject: str) -> bool:
    if not all((folder / name).is_file() for name in ROI_FILENAMES):
        return False
    manifest = folder / "roi_derivation_manifest.json"
    if manifest.is_file():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        return payload.get("schema") == DERIVED_ROI_SCHEMA and payload.get("subject") == subject and payload.get("resampling") is False
    if subject == "DYL_2":
        direct = folder / "final_aha_roi_provenance.json"
        if not direct.is_file():
            return False
        try:
            payload = json.loads(direct.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        return payload.get("schema") == DIRECT_ROI_SCHEMA
    return False


def resolve_subject_roi(roi_root: str | Path, subject: str) -> Path | None:
    """Resolve a Final5-ready ROI by declared layout; legacy full-FOV ROI is never a fallback."""
    supplied = Path(roi_root)
    candidates = [supplied]
    if subject == "DYL_2":
        candidates.extend((
            supplied / subject / "data" / "roi" / FINAL5_DIRECT_ROI_VERSION,
            supplied / subject / "roi",
        ))
    else:
        candidates.extend((
            supplied / subject / "data" / "roi" / FINAL5_CROPPED_ROI_VERSION,
            supplied / subject / "roi",
        ))
    for folder in candidates:
        if _is_final5_roi_bundle(folder, subject):
            return folder
    return None


def assess_final5_aha_readiness(final_root: str | Path, subject: str, case: str, roi_root: str | Path, *,
                                mapping_path: str | Path | None = None) -> dict[str, Any]:
    """Report a strict readiness state; never substitutes data or discovers paths."""
    endpoints = final5_endpoint_paths(final_root, subject, case)
    roi = resolve_subject_roi(roi_root, subject)
    report: dict[str, Any] = {
        "subject": subject, "case": case, "proposed_endpoint": str(endpoints["proposed"]),
        "fit_first_endpoint": str(endpoints["fit_first"]), "roi_root": str(roi) if roi else None,
    }
    if roi is None:
        return {**report, "status": "ROI_PENDING", "reason": "subject-specific audited manual ROI is unavailable; DYL ROI is never reused for DYL_2"}
    if not endpoints["proposed"].is_dir() or not endpoints["fit_first"].is_dir():
        return {**report, "status": "ENDPOINTS_PENDING", "reason": "required Final5 central endpoint directories are absent"}
    try:
        validate_sparse_endpoint_roots(endpoints["proposed"], endpoints["fit_first"], case, subject=subject)
    except (FileNotFoundError, ValueError) as exc:
        return {**report, "status": "ENDPOINTS_INVALID", "reason": str(exc)}
    if mapping_path is None:
        return {**report, "status": "NATIVE_MAPPING_PENDING", "reason": "explicit standardization/physical native-to-ROI mapping is required"}
    try:
        mapping = require_exact_native_roi_mapping(mapping_path, subject=subject, case=case, roi_root=roi)
    except (FileNotFoundError, ValueError) as exc:
        return {**report, "status": "NATIVE_MAPPING_INVALID", "reason": str(exc)}
    return {**report, "status": "READY", "mapping": mapping}


def assess_final5_aha_cohort(final_root: str | Path, case: str, roi_root: str | Path) -> list[dict[str, Any]]:
    """Return deterministic readiness records for the frozen Final5 cohort."""
    return [assess_final5_aha_readiness(final_root, subject, case, roi_root) for subject in FINAL5_SUBJECTS]


def _require_mapping_derivation(payload: dict[str, Any], roi_root: Path, subject: str) -> None:
    """Bind a legacy-subject Final5 mapping to the immutable derived ROI manifest."""
    if subject == "DYL_2":
        return
    manifest = roi_root / "roi_derivation_manifest.json"
    if not manifest.is_file():
        raise ValueError("Final5 legacy subject ROI lacks derivation manifest")
    derived = json.loads(manifest.read_text(encoding="utf-8"))
    if derived.get("schema") != DERIVED_ROI_SCHEMA or derived.get("subject") != subject or derived.get("resampling") is not False:
        raise ValueError("Final5 legacy subject ROI derivation manifest is invalid")
    declared = payload.get("roi_derivation_manifest")
    if not isinstance(declared, dict) or declared.get("path") != str(manifest.resolve()) or declared.get("sha256") != sha256(manifest):
        raise ValueError("AHA mapping derivation manifest does not match the Final5 ROI bundle")


def require_exact_native_roi_mapping(path: str | Path, *, subject: str, case: str,
                                     roi_root: str | Path | None = None) -> dict[str, Any]:
    """Validate a declared physical/standardization mapping without guessing IDs."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != MAPPING_SCHEMA or payload.get("subject") != subject or payload.get("case") != case:
        raise ValueError("AHA mapping schema or subject/case identity mismatch")
    standard = payload.get("standardization_manifest")
    if not isinstance(standard, dict):
        raise ValueError("AHA mapping requires standardization manifest provenance")
    standard_path = Path(str(standard.get("path", "")))
    if not standard_path.is_file() or standard.get("sha256") != sha256(standard_path):
        raise ValueError("AHA mapping standardization manifest SHA256 mismatch")
    standard_payload = json.loads(standard_path.read_text(encoding="utf-8"))
    sax = standard_payload.get("stacks", {}).get("sax") if isinstance(standard_payload.get("stacks"), dict) else None
    declared = sax.get("standardized_to_original_group_idx") if isinstance(sax, dict) else None
    if standard_payload.get("subject") != subject or not isinstance(declared, list):
        raise ValueError("AHA mapping standardization manifest lacks subject-matched SAX physical group mapping")
    physical_groups: dict[int, int] = {}
    for item in declared:
        if not isinstance(item, dict) or not isinstance(item.get("derived_group_idx"), int) or not isinstance(item.get("original_group_idx"), int):
            raise ValueError("AHA mapping standardization record is malformed")
        physical_groups[item["derived_group_idx"]] = item["original_group_idx"]
    planes = payload.get("planes")
    if not isinstance(planes, list) or not planes:
        raise ValueError("AHA mapping requires explicit native SAX plane records")
    seen: set[tuple[str, int]] = set()
    for plane in planes:
        if not isinstance(plane, dict) or plane.get("level") not in ("basal", "mid", "apical"):
            raise ValueError("AHA mapping has an invalid level record")
        key = (str(plane.get("level")), plane.get("native_group_idx"))
        if not isinstance(key[1], int) or key in seen:
            raise ValueError("AHA mapping requires unique integer native group identities")
        seen.add(key)
        for name in ("original_group_idx", "roi_slice_index"):
            if not isinstance(plane.get(name), int):
                raise ValueError(f"AHA mapping requires integer {name}")
        if physical_groups.get(plane["native_group_idx"]) != plane["original_group_idx"]:
            raise ValueError("AHA mapping native/original group identity disagrees with the standardization manifest")
        if plane.get("roi_transform") not in ROI_TRANSFORMS:
            raise ValueError("AHA mapping requires one explicit supported roi_transform")
        for name in ("anterior_xy", "inferior_xy"):
            point = np.asarray(plane.get(name), float)
            if point.shape != (2,) or not np.isfinite(point).all():
                raise ValueError(f"AHA mapping requires finite {name}")
    if roi_root is not None:
        _require_mapping_derivation(payload, Path(roi_root), subject)
    return payload


def reproject_current_endpoints_to_native_sax(proposed_image: Any, fit_first_image: Any, native_affine_lps_rc: np.ndarray,
                                              plane_shape: tuple[int, int]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Reproject both immutable central endpoints onto exactly one native lattice."""
    affine = native_ras_affine(native_affine_lps_rc)
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, image in (("proposed", proposed_image), ("fit_first", fit_first_image)):
        result[name] = reproject_volume_to_native_plane(image.get_fdata(), image.affine, affine, plane_shape)
    return result
