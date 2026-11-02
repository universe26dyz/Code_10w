"""CYJ1.1 unified-root endpoint sampler for downstream AHA/ROI runners."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..phase2_clean.evaluation import reproject_binary_mask_to_native_plane, reproject_volume_to_native_plane
from ..phase2_clean.final_native_grid_metrics import native_ras_affine
from .cyj11_adapter import cyj11_endpoint_paths, cyj11_native_stack_root, load_cyj11_native_sax
from ..phase2_clean.contracts import sha256


FINAL_POSE_DIRECT_SCHEMAS = (
    "direct_native_prediction_bundle/v1",
    "direct_sparse_native_prediction_bundle/v1",
    "cyj11_r3_direct_final_pose_native_bundle/v1",
)
FINAL_POSE_FIT_SCHEMAS = (
    "fit_first_native_prediction_bundle/v6",
    "cyj11_r3_fit_first_final_pose_native_bundle/v1",
    "fit_first_raw_final_pose_native_bundle/v1",
)
RAW_FIT_FIRST_FINAL_POSE_SCHEMA = "fit_first_raw_final_pose_native_bundle/v1"


def load_cyj11_endpoint_images(analysis_root: str | Path, subject: str, case: str) -> dict[str, dict[str, Any]]:
    """Load exactly the central endpoint files emitted by r3; no path discovery."""
    import nibabel as nib
    roots = cyj11_endpoint_paths(analysis_root, subject, case)
    paths = {
        "t1": {"proposed": roots["proposed"] / "T1_3D.nii.gz", "fit_first": roots["fit_first"] / "t1" / "volume_ms.nii.gz", "fit_mask": roots["fit_first"] / "t1" / "query_effective_mask.nii.gz"},
        "t2": {"proposed": roots["proposed"] / "T2_3D.nii.gz", "fit_first": roots["fit_first"] / "t2" / "volume_ms.nii.gz", "fit_mask": roots["fit_first"] / "t2" / "query_effective_mask.nii.gz"},
    }
    for root in roots.values():
        manifest = root / "manifest.json"
        if not manifest.is_file(): raise FileNotFoundError(manifest)
        payload = json.loads(manifest.read_text())
        if payload.get("subject") != subject or payload.get("condition") != case or payload.get("endpoint") != "central":
            raise ValueError("CYJ1.1 central endpoint identity mismatch")
    if any(not path.is_file() for item in paths.values() for path in item.values()):
        raise FileNotFoundError("CYJ1.1 central endpoint volume or effective-query mask is missing")
    return {parameter: {name: nib.load(str(path)) for name, path in item.items()} for parameter, item in paths.items()}


def native_sax_plane(*, native: dict[str, np.ndarray], group: int, endpoint_images: dict[str, Any], parameter: str) -> dict[str, np.ndarray]:
    """Sample both methods and Fit-first's binary query mask on one cache plane."""
    groups = [int(value) for value in native["group_idx"]]
    if group not in groups: raise ValueError(f"native SAX group {group} is absent")
    index = groups.index(group); reference = np.asarray(native[f"{parameter}_ms"][index], float)
    affine = native_ras_affine(np.asarray(native["affine_lps_rc"][index], float))
    proposed, proposed_valid = reproject_volume_to_native_plane(endpoint_images[parameter]["proposed"].get_fdata(), endpoint_images[parameter]["proposed"].affine, affine, reference.shape)
    fit_first, fit_valid = reproject_volume_to_native_plane(endpoint_images[parameter]["fit_first"].get_fdata(), endpoint_images[parameter]["fit_first"].affine, affine, reference.shape)
    mask = reproject_binary_mask_to_native_plane(endpoint_images[parameter]["fit_mask"].get_fdata(), endpoint_images[parameter]["fit_mask"].affine, affine, reference.shape)
    return {"reference": reference, "proposed": proposed, "fit_first": fit_first, "proposed_valid": proposed_valid, "fit_first_valid": fit_valid & mask}


def load_cyj11_final_pose_bundles(proposed_bundle_root: str | Path, fit_first_bundle_root: str | Path,
                                  *, subject: str, case: str) -> dict[str, Any]:
    """Admit current-experiment final-pose native bundles for downstream SAX use.

    The bundles are produced from each current reconstruction's own final pose:
    the Proposed checkpoint's learned rigid pose and Fit-first's parameter-specific
    NeSVoR output-slice poses.  This function deliberately does not accept an
    endpoint-volume substitute or any historical SCMR pose artifact.
    """
    direct_root, fit_root = Path(proposed_bundle_root), Path(fit_first_bundle_root)
    direct_path, fit_path = direct_root / "manifest.json", fit_root / "manifest.json"
    if not direct_path.is_file() or not fit_path.is_file():
        raise FileNotFoundError("FINAL_POSE_AHA_PENDING: both current final-pose bundle manifests are required")
    direct, fit = json.loads(direct_path.read_text(encoding="utf-8")), json.loads(fit_path.read_text(encoding="utf-8"))
    if direct.get("schema") not in FINAL_POSE_DIRECT_SCHEMAS or direct.get("endpoint") != "central":
        raise ValueError("current Proposed final-pose native bundle schema/endpoint mismatch")
    if direct.get("schema") in {"direct_sparse_native_prediction_bundle/v1", "cyj11_r3_direct_final_pose_native_bundle/v1"} and (direct.get("subject"), direct.get("case_id")) != (subject, case):
        raise ValueError("current Proposed final-pose native bundle subject/case mismatch")
    if fit.get("schema") not in FINAL_POSE_FIT_SCHEMAS or fit.get("endpoint") != "central":
        raise ValueError("current Fit-first final-pose native bundle schema/endpoint mismatch")
    if fit.get("schema") == RAW_FIT_FIRST_FINAL_POSE_SCHEMA:
        if (fit.get("subject"), fit.get("case"), fit.get("support_policy")) != (
            subject, case, "volume_geometry_and_finite_prediction_only",
        ):
            raise ValueError("current raw Fit-first final-pose bundle subject/case/support mismatch")
    else:
        endpoint_record = fit.get("endpoint_source_manifest")
        endpoint_path = Path(str(endpoint_record.get("path", ""))) if isinstance(endpoint_record, dict) else None
        if endpoint_path is None or not endpoint_path.is_file() or endpoint_record.get("sha256") != sha256(endpoint_path):
            raise ValueError("current Fit-first final-pose bundle endpoint provenance mismatch")
        endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
        if (endpoint.get("subject"), endpoint.get("condition"), endpoint.get("endpoint")) != (subject, case, "central"):
            raise ValueError("current Fit-first final-pose endpoint subject/case mismatch")
    if not isinstance(direct.get("planes"), list):
        raise ValueError("current Proposed final-pose native bundle lacks plane records")
    return {"direct_root": direct_root, "direct_manifest": direct, "fit_root": fit_root, "fit_manifest": fit}


def final_pose_sax_plane(*, native: dict[str, np.ndarray], group: int, parameter: str,
                         direct_root: Path, direct_manifest: dict[str, Any],
                         fit_root: Path, fit_manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    """Load one current final-pose SAX plane on the original pixel ordering."""
    if parameter not in ("t1", "t2"):
        raise ValueError("unknown quantitative parameter")
    groups = [int(value) for value in native["group_idx"]]
    if group not in groups:
        raise ValueError(f"native SAX group {group} is absent")
    record = next((item for item in direct_manifest["planes"]
                   if isinstance(item, dict) and item.get("stack") == "sax"
                   and item.get("original_group_idx") == group), None)
    if record is None or not isinstance(record.get("path"), str):
        raise ValueError(f"FINAL_POSE_AHA_PENDING: Proposed final-pose SAX group {group} is unavailable")
    direct_path = direct_root / record["path"]
    fit_path = fit_root / "planes" / parameter / "sax" / f"group{group:03d}.npz"
    if not direct_path.is_file() or not fit_path.is_file():
        raise ValueError(f"FINAL_POSE_AHA_PENDING: paired final-pose SAX group {group}/{parameter} is unavailable")
    index = groups.index(group)
    reference = np.asarray(native[f"{parameter}_ms"][index], float)
    raw_fit = fit_manifest.get("schema") == RAW_FIT_FIRST_FINAL_POSE_SCHEMA
    with np.load(direct_path, allow_pickle=False) as direct, np.load(fit_path, allow_pickle=False) as fit:
        required = {"prediction_ms", "volume_geometry_valid", "valid"} if raw_fit else {"prediction_ms", "volume_geometry_valid", "query_effective_mask"}
        if f"{parameter}_ms" not in direct or not required.issubset(fit.files):
            raise ValueError("current final-pose native bundle plane arrays are incomplete")
        proposed = np.asarray(direct[f"{parameter}_ms"], float)
        fit_first = np.asarray(fit["prediction_ms"], float)
        if raw_fit:
            derived = np.asarray(fit["volume_geometry_valid"], bool) & np.isfinite(fit_first)
            fit_valid = np.asarray(fit["valid"], bool)
            if not np.array_equal(fit_valid, derived):
                raise ValueError("raw Fit-first final-pose valid mask violates bundle contract")
        else:
            fit_valid = (np.asarray(fit["volume_geometry_valid"], bool)
                         & np.asarray(fit["query_effective_mask"], bool) & np.isfinite(fit_first))
    if not (reference.shape == proposed.shape == fit_first.shape == fit_valid.shape):
        raise ValueError("current final-pose SAX bundle/native geometry mismatch")
    return {"reference": reference, "proposed": proposed, "fit_first": fit_first,
            "proposed_valid": np.isfinite(proposed), "fit_first_valid": fit_valid}


def load_cyj11_run_inputs(analysis_root: str | Path, subject: str, case: str) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    native = load_cyj11_native_inputs(analysis_root, subject)
    return native, load_cyj11_endpoint_images(analysis_root, subject, case)


def load_cyj11_native_inputs(analysis_root: str | Path, subject: str) -> dict[str, np.ndarray]:
    root = Path(analysis_root)
    return load_cyj11_native_sax(cyj11_native_stack_root(root, subject), root / "prepared" / subject / "sax" / "observations.npz")
