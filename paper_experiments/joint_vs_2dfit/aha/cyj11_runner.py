"""CYJ1.1 unified-root endpoint sampler for downstream AHA/ROI runners."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..phase2_clean.evaluation import reproject_binary_mask_to_native_plane, reproject_volume_to_native_plane
from ..phase2_clean.final_native_grid_metrics import native_ras_affine
from .cyj11_adapter import cyj11_endpoint_paths, cyj11_native_stack_root, load_cyj11_native_sax


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


def load_cyj11_run_inputs(analysis_root: str | Path, subject: str, case: str) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    root = Path(analysis_root)
    native = load_cyj11_native_sax(cyj11_native_stack_root(root, subject), root / "prepared" / subject / "sax" / "observations.npz")
    return native, load_cyj11_endpoint_images(root, subject, case)
