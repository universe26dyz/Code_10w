"""Formal sparse held-out native metrics on frozen current-native masks."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import sha256, write_immutable_json
from .current_native_central_component_mask import load_validated_current_native_mask_plane
from .evaluation import _metrics, reproject_binary_mask_to_native_plane, reproject_volume_to_native_plane
from .fit_first import select_verified_native_cache_groups
from .heldout import _heldout_ids
from .matched_output import require_matched_endpoints, verified_fit_first_effective_query_mask
from .fit_first_route_contract import EXP1_1_SCHEMA


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _published_record(staging_path: Path, published_path: Path) -> dict[str, str]:
    return {"path": str(published_path.resolve()), "sha256": sha256(staging_path)}


def _manifest(root: Path) -> tuple[Path, dict[str, Any]]:
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, json.loads(path.read_text(encoding="utf-8"))


def sparse_manifest_case(sparse: dict[str, Any]) -> str:
    """Return a custom explicit case ID or the narrow legacy level/phase ID."""
    if sparse.get("schema_version") == "phase4r_sparse_profile_v2":
        case_id, profile_id = sparse.get("case_id"), sparse.get("profile_id")
        if not isinstance(case_id, str) or not case_id or not isinstance(profile_id, str) or not profile_id:
            raise ValueError("custom sparse manifest requires explicit profile_id and case_id")
        return case_id
    level, phase = sparse.get("condition"), sparse.get("phase")
    valid = {"L50": (0, 1), "L25": (0, 1, 2, 3)}
    if level not in valid or type(phase) is not int or phase not in valid[level]:
        raise ValueError("sparse formal evaluator supports L50/L25 phases only")
    return f"{level}_fold{phase}"


def validate_sparse_heldout_admission(sparse: dict[str, Any]) -> dict[str, Any]:
    """Admit a sparse case with one or two held-out LAX stacks, never neither."""
    case_id = sparse_manifest_case(sparse)
    stacks = sparse.get("stacks")
    if not isinstance(stacks, dict):
        raise ValueError("sparse formal evaluator requires stack records")
    heldout: dict[str, list[int]] = {}
    for stack in ("2ch", "4ch"):
        item = stacks.get(stack)
        values = item.get("heldout_original_group_ids") if isinstance(item, dict) else None
        if not isinstance(values, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError(f"{stack}: heldout_original_group_ids must be an integer list")
        heldout[stack] = [int(value) for value in values]
    total = sum(len(values) for values in heldout.values())
    if total == 0:
        raise ValueError("sparse formal evaluator requires at least one held-out LAX plane")
    return {"case_id": case_id, "heldout": heldout, "total_heldout_lax": total}


def validate_sparse_endpoint_roots(proposed_root: str | Path, fit_first_root: str | Path, condition: str, *, subject: str = "CYJ") -> tuple[dict[str, Path], dict[str, Any]]:
    """Validate matched immutable endpoint exports before any native sampling."""
    proposed_root, fit_first_root = Path(proposed_root), Path(fit_first_root)
    proposed_path, proposed = _manifest(proposed_root); fit_path, fit = _manifest(fit_first_root)
    if proposed.get("schema") != "phase4r_formal_proposed_export/v1":
        raise ValueError("Proposed formal endpoint schema mismatch")
    if fit.get("schema") != "phase4r_fixed_l100_fit_first_export_bundle/v2":
        raise ValueError("Fit-first formal endpoint schema mismatch")
    routes = fit.get("fit_first_routes")
    if not isinstance(routes, dict) or any(not isinstance(routes.get(name), dict) or routes[name].get("route_schema") != EXP1_1_SCHEMA for name in ("l100", "sparse")):
        raise ValueError("Fit-first formal endpoint is not bound to CYJ Experiment 1.1 reconstruction routes")
    for name, payload in (("Proposed", proposed), ("Fit-first", fit)):
        if payload.get("subject") != subject or payload.get("condition") != condition:
            raise ValueError(f"{name} endpoint subject/condition mismatch")
    require_matched_endpoints(str(proposed.get("endpoint")), str(fit.get("endpoint")))
    paths = {
        "t1": {"proposed": proposed_root / "T1_3D.nii.gz", "fit_first": fit_first_root / "t1" / "volume_ms.nii.gz"},
        "t2": {"proposed": proposed_root / "T2_3D.nii.gz", "fit_first": fit_first_root / "t2" / "volume_ms.nii.gz"},
    }
    for parameter in paths:
        for path in paths[parameter].values():
            if not path.is_file():
                raise FileNotFoundError(path)
    effective_masks = {parameter: verified_fit_first_effective_query_mask(fit_first_root, parameter) for parameter in paths}
    return paths, {"endpoint": proposed["endpoint"], "proposed_manifest": _record(proposed_path), "fit_first_manifest": _record(fit_path), "fit_first_effective_query_masks": {parameter: record for parameter, (_path, record) in effective_masks.items()}}


def _aggregate(values: dict[str, list[np.ndarray]]) -> dict[str, Any]:
    if not values["reference"]:
        return {"status": "EMPTY_COMMON_SUPPORT", "voxel_count": 0}
    reference = np.concatenate(values["reference"]); support = np.ones(reference.size, bool)
    return {name: _metrics(reference, np.concatenate(values[name]), support) for name in ("proposed", "fit_first")}


def build_final_sparse_native_metrics(*, proposed_endpoint_root: str | Path, fit_first_endpoint_root: str | Path, native_mask_root: str | Path, r0_native_maps_root: str | Path, sparse_manifest: str | Path, output: str | Path) -> Path:
    """Evaluate only sparse-manifest held-out LAX groups on original native geometry."""
    import nibabel as nib

    sparse_manifest, native_mask_root, output = Path(sparse_manifest), Path(native_mask_root), Path(output)
    sparse = json.loads(sparse_manifest.read_text(encoding="utf-8"))
    subject = sparse.get("subject")
    if not isinstance(subject, str) or not subject:
        raise ValueError("sparse formal evaluator requires a named subject")
    admission = validate_sparse_heldout_admission(sparse)
    condition = admission["case_id"]
    if sparse.get("status") != "ELIGIBLE":
        raise ValueError("sparse formal evaluator requires an ELIGIBLE manifest")
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    volume_paths, endpoint_provenance = validate_sparse_endpoint_roots(proposed_endpoint_root, fit_first_endpoint_root, condition, subject=subject)
    images = {parameter: {name: nib.load(str(path)) for name, path in volumes.items()} for parameter, volumes in volume_paths.items()}
    effective_masks = {parameter: nib.load(str(verified_fit_first_effective_query_mask(fit_first_endpoint_root, parameter)[0])) for parameter in volume_paths}
    for parameter, mask in effective_masks.items():
        fit_volume = images[parameter]["fit_first"]
        if mask.shape != fit_volume.shape or not np.allclose(mask.affine, fit_volume.affine, atol=1e-6, rtol=0):
            raise ValueError(f"{parameter}: formal effective query mask geometry differs from Fit-first endpoint volume")
        if not np.all(np.isin(mask.get_fdata(), (0.0, 1.0))):
            raise ValueError(f"{parameter}: formal effective query mask is not binary")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        records: list[dict[str, Any]] = []
        aggregate = {parameter: {stack: {name: [] for name in ("reference", "proposed", "fit_first")} for stack in ("2ch", "4ch", "all_lax")} for parameter in ("t1", "t2")}
        for stack in ("2ch", "4ch"):
            ids = admission["heldout"][stack]
            if not ids:
                continue
            cache = select_verified_native_cache_groups(r0_native_maps_root, sparse_manifest, stack, ids)
            if [int(value) for value in cache["group_idx"]] != ids:
                raise ValueError(f"{stack}: held-out cache selection is not exact manifest order")
            for index, group in enumerate(ids):
                affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ np.asarray(cache["affine_lps_rc"][index], float)
                plane: dict[str, Any] = {"stack": stack, "original_group_idx": int(group), "status": "HELDOUT_ORIGINAL_NATIVE_GEOMETRY", "parameters": {}}
                for parameter in ("t1", "t2"):
                    reference = np.asarray(cache[f"{parameter}_ms"][index], float); source = np.asarray(cache["source_support"][index], bool)
                    masks, mask_provenance = load_validated_current_native_mask_plane(native_mask_root, parameter=parameter, stack=stack, original_group_idx=int(group), shape_rc=tuple(reference.shape), source_support=source)
                    proposed, proposed_valid = reproject_volume_to_native_plane(images[parameter]["proposed"].get_fdata(), images[parameter]["proposed"].affine, affine, reference.shape)
                    fit_first, fit_volume_geometry_valid = reproject_volume_to_native_plane(images[parameter]["fit_first"].get_fdata(), images[parameter]["fit_first"].affine, affine, reference.shape)
                    fit_effective_mask = reproject_binary_mask_to_native_plane(effective_masks[parameter].get_fdata(), effective_masks[parameter].affine, affine, reference.shape)
                    support = np.asarray(masks["primary_native_mask"], bool) & np.isfinite(reference) & proposed_valid & fit_volume_geometry_valid & fit_effective_mask & np.isfinite(proposed) & np.isfinite(fit_first)
                    metrics = None if not support.any() else {name: _metrics(reference, values, support) for name, values in (("proposed", proposed), ("fit_first", fit_first))}
                    plane["parameters"][parameter] = {"primary_mask": "primary_native_mask", "common_support_voxels": int(support.sum()), "coverage_fraction": float(support.sum() / np.asarray(masks["primary_native_mask"], bool).sum()) if np.asarray(masks["primary_native_mask"], bool).any() else 0.0, "metrics": metrics, "mask_provenance": mask_provenance, "fit_first_effective_query_mask": endpoint_provenance["fit_first_effective_query_masks"][parameter], "fit_first_effective_query_mask_mapping": "nearest-neighbor order=0 to native plane"}
                    if support.any():
                        for target in (stack, "all_lax"):
                            for name, values in (("reference", reference), ("proposed", proposed), ("fit_first", fit_first)):
                                aggregate[parameter][target][name].append(np.asarray(values)[support])
                records.append(plane)
        regions = {parameter: {stack: _aggregate(values) for stack, values in stacks.items()} for parameter, stacks in aggregate.items()}
        metrics_payload = {"schema": "final_sparse_native_common_support_metrics/v2", "primary_mask": "primary_native_mask", "primary_support_equation": "primary_native_mask AND finite(reference) AND proposed_valid AND fit_first_volume_geometry_valid AND fit_first_query_effective_mask AND finite(proposed, fit_first)", "regions": regions, "per_plane_records": records, "heldout_only": True, "native_geometry": "original held-out R0 native affine; no_final_pose_lookup", "fit_first_mask_semantics": "formal SCMR one-stage Fit-first; effective query mask is part of pairwise common support"}
        aggregate_staging, aggregate_final = staging / "aggregate_metrics.json", output / "aggregate_metrics.json"
        write_immutable_json(aggregate_staging, metrics_payload)
        write_immutable_json(staging / "manifest.json", {"schema": "final_sparse_native_bundle_metrics/v2", "subject": subject, "condition": condition, "endpoint": endpoint_provenance["endpoint"], "endpoint_provenance": endpoint_provenance, "sparse_manifest": _record(sparse_manifest), "native_mask_manifest": _record(native_mask_root / "manifest.json"), "aggregate_metrics": _published_record(aggregate_staging, aggregate_final), "primary_mask": "primary_native_mask", "primary_support_equation": "primary_native_mask AND finite(reference) AND proposed_valid AND fit_first_volume_geometry_valid AND fit_first_query_effective_mask AND finite(proposed, fit_first)", "metric_contract": "formal SCMR one-stage Fit-first; effective query mask is part of pairwise common support", "heldout_only": True, "no_learned_pose": True, "no_training": True, "no_reconstruction": True})
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("proposed-endpoint-root", "fit-first-endpoint-root", "native-mask-root", "r0-native-maps-root", "sparse-manifest", "output"):
        parser.add_argument("--" + name, required=True)
    print(build_final_sparse_native_metrics(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
