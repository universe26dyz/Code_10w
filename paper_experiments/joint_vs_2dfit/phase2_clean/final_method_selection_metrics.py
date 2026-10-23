"""Immutable explicit-fold Q002/K8, Q005/K4, and Fit-first selection evidence."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import sha256, write_immutable_json
from .current_native_central_component_mask import load_validated_current_native_mask_plane
from .evaluation import reproject_volume_to_native_plane
from .final_cyj_reporting import _load_all_verified_native_cache_groups
from .final_sparse_native_metrics import validate_sparse_endpoint_roots
from .final_stratified_native_metrics import PARAMETERS, REGIONS, STACKS, classify_stratified_groups
from .sparse_profile_manifest import validate_sparse_profile_manifest

SCHEMA = "phase4r_method_selection_common_support/v1"
BUNDLE_METRICS_SCHEMA = "phase4r_method_selection_stratified_common_support_metrics/v1"
BUNDLE_SCHEMA = "phase4r_method_selection_stratified_common_support_bundle/v1"
SUPPORT_EQUATION = "primary_native_mask AND finite(reference) AND Q002_endpoint_valid AND Q005_endpoint_valid AND Fit_first_endpoint_valid AND finite(Q002) AND finite(Q005) AND finite(Fit_first)"
PROFILE_ID = "S2CH100_4CH50"
Q002_ROUTE = "Q002S640_B6_roi_same_anchor_mse"
Q005_ROUTE = "Q005S640_K4_B6_roi_same_anchor_mse"
METHODS = ("Q002S640", "Q005S640_K4", "Fit-first")
FIT_FIRST_TRAINING_ROUTES = frozenset(("Fit-first sparse checkpoint query",))


def _metrics(reference: np.ndarray, prediction: np.ndarray, support: np.ndarray) -> dict[str, float]:
    delta = np.asarray(prediction)[support] - np.asarray(reference)[support]
    return {"rmse": float(np.sqrt(np.mean(delta ** 2))), "mae": float(np.mean(np.abs(delta))), "bias": float(np.mean(delta))}


def _common_support(*, reference: np.ndarray, q002: np.ndarray, q005: np.ndarray, fit_first: np.ndarray, primary_native_mask: np.ndarray, q002_endpoint_valid: np.ndarray, q005_endpoint_valid: np.ndarray, fit_first_endpoint_valid: np.ndarray) -> np.ndarray:
    """The sole implementation of the three-method native common support."""
    arrays = [np.asarray(value) for value in (reference, q002, q005, fit_first, primary_native_mask, q002_endpoint_valid, q005_endpoint_valid, fit_first_endpoint_valid)]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("method selection arrays must share native-plane shape")
    return np.asarray(primary_native_mask, bool) & np.isfinite(reference) & np.asarray(q002_endpoint_valid, bool) & np.asarray(q005_endpoint_valid, bool) & np.asarray(fit_first_endpoint_valid, bool) & np.isfinite(q002) & np.isfinite(q005) & np.isfinite(fit_first)


def evaluate_method_selection_common_support(*, reference: np.ndarray, q002: np.ndarray, q005: np.ndarray, fit_first: np.ndarray, primary_native_mask: np.ndarray, q002_endpoint_valid: np.ndarray, q005_endpoint_valid: np.ndarray, fit_first_endpoint_valid: np.ndarray, endpoint: str, parameter: str, output: str | Path) -> Path:
    """Write one immutable metric record; endpoint and parameter are never pooled."""
    if endpoint not in {"central", "psf128"} or parameter not in {"t1_ms", "t2_ms"}:
        raise ValueError("method selection metrics require central/psf128 and t1_ms/t2_ms identity")
    target = Path(output)
    if target.exists(): raise FileExistsError(f"refusing overwrite: {target}")
    support = _common_support(reference=reference, q002=q002, q005=q005, fit_first=fit_first, primary_native_mask=primary_native_mask, q002_endpoint_valid=q002_endpoint_valid, q005_endpoint_valid=q005_endpoint_valid, fit_first_endpoint_valid=fit_first_endpoint_valid)
    if not support.any(): raise ValueError("method selection common support is empty")
    target.parent.mkdir(parents=True, exist_ok=True)
    write_immutable_json(target, {"schema": SCHEMA, "subject": "CYJ", "endpoint": endpoint, "parameter": parameter, "support_equation": SUPPORT_EQUATION, "support_voxel_count": int(support.sum()), "coverage_fraction": float(support.sum() / np.asarray(primary_native_mask, bool).sum()) if np.asarray(primary_native_mask, bool).any() else 0.0, "methods": {"Q002S640": _metrics(reference, q002, support), "Q005S640_K4": _metrics(reference, q005, support), "Fit-first": _metrics(reference, fit_first, support)}, "selection_scope": "CYJ method-selection evidence only; not population statistics", "immutable": True})
    return target


def _record(path: str | Path) -> dict[str, str]:
    item = Path(path)
    return {"path": str(item.resolve()), "sha256": sha256(item)}


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def validate_method_selection_sparse_manifest(sparse_manifest: str | Path) -> dict[str, Any]:
    """Admit exactly the selected custom profile and one explicit replication fold."""
    path = Path(sparse_manifest)
    raw = json.loads(path.read_text(encoding="utf-8"))
    subject = raw.get("subject")
    if not isinstance(subject, str) or not subject:
        raise ValueError("method selection sparse manifest requires a named subject")
    sparse = validate_sparse_profile_manifest(path, subject=subject)
    if sparse.get("schema_version") != "phase4r_sparse_profile_v2" or sparse.get("status") != "ELIGIBLE":
        raise ValueError("method selection requires an ELIGIBLE exact custom sparse profile manifest")
    if sparse.get("profile_id") != PROFILE_ID:
        raise ValueError(f"method selection requires profile_id={PROFILE_ID}")
    fold_id, fold_count = sparse.get("fold_id"), sparse.get("fold_count")
    if type(fold_id) is not int or fold_id not in (0, 1) or fold_count != 2:
        raise ValueError("method selection requires exactly one S2CH100_4CH50 fold0 or fold1 manifest")
    expected_case = f"{PROFILE_ID}_fold{fold_id}"
    if sparse.get("case_id") != expected_case:
        raise ValueError("method selection custom profile case_id/fold identity mismatch")
    return dict(sparse)


def _manifest(root: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(root) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, json.loads(path.read_text(encoding="utf-8"))


def _validate_proposed_route(root: str | Path, *, label: str, expected_route: str, expected_samples: int) -> tuple[Path, dict[str, Any]]:
    path, manifest = _manifest(root)
    if manifest.get("schema") != "phase4r_formal_proposed_export/v1" or manifest.get("method") != "Proposed":
        raise ValueError(f"{label} endpoint identity is not a formal Proposed export")
    samples = manifest.get("training_psf", {}).get("training_psf_samples") if isinstance(manifest.get("training_psf"), Mapping) else None
    if manifest.get("training_route") != expected_route or samples != expected_samples:
        raise ValueError(f"{label} endpoint must be {expected_route} / K{expected_samples}")
    return path, manifest


def _validate_fit_first_identity(root: str | Path) -> tuple[Path, dict[str, Any]]:
    path, manifest = _manifest(root)
    if manifest.get("schema") != "phase4r_fixed_l100_fit_first_export_bundle/v1" or manifest.get("method") != "Fit-first":
        raise ValueError("Fit-first endpoint identity is invalid")
    parameters = manifest.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("Fit-first endpoint bundle lacks parameter manifests")
    for parameter in ("t1", "t2"):
        embedded = parameters.get(parameter)
        parameter_path = Path(root) / parameter / "manifest.json"
        if not isinstance(embedded, Mapping):
            raise ValueError(f"Fit-first endpoint bundle lacks embedded {parameter} manifest")
        if not parameter_path.is_file():
            raise FileNotFoundError(parameter_path)
        parameter_manifest = json.loads(parameter_path.read_text(encoding="utf-8"))
        if embedded != parameter_manifest:
            raise ValueError(f"Fit-first embedded {parameter} manifest disagrees with disk manifest")
        if parameter_manifest.get("schema") != "phase4r_fixed_l100_fit_first_export/v1" or parameter_manifest.get("method") != "Fit-first" or parameter_manifest.get("parameter") != parameter:
            raise ValueError(f"Fit-first {parameter} parameter manifest identity is invalid")
        for field in ("subject", "condition", "endpoint"):
            if parameter_manifest.get(field) != manifest.get(field):
                raise ValueError(f"Fit-first {parameter} parameter manifest {field} disagrees with bundle")
        if parameter_manifest.get("training_route") not in FIT_FIRST_TRAINING_ROUTES:
            raise ValueError(f"Fit-first {parameter} parameter manifest training route identity is invalid")
        volume = Path(root) / parameter / "volume_ms.nii.gz"
        if not volume.is_file():
            raise FileNotFoundError(volume)
        provenance = parameter_manifest.get("output_volume_ms")
        if provenance is not None:
            if not isinstance(provenance, Mapping) or not isinstance(provenance.get("path"), str) or not isinstance(provenance.get("sha256"), str) or sha256(volume) != provenance["sha256"]:
                raise ValueError(f"Fit-first {parameter} output_volume_ms provenance mismatch")
    return path, manifest


def _validate_three_endpoint_roots(*, q002_endpoint_root: str | Path, q005_endpoint_root: str | Path, fit_first_endpoint_root: str | Path, subject: str, case_id: str) -> tuple[dict[str, dict[str, Path]], dict[str, Any]]:
    q002_manifest_path, q002_manifest = _validate_proposed_route(q002_endpoint_root, label="Q002", expected_route=Q002_ROUTE, expected_samples=8)
    q005_manifest_path, q005_manifest = _validate_proposed_route(q005_endpoint_root, label="Q005", expected_route=Q005_ROUTE, expected_samples=4)
    fit_manifest_path, fit_manifest = _validate_fit_first_identity(fit_first_endpoint_root)
    q002_paths, q002_provenance = validate_sparse_endpoint_roots(q002_endpoint_root, fit_first_endpoint_root, case_id, subject=subject)
    q005_paths, q005_provenance = validate_sparse_endpoint_roots(q005_endpoint_root, fit_first_endpoint_root, case_id, subject=subject)
    endpoint = q002_provenance["endpoint"]
    if endpoint not in {"central", "psf128"} or q005_provenance["endpoint"] != endpoint:
        raise ValueError("Q002/Q005/Fit-first endpoint mismatch")
    if any(manifest.get("subject") != subject or manifest.get("condition") != case_id or manifest.get("endpoint") != endpoint for manifest in (q002_manifest, q005_manifest, fit_manifest)):
        raise ValueError("three-method endpoint subject/case/endpoint identity mismatch")
    paths = {parameter: {"Q002S640": q002_paths[parameter]["proposed"], "Q005S640_K4": q005_paths[parameter]["proposed"], "Fit-first": q002_paths[parameter]["fit_first"]} for parameter in PARAMETERS}
    provenance = {"endpoint": endpoint, "q002_endpoint_manifest": _record(q002_manifest_path), "q005_endpoint_manifest": _record(q005_manifest_path), "fit_first_endpoint_manifest": _record(fit_manifest_path), "q002_endpoint_validation": q002_provenance, "q005_endpoint_validation": q005_provenance}
    return paths, provenance


def _aggregate(values: dict[str, list[np.ndarray]], primary_mask_voxels: int) -> dict[str, Any]:
    if not values["reference"]:
        return {"status": "EMPTY_COMMON_SUPPORT", "voxel_count": 0, "primary_mask_voxel_count": primary_mask_voxels, "coverage_fraction": 0.0, **{method: None for method in METHODS}}
    reference = np.concatenate(values["reference"])
    support = np.ones(reference.size, dtype=bool)
    result = {method: _metrics(reference, np.concatenate(values[method]), support) for method in METHODS}
    return {"status": "COMPLETE", "voxel_count": int(reference.size), "primary_mask_voxel_count": primary_mask_voxels, "coverage_fraction": float(reference.size / primary_mask_voxels) if primary_mask_voxels else 0.0, **result}


def build_final_method_selection_bundle(*, q002_endpoint_root: str | Path, q005_endpoint_root: str | Path, fit_first_endpoint_root: str | Path, native_mask_root: str | Path, r0_native_maps_root: str | Path, sparse_manifest: str | Path, output: str | Path) -> Path:
    """Evaluate one explicit S2CH100_4CH50 fold; never pool folds or run reconstruction."""
    import nibabel as nib

    output, sparse_manifest, native_mask_root = Path(output), Path(sparse_manifest), Path(native_mask_root)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    sparse = validate_method_selection_sparse_manifest(sparse_manifest)
    subject, case_id = str(sparse["subject"]), str(sparse["case_id"])
    paths, endpoint_provenance = _validate_three_endpoint_roots(q002_endpoint_root=q002_endpoint_root, q005_endpoint_root=q005_endpoint_root, fit_first_endpoint_root=fit_first_endpoint_root, subject=subject, case_id=case_id)
    images = {parameter: {method: nib.load(str(path)) for method, path in method_paths.items()} for parameter, method_paths in paths.items()}
    cache = {stack: _load_all_verified_native_cache_groups(r0_native_maps_root, sparse_manifest, stack) for stack in STACKS}
    regions, statuses = classify_stratified_groups(sparse, {stack: [int(group) for group in cache[stack]["group_idx"]] for stack in STACKS})
    accumulators = {parameter: {region: {"reference": [], **{method: [] for method in METHODS}, "primary_mask_voxels": 0} for region in REGIONS} for parameter in PARAMETERS}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        planes: list[dict[str, Any]] = []
        for stack in STACKS:
            for index, group_value in enumerate(cache[stack]["group_idx"]):
                group = int(group_value)
                affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ np.asarray(cache[stack]["affine_lps_rc"][index], float)
                plane: dict[str, Any] = {"stack": stack, "original_group_idx": group, "membership": statuses[(stack, group)], "parameters": {}}
                for parameter in PARAMETERS:
                    reference = np.asarray(cache[stack][f"{parameter}_ms"][index], float)
                    source = np.asarray(cache[stack]["source_support"][index], bool)
                    masks, mask_provenance = load_validated_current_native_mask_plane(native_mask_root, parameter=parameter, stack=stack, original_group_idx=group, shape_rc=tuple(reference.shape), source_support=source)
                    predictions: dict[str, np.ndarray] = {}
                    valid: dict[str, np.ndarray] = {}
                    for method in METHODS:
                        predictions[method], valid[method] = reproject_volume_to_native_plane(images[parameter][method].get_fdata(), images[parameter][method].affine, affine, reference.shape)
                    primary = np.asarray(masks["primary_native_mask"], bool)
                    support = _common_support(reference=reference, q002=predictions["Q002S640"], q005=predictions["Q005S640_K4"], fit_first=predictions["Fit-first"], primary_native_mask=primary, q002_endpoint_valid=valid["Q002S640"], q005_endpoint_valid=valid["Q005S640_K4"], fit_first_endpoint_valid=valid["Fit-first"])
                    metric = None if not support.any() else {method: _metrics(reference, predictions[method], support) for method in METHODS}
                    plane["parameters"][parameter] = {"status": "COMPLETE" if support.any() else "EMPTY_COMMON_SUPPORT", "primary_mask": "primary_native_mask", "primary_mask_voxel_count": int(primary.sum()), "common_support_voxels": int(support.sum()), "coverage_fraction": float(support.sum() / primary.sum()) if primary.any() else 0.0, "metrics": metric, "mask_provenance": mask_provenance}
                    for region, members in regions.items():
                        if members is None or group not in members.get(stack, []):
                            continue
                        target = accumulators[parameter][region]
                        target["primary_mask_voxels"] += int(primary.sum())
                        if support.any():
                            target["reference"].append(reference[support])
                            for method in METHODS:
                                target[method].append(np.asarray(predictions[method])[support])
                planes.append(plane)
        region_metrics = {parameter: {region: ({"status": "NOT_APPLICABLE_L100", "group_ids": None, "voxel_count": 0, "primary_mask_voxel_count": 0, "coverage_fraction": 0.0, **{method: None for method in METHODS}} if regions[region] is None else {"group_ids": regions[region], **_aggregate(values, int(values["primary_mask_voxels"]))}) for region, values in accumulators[parameter].items()} for parameter in PARAMETERS}
        metrics_path = staging / "aggregate_metrics.json"
        write_immutable_json(metrics_path, {"schema": BUNDLE_METRICS_SCHEMA, "primary_mask": "primary_native_mask", "support_equation": SUPPORT_EQUATION, "regions": region_metrics, "per_plane_records": planes, "original_native_geometry_for_all_strata": True, "fold_pooling": "forbidden; one explicit case/fold per bundle"})
        manifest = {"schema": BUNDLE_SCHEMA, "subject": subject, "case_id": case_id, "condition": case_id, "profile_id": sparse["profile_id"], "fold_id": sparse["fold_id"], "fold_count": sparse["fold_count"], "endpoint": endpoint_provenance["endpoint"], "q002_endpoint_manifest": endpoint_provenance["q002_endpoint_manifest"], "q005_endpoint_manifest": endpoint_provenance["q005_endpoint_manifest"], "fit_first_endpoint_manifest": endpoint_provenance["fit_first_endpoint_manifest"], "endpoint_provenance": endpoint_provenance, "sparse_manifest": _record(sparse_manifest), "native_mask_manifest": _record(native_mask_root / "manifest.json"), "r0_native_source_provenance": {stack: cache[stack].get("provenance") for stack in STACKS}, "aggregate_metrics": _record(metrics_path), "support_equation": SUPPORT_EQUATION, "code_git_sha": _git_sha(), "no_training": True, "no_refitting": True, "no_reconstruction": True, "immutable": True}
        write_immutable_json(staging / "manifest.json", manifest)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("q002-endpoint-root", "q005-endpoint-root", "fit-first-endpoint-root", "native-mask-root", "r0-native-maps-root", "sparse-manifest", "output"):
        parser.add_argument("--" + name, required=True)
    print(build_final_method_selection_bundle(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
