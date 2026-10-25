"""Immutable Final5 native metrics split into all, observed, and held-out planes.

Observed planes are sampled only from final-pose native bundles.  Held-out
planes are sampled only by the established endpoint-to-original-native route.
The output is an additive reporting product and never replaces either CYJ 1.1
L100 final-pose metrics or held-out sparse metrics.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import sha256, write_immutable_json
from .current_native_central_component_mask import load_validated_current_native_mask_plane
from .evaluation import _metrics, reproject_binary_mask_to_native_plane, reproject_volume_to_native_plane
from .final5_publication_visualization import publication_group_roles, sampling_geometry_for_role
from .final_cyj_reporting import _load_all_verified_native_cache_groups
from .final_native_grid_metrics import native_ras_affine
from .final_sparse_native_metrics import validate_sparse_endpoint_roots
from .matched_output import verified_fit_first_effective_query_mask


STACKS = ("sax", "2ch", "4ch")
PARAMETERS = ("t1", "t2")
SCHEMA = "final5_partitioned_native_metrics/v1"
SUPPORT_EQUATION = "primary_native_mask AND finite(reference) AND proposed_valid AND fit_first_volume_geometry_valid AND fit_first_query_effective_mask AND finite(proposed, fit_first)"


def _record(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def partition_group_roles(roles: Mapping[tuple[str, int], str]) -> dict[str, set[tuple[str, int]]]:
    """Return disjoint named groups without changing declared sparse roles."""
    all_groups = set(roles)
    observed = {key for key, role in roles.items() if role == "observed"}
    heldout = {key for key, role in roles.items() if role == "heldout"}
    if observed | heldout != all_groups or observed & heldout:
        raise ValueError("plane roles must partition all groups into observed and heldout")
    return {"all": all_groups, "observed": observed, "heldout": heldout}


def _bundle_plane(manifest: Mapping[str, Any], root: Path, *, parameter: str, stack: str, group: int,
                  method: str) -> tuple[np.ndarray, np.ndarray]:
    if method == "proposed":
        record = next((item for item in manifest.get("planes", []) if item.get("stack") == stack and item.get("original_group_idx") == group), None)
        if not isinstance(record, Mapping):
            raise ValueError(f"observed {stack}/group{group:03d} missing Proposed final-pose bundle")
        with np.load(root / str(record["path"]), allow_pickle=False) as data:
            values = np.asarray(data[f"{parameter}_ms"], float)
        return values, np.isfinite(values)
    path = root / "planes" / parameter / stack / f"group{group:03d}.npz"
    if not path.is_file():
        raise ValueError(f"observed {parameter}/{stack}/group{group:03d} missing Fit-first final-pose bundle")
    with np.load(path, allow_pickle=False) as data:
        values = np.asarray(data["prediction_ms"], float)
        geometry = np.asarray(data["volume_geometry_valid"], bool)
        query = np.asarray(data["query_effective_mask"], bool)
    return values, geometry & query & np.isfinite(values)


def _validate_bundles(*, direct_root: Path, fit_root: Path, subject: str, case: str,
                      fit_endpoint: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    direct_path, fit_path = direct_root / "manifest.json", fit_root / "manifest.json"
    direct, fit = json.loads(direct_path.read_text(encoding="utf-8")), json.loads(fit_path.read_text(encoding="utf-8"))
    direct_schema = "direct_native_prediction_bundle/v1" if case == "L100" else "direct_sparse_native_prediction_bundle/v1"
    if direct.get("schema") != direct_schema or direct.get("endpoint") != "central":
        raise ValueError("Proposed final-pose native bundle schema/endpoint mismatch")
    if case != "L100" and (direct.get("subject"), direct.get("case_id")) != (subject, case):
        raise ValueError("Proposed sparse final-pose native bundle subject/case mismatch")
    if fit.get("schema") != "fit_first_native_prediction_bundle/v6" or fit.get("endpoint") != "central":
        raise ValueError("Fit-first final-pose native bundle schema/endpoint mismatch")
    if fit.get("endpoint_source_manifest") != _record(fit_endpoint / "manifest.json"):
        raise ValueError("Fit-first final-pose native bundle endpoint provenance mismatch")
    return direct, fit


def _aggregate(values: dict[str, dict[str, list[np.ndarray]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for partition, items in values.items():
        if not items["reference"]:
            result[partition] = {"status": "EMPTY_COMMON_SUPPORT", "common_support_voxels": 0}
            continue
        reference = np.concatenate(items["reference"])
        support = np.ones(reference.size, bool)
        result[partition] = {"status": "COMPLETE", "common_support_voxels": int(reference.size),
                             "proposed": _metrics(reference, np.concatenate(items["proposed"]), support),
                             "fit_first": _metrics(reference, np.concatenate(items["fit_first"]), support)}
    return result


def build_final5_partitioned_native_metrics(*, proposed_endpoint_root: str | Path, fit_first_endpoint_root: str | Path,
                                            proposed_native_bundle_root: str | Path, fit_first_native_bundle_root: str | Path,
                                            native_mask_root: str | Path, r0_native_maps_root: str | Path,
                                            sparse_manifest: str | Path, output: str | Path) -> Path:
    """Write one fresh role-partitioned metric bundle from declared immutable inputs."""
    import nibabel as nib

    proposed_endpoint, fit_endpoint, direct_root, fit_root, mask_root, native_root, sparse_path, output = (
        Path(value) for value in (proposed_endpoint_root, fit_first_endpoint_root, proposed_native_bundle_root,
                                  fit_first_native_bundle_root, native_mask_root, r0_native_maps_root,
                                  sparse_manifest, output))
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    sparse = json.loads(sparse_path.read_text(encoding="utf-8"))
    subject, case = sparse.get("subject"), sparse.get("case_id")
    if not isinstance(subject, str) or not isinstance(case, str) or sparse.get("status") != "ELIGIBLE":
        raise ValueError("partitioned native metrics require an eligible named sparse manifest")
    volumes, endpoint = validate_sparse_endpoint_roots(proposed_endpoint, fit_endpoint, case, subject=subject)
    images = {parameter: {method: nib.load(str(path)) for method, path in paths.items()} for parameter, paths in volumes.items()}
    effective = {parameter: nib.load(str(verified_fit_first_effective_query_mask(fit_endpoint, parameter)[0])) for parameter in PARAMETERS}
    direct, fit = _validate_bundles(direct_root=direct_root, fit_root=fit_root, subject=subject, case=case, fit_endpoint=fit_endpoint)
    caches = {stack: _load_all_verified_native_cache_groups(native_root, sparse_path, stack) for stack in STACKS}
    roles = publication_group_roles(sparse, {stack: [int(value) for value in caches[stack]["group_idx"]] for stack in STACKS})
    partitions = partition_group_roles(roles)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        per_plane: list[dict[str, Any]] = []
        pooled = {parameter: {partition: {name: [] for name in ("reference", "proposed", "fit_first")} for partition in partitions} for parameter in PARAMETERS}
        for stack in STACKS:
            cache = caches[stack]
            for index, group_value in enumerate(cache["group_idx"]):
                group = int(group_value); role = roles[(stack, group)]; affine = native_ras_affine(cache["affine_lps_rc"][index])
                plane: dict[str, Any] = {"stack": stack, "original_group_idx": group, "sampling_role": role,
                                         "sampling_geometry": sampling_geometry_for_role(role), "parameters": {}}
                for parameter in PARAMETERS:
                    reference = np.asarray(cache[f"{parameter}_ms"][index], float)
                    source = np.asarray(cache["source_support"][index], bool)
                    primary, provenance = load_validated_current_native_mask_plane(mask_root, parameter=parameter, stack=stack,
                                                                                    original_group_idx=group, shape_rc=tuple(reference.shape),
                                                                                    source_support=source)
                    primary_mask = np.asarray(primary["primary_native_mask"], bool)
                    if role == "observed":
                        proposed, pvalid = _bundle_plane(direct, direct_root, parameter=parameter, stack=stack, group=group, method="proposed")
                        fit_first, fvalid = _bundle_plane(fit, fit_root, parameter=parameter, stack=stack, group=group, method="fit_first")
                    else:
                        proposed, pvalid = reproject_volume_to_native_plane(images[parameter]["proposed"].get_fdata(), images[parameter]["proposed"].affine, affine, reference.shape)
                        fit_first, fgeometry = reproject_volume_to_native_plane(images[parameter]["fit_first"].get_fdata(), images[parameter]["fit_first"].affine, affine, reference.shape)
                        query = reproject_binary_mask_to_native_plane(effective[parameter].get_fdata(), effective[parameter].affine, affine, reference.shape)
                        fvalid = fgeometry & query & np.isfinite(fit_first)
                    if not (reference.shape == proposed.shape == fit_first.shape == pvalid.shape == fvalid.shape == primary_mask.shape):
                        raise ValueError("native plane shape mismatch")
                    support = primary_mask & np.isfinite(reference) & pvalid & fvalid & np.isfinite(proposed) & np.isfinite(fit_first)
                    plane["parameters"][parameter] = {"primary_mask": "primary_native_mask", "common_support_voxels": int(support.sum()),
                                                        "support_equation": SUPPORT_EQUATION, "mask_provenance": provenance,
                                                        "proposed": None if not support.any() else _metrics(reference, proposed, support),
                                                        "fit_first": None if not support.any() else _metrics(reference, fit_first, support)}
                    for partition, members in partitions.items():
                        if (stack, group) in members and support.any():
                            pooled[parameter][partition]["reference"].append(reference[support])
                            pooled[parameter][partition]["proposed"].append(proposed[support])
                            pooled[parameter][partition]["fit_first"].append(fit_first[support])
                per_plane.append(plane)
        aggregate = {parameter: _aggregate(values) for parameter, values in pooled.items()}
        aggregate_path = staging / "aggregate_metrics.json"
        write_immutable_json(aggregate_path, {"schema": SCHEMA, "subject": subject, "case": case, "endpoint": endpoint["endpoint"],
                                              "partitions": aggregate, "support_equation": SUPPORT_EQUATION,
                                              "geometry_policy": {"observed": "final learned pose native bundles", "heldout": "original native endpoint reprojection"}})
        write_immutable_json(staging / "per_plane_metrics.json", {"schema": SCHEMA, "planes": per_plane})
        write_immutable_json(staging / "manifest.json", {"schema": SCHEMA, "subject": subject, "case": case, "endpoint": endpoint["endpoint"],
            "sparse_manifest": _record(sparse_path), "endpoint_provenance": endpoint,
            "proposed_native_bundle": _record(direct_root / "manifest.json"), "fit_first_native_bundle": _record(fit_root / "manifest.json"),
            "native_mask_manifest": _record(mask_root / "manifest.json"), "aggregate_metrics": {"path": str((output / "aggregate_metrics.json").resolve()), "sha256": sha256(aggregate_path)},
            "support_equation": SUPPORT_EQUATION, "partitions": {name: "declared sparse-manifest roles" for name in partitions},
            "no_training": True, "no_reconstruction": True, "no_existing_metrics_overwritten": True})
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("proposed-endpoint-root", "fit-first-endpoint-root", "proposed-native-bundle-root", "fit-first-native-bundle-root", "native-mask-root", "r0-native-maps-root", "sparse-manifest", "output"):
        parser.add_argument("--" + name, required=True)
    print(build_final5_partitioned_native_metrics(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
