"""Immutable all-source native metrics stratified by sparse-manifest membership.

This evaluator intentionally uses original R0 native geometry for every source
group.  It is distinct from both L100 final-pose metrics and sparse held-out
metrics, which retain their existing scientific meanings.
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
from .evaluation import PSNR_DATA_RANGES_MS, _metrics, masked_ssim, reproject_volume_to_native_plane
from .final_cyj_reporting import _load_all_verified_native_cache_groups
from .final_sparse_native_metrics import sparse_manifest_case, validate_sparse_endpoint_roots


STACKS = ("sax", "2ch", "4ch")
PARAMETERS = ("t1", "t2")
REGIONS = ("sax_observed", "lax_observed", "lax_heldout", "nonheldout_all", "all_lax", "all_source")
SUPPORT_EQUATION = "primary_native_mask AND finite(reference) AND proposed_endpoint_valid AND fit_first_endpoint_valid AND finite(proposed) AND finite(fit_first)"


def _record(path: str | Path) -> dict[str, str]:
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _ids(value: Mapping[str, Any], name: str) -> list[int]:
    raw = value.get(name, [])
    if not isinstance(raw, list) or len(raw) != len(set(map(int, raw))):
        raise ValueError(f"sparse manifest has invalid {name}")
    return sorted(map(int, raw))


def classify_stratified_groups(sparse: Mapping[str, Any], source_groups: Mapping[str, list[int]]) -> tuple[dict[str, dict[str, list[int]] | None], dict[tuple[str, int], str]]:
    """Return disjoint stratum identities solely from the sparse manifest/cache."""
    source = {stack: sorted(map(int, source_groups[stack])) for stack in STACKS}
    if any(len(values) != len(set(values)) for values in source.values()):
        raise ValueError("R0 native cache has duplicate source group identities")
    statuses: dict[tuple[str, int], str] = {}
    if sparse.get("condition") == "L100":
        for stack, groups in source.items():
            statuses.update({(stack, group): "observed" for group in groups})
        lax = {stack: source[stack] for stack in ("2ch", "4ch")}
        all_source = {stack: source[stack] for stack in STACKS}
        return {"sax_observed": {"sax": source["sax"]}, "lax_observed": lax, "lax_heldout": None, "nonheldout_all": all_source, "all_lax": lax, "all_source": all_source}, statuses
    stacks = sparse.get("stacks")
    if not isinstance(stacks, Mapping):
        raise ValueError("sparse manifest stacks are missing")
    observed_lax: dict[str, list[int]] = {}
    heldout_lax: dict[str, list[int]] = {}
    for stack in ("2ch", "4ch"):
        spec = stacks.get(stack)
        if not isinstance(spec, Mapping):
            raise ValueError(f"sparse manifest lacks {stack}")
        observed, heldout = _ids(spec, "retained_original_group_ids"), _ids(spec, "heldout_original_group_ids")
        if set(observed) & set(heldout) or set(observed) | set(heldout) != set(source[stack]):
            raise ValueError(f"{stack}: observed/held-out identities do not partition R0 source groups")
        observed_lax[stack], heldout_lax[stack] = observed, heldout
        statuses.update({(stack, group): "observed" for group in observed})
        statuses.update({(stack, group): "held-out" for group in heldout})
    statuses.update({("sax", group): "observed" for group in source["sax"]})
    return {
        "sax_observed": {"sax": source["sax"]}, "lax_observed": observed_lax, "lax_heldout": heldout_lax,
        "nonheldout_all": {"sax": source["sax"], **observed_lax},
        "all_lax": {"2ch": source["2ch"], "4ch": source["4ch"]},
        "all_source": {stack: source[stack] for stack in STACKS},
    }, statuses


def _aggregate(values: dict[str, list[np.ndarray]], mask_voxels: int, *, parameter: str) -> dict[str, Any]:
    if not values["reference"]:
        return {"status": "EMPTY_COMMON_SUPPORT", "voxel_count": 0, "coverage_fraction": 0.0}
    reference = np.concatenate(values["reference"])
    support = np.ones(reference.size, bool)
    result = {}
    for name in ("proposed", "fit_first"):
        result[name] = _metrics(reference, np.concatenate(values[name]), support,
                                 psnr_data_range=PSNR_DATA_RANGES_MS[parameter])
        ssim_values = values.get(f"{name}_ssim", [])
        result[name]["ssim"] = float(sum(ssim_values) / len(ssim_values)) if ssim_values else None
        result[name]["ssim_status"] = "mean_of_per_plane_common_support_ssim" if ssim_values else "not_available_2d_support"
    result.update({"status": "COMPLETE", "voxel_count": int(reference.size), "coverage_fraction": float(reference.size / mask_voxels) if mask_voxels else 0.0})
    return result


def build_final_stratified_native_metrics(*, proposed_endpoint_root: str | Path, fit_first_endpoint_root: str | Path, native_mask_root: str | Path, r0_native_maps_root: str | Path, sparse_manifest: str | Path, output: str | Path) -> Path:
    """Evaluate all R0 source groups on original native geometry and stratify them."""
    import nibabel as nib

    sparse_manifest, native_mask_root, output = Path(sparse_manifest), Path(native_mask_root), Path(output)
    sparse = json.loads(sparse_manifest.read_text(encoding="utf-8"))
    subject, level = sparse.get("subject"), sparse.get("condition")
    custom = sparse.get("schema_version") == "phase4r_sparse_profile_v2"
    if not isinstance(subject, str) or not subject or sparse.get("status") != "ELIGIBLE" or (not custom and level not in {"L100", "L50", "L25"}):
        raise ValueError("stratified evaluation requires an ELIGIBLE named legacy or explicit custom sparse manifest")
    condition = sparse_manifest_case(sparse) if custom or level != "L100" else "L100_fold0"
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    paths, endpoint_provenance = validate_sparse_endpoint_roots(proposed_endpoint_root, fit_first_endpoint_root, condition, subject=subject)
    images = {parameter: {name: nib.load(str(path)) for name, path in method_paths.items()} for parameter, method_paths in paths.items()}
    cache = {stack: _load_all_verified_native_cache_groups(r0_native_maps_root, sparse_manifest, stack) for stack in STACKS}
    regions, statuses = classify_stratified_groups(sparse, {stack: [int(group) for group in cache[stack]["group_idx"]] for stack in STACKS})
    accumulators = {parameter: {region: {"reference": [], "proposed": [], "fit_first": [],
                                         "proposed_ssim": [], "fit_first_ssim": [], "mask_voxels": 0}
                                for region in REGIONS} for parameter in PARAMETERS}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        planes: list[dict[str, Any]] = []
        for stack in STACKS:
            for index, group_value in enumerate(cache[stack]["group_idx"]):
                group, status = int(group_value), statuses[(stack, int(group_value))]
                affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ np.asarray(cache[stack]["affine_lps_rc"][index], float)
                plane: dict[str, Any] = {"stack": stack, "original_group_idx": group, "membership": status, "parameters": {}}
                for parameter in PARAMETERS:
                    reference = np.asarray(cache[stack][f"{parameter}_ms"][index], float)
                    source = np.asarray(cache[stack]["source_support"][index], bool)
                    masks, mask_provenance = load_validated_current_native_mask_plane(native_mask_root, parameter=parameter, stack=stack, original_group_idx=group, shape_rc=tuple(reference.shape), source_support=source)
                    proposed, proposed_valid = reproject_volume_to_native_plane(images[parameter]["proposed"].get_fdata(), images[parameter]["proposed"].affine, affine, reference.shape)
                    fit_first, fit_valid = reproject_volume_to_native_plane(images[parameter]["fit_first"].get_fdata(), images[parameter]["fit_first"].affine, affine, reference.shape)
                    support = np.asarray(masks["primary_native_mask"], bool) & np.isfinite(reference) & proposed_valid & fit_valid & np.isfinite(proposed) & np.isfinite(fit_first)
                    metrics = None
                    if support.any():
                        metrics = {}
                        for name, values in (("proposed", proposed), ("fit_first", fit_first)):
                            ssim_value, ssim_status = masked_ssim(reference, values, support,
                                                                   data_range=PSNR_DATA_RANGES_MS[parameter])
                            metrics[name] = _metrics(reference, values, support,
                                                     psnr_data_range=PSNR_DATA_RANGES_MS[parameter],
                                                     ssim_value=ssim_value, ssim_status=ssim_status)
                    plane["parameters"][parameter] = {"common_support_voxels": int(support.sum()), "coverage_fraction": float(support.sum() / np.asarray(masks["primary_native_mask"], bool).sum()) if np.asarray(masks["primary_native_mask"], bool).any() else 0.0, "metrics": metrics, "mask_provenance": mask_provenance}
                    for region, members in regions.items():
                        if members is None or group not in members.get(stack, []):
                            continue
                        target = accumulators[parameter][region]
                        target["mask_voxels"] += int(np.asarray(masks["primary_native_mask"], bool).sum())
                        if support.any():
                            for name, values in (("reference", reference), ("proposed", proposed), ("fit_first", fit_first)):
                                target[name].append(np.asarray(values)[support])
                            for name in ("proposed", "fit_first"):
                                value = metrics[name].get("ssim") if metrics is not None else None
                                if value is not None:
                                    target[f"{name}_ssim"].append(float(value))
                planes.append(plane)
        region_metrics: dict[str, Any] = {}
        for parameter in PARAMETERS:
            region_metrics[parameter] = {}
            for region, values in accumulators[parameter].items():
                region_metrics[parameter][region] = {"status": "NOT_APPLICABLE_L100", "group_ids": None} if regions[region] is None else {"group_ids": regions[region], **_aggregate(values, int(values["mask_voxels"]), parameter=parameter)}
        write_immutable_json(staging / "aggregate_metrics.json", {"schema": "final_stratified_native_common_support_metrics/v2", "primary_mask": "primary_native_mask", "primary_support_equation": SUPPORT_EQUATION, "metric_contract": "RMSE, NRMSE, bias, MAE, Pearson CC, NCC, fixed-range PSNR, and mean masked 2-D SSIM", "psnr_data_ranges_ms": PSNR_DATA_RANGES_MS, "regions": region_metrics, "per_plane_records": planes, "original_native_geometry_for_all_strata": True})
        manifest = {"schema": "final_stratified_native_bundle_metrics/v2", "subject": subject, "condition": condition, "endpoint": endpoint_provenance["endpoint"], "endpoint_provenance": endpoint_provenance, "sparse_manifest": _record(sparse_manifest), "native_mask_manifest": _record(native_mask_root / "manifest.json"), "aggregate_metrics": _record(staging / "aggregate_metrics.json"), "metric_contract": "RMSE, NRMSE, bias, MAE, Pearson CC, NCC, fixed-range PSNR, and mean masked 2-D SSIM", "region_definitions": regions, "no_final_pose_voxels": True, "no_training": True, "no_reconstruction": True}
        if custom:
            for field in ("profile_id", "case_id", "fold_id", "fold_count"):
                manifest[field] = sparse[field]
            manifest["sparse_schema_version"] = sparse["schema_version"]
        write_immutable_json(staging / "manifest.json", manifest)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("proposed-endpoint-root", "fit-first-endpoint-root", "native-mask-root", "r0-native-maps-root", "sparse-manifest", "output"):
        parser.add_argument("--" + name, required=True)
    print(build_final_stratified_native_metrics(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
