"""Final5 formal metrics on one original R0 native physical grid per plane.

This is deliberately separate from legacy final-pose and central-component
outputs.  L100 support is a frozen sampling/reconstruction domain, never an
anatomical segmentation and never a connected-component selection target.
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
from .evaluation import reproject_binary_mask_to_native_plane, reproject_volume_to_native_plane
from .final_sparse_native_metrics import validate_sparse_endpoint_roots
from .fit_first import select_verified_native_cache_groups
from .l100_mask_paired_eval import (
    REGIONS, SUPPORT_EQUATIONS, _delta, _pooled_metrics, _save_native_masks,
    mask_coverage, metrics_on_exact_support, native_support_regions,
    paired_support, pooled_mask_coverage, recovered_distribution,
    render_plane_panels,
)
from .matched_output import verified_fit_first_effective_query_mask, verified_fit_first_sparse_checkpoint_mask


STACKS = ("sax", "2ch", "4ch")
PARAMETERS = ("t1", "t2")
COMMON_SUPPORT_EQUATION = (
    "region AND proposed_reprojection_valid AND fit_first_volume_geometry_valid "
    "AND finite(reference, proposed, fit_first)"
)


def require_new_output(output: str | Path) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    return output


def native_ras_affine(affine_lps_rc: np.ndarray) -> np.ndarray:
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ np.asarray(affine_lps_rc, float)


def native_grid_regions(source_support: np.ndarray, reference: np.ndarray, l100_mask: np.ndarray, sparse_mask: np.ndarray,
                        manual_outer_region: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Historical CYJ 1.1 regions, without any morphological operation."""
    if manual_outer_region is None:
        return native_support_regions(source_support, reference, l100_mask, sparse_mask)
    arrays = (source_support, reference, l100_mask, sparse_mask, manual_outer_region)
    if len({np.asarray(value).shape for value in arrays}) != 1:
        raise ValueError("manual outer segment and native metric arrays must share one plane shape")
    outer = np.asarray(manual_outer_region, bool)
    return native_support_regions(np.asarray(source_support, bool) & outer, reference, l100_mask, sparse_mask)


def l100_region_mask(fit_first_effective_query_mask: np.ndarray) -> np.ndarray:
    """The historical formal L100 sampling domain is the Fit-first query mask."""
    return np.asarray(fit_first_effective_query_mask, bool)


def _binary_mask_data(mask: Any, parameter: str, name: str) -> np.ndarray:
    values = np.asarray(mask.get_fdata(), float)
    if values.ndim != 3:
        raise ValueError(f"{parameter}: {name} mask must be three-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{parameter}: {name} mask is not finite")
    if not np.all(np.isin(values, (0.0, 1.0))):
        raise ValueError(f"{parameter}: {name} mask is not binary")
    return values


def _validate_mapping_affine(affine: np.ndarray, parameter: str, name: str) -> np.ndarray:
    affine = np.asarray(affine, float)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise ValueError(f"{parameter}: {name} mask affine is not finite 4x4 geometry")
    if not np.allclose(affine[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6, rtol=0):
        raise ValueError(f"{parameter}: {name} mask affine is not a homogeneous world mapping")
    try:
        np.linalg.inv(affine)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{parameter}: {name} mask affine is not invertible") from exc
    return affine


def _validate_effective_query_mask(mask: Any, fit_first: Any, parameter: str) -> np.ndarray:
    """Keep the sampled Fit-first query domain on its exact endpoint grid."""
    values = _binary_mask_data(mask, parameter, "effective query")
    if mask.shape != fit_first.shape or not np.allclose(mask.affine, fit_first.affine, atol=1e-6, rtol=0):
        raise ValueError(f"{parameter}: effective query mask geometry differs from Fit-first endpoint volume")
    return values


def _validate_sparse_checkpoint_mask(mask: Any, parameter: str) -> np.ndarray:
    """Validate the raw checkpoint mask before world-coordinate reprojection."""
    values = _binary_mask_data(mask, parameter, "sparse checkpoint")
    _validate_mapping_affine(mask.affine, parameter, "sparse checkpoint")
    return values


def _mask_geometry(mask: Any) -> dict[str, Any]:
    return {"shape": [int(value) for value in mask.shape], "affine": np.asarray(mask.affine, float).tolist()}


def mask_overlap_counts(first: np.ndarray, second: np.ndarray) -> dict[str, int | float]:
    """Deterministic binary-mask comparison for read-only historical audits."""
    first, second = np.asarray(first, bool), np.asarray(second, bool)
    if first.shape != second.shape:
        raise ValueError("mask overlap requires equal native-plane shapes")
    intersection, union = int((first & second).sum()), int((first | second).sum())
    return {"intersection_voxels": intersection, "union_voxels": union, "xor_voxels": int((first ^ second).sum()), "iou": float(intersection / union) if union else 1.0}


def exact_paired_support(region: np.ndarray, reference: np.ndarray, proposed: np.ndarray, fit_first: np.ndarray,
                         proposed_valid: np.ndarray, fit_first_geometry_valid: np.ndarray) -> np.ndarray:
    """One shared support for both methods after region membership is defined."""
    return paired_support(
        np.asarray(region, bool) & np.isfinite(reference),
        proposed, fit_first, proposed_valid, fit_first_geometry_valid,
    )


def evaluation_group_ids(case_id: str, all_groups: Mapping[str, np.ndarray], heldout: Mapping[str, list[int]]) -> dict[str, list[int]]:
    """Use all standardized planes for L100, otherwise only manifest held-outs."""
    if case_id == "L100":
        return {stack: [int(value) for value in all_groups[stack]] for stack in STACKS}
    selected = {stack: [int(value) for value in heldout.get(stack, [])] for stack in STACKS}
    if not sum(len(values) for values in selected.values()):
        raise ValueError("sparse formal native-grid evaluation requires at least one manifest held-out plane")
    return selected


def _record(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _published_record(staging_path: Path, published_path: Path) -> dict[str, str]:
    return {"path": str(published_path.resolve()), "sha256": sha256(staging_path)}


def _all_native_groups(native_root: Path, sparse_manifest: Path, stack: str) -> dict[str, Any]:
    from .final_cyj_reporting import _load_all_verified_native_cache_groups
    return _load_all_verified_native_cache_groups(native_root, sparse_manifest, stack)


def _manifest_heldout(payload: Mapping[str, Any]) -> dict[str, list[int]]:
    stacks = payload.get("stacks")
    if not isinstance(stacks, Mapping):
        raise ValueError("sparse manifest requires stack records")
    result: dict[str, list[int]] = {}
    for stack in STACKS:
        item = stacks.get(stack)
        values = item.get("heldout_original_group_ids") if isinstance(item, Mapping) else None
        if not isinstance(values, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError(f"{stack}: heldout_original_group_ids must be an integer list")
        result[stack] = [int(value) for value in values]
    return result


def _case_id(payload: Mapping[str, Any]) -> str:
    value = payload.get("case_id")
    if isinstance(value, str) and value:
        return value
    if payload.get("condition") == "L100":
        return "L100"
    level, fold = payload.get("condition"), payload.get("phase")
    if isinstance(level, str) and isinstance(fold, int):
        return f"{level}_fold{fold}"
    raise ValueError("sparse manifest has no explicit Final5 case identity")


def _aggregate(pooled: Mapping[tuple[str, str, str], dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for (parameter, stack, region), values in pooled.items():
        target = result.setdefault(parameter, {}).setdefault(stack, {})
        proposed = _pooled_metrics(values["reference"], values["proposed"], values["native_total"])
        fit_first = _pooled_metrics(values["reference"], values["fit_first"], values["native_total"])
        target[region] = {"proposed": proposed, "fit_first": fit_first, "paired_deltas": _delta(proposed, fit_first)}
    return result


def build_final_native_grid_metrics(*, proposed_endpoint_root: str | Path, fit_first_endpoint_root: str | Path,
                                    r0_native_maps_root: str | Path, sparse_manifest: str | Path,
                                    output: str | Path, render_qc: bool = True,
                                    manual_outer_masks_root: str | Path | None = None) -> Path:
    """Evaluate immutable central endpoints only; no reconstruction or mask mutation."""
    import nibabel as nib

    output, sparse_manifest, native_root = require_new_output(output), Path(sparse_manifest), Path(r0_native_maps_root)
    sparse = json.loads(sparse_manifest.read_text(encoding="utf-8"))
    subject, condition = sparse.get("subject"), _case_id(sparse)
    if not isinstance(subject, str) or not subject or sparse.get("status") != "ELIGIBLE":
        raise ValueError("formal native-grid evaluator requires an ELIGIBLE named sparse manifest")
    if condition != _case_id(sparse):
        raise ValueError("formal native-grid condition identity mismatch")
    endpoint_paths, endpoint_provenance = validate_sparse_endpoint_roots(proposed_endpoint_root, fit_first_endpoint_root, condition, subject=subject)
    images = {parameter: {name: nib.load(str(path)) for name, path in values.items()} for parameter, values in endpoint_paths.items()}
    effective_records = {parameter: verified_fit_first_effective_query_mask(fit_first_endpoint_root, parameter) for parameter in PARAMETERS}
    sparse_records = {parameter: verified_fit_first_sparse_checkpoint_mask(fit_first_endpoint_root, parameter) for parameter in PARAMETERS}
    effective = {parameter: nib.load(str(record[0])) for parameter, record in effective_records.items()}
    sparse_masks = {parameter: nib.load(str(record[0])) for parameter, record in sparse_records.items()}
    endpoint_provenance = {
        **endpoint_provenance,
        "fit_first_effective_query_masks": {parameter: record for parameter, (_path, record) in effective_records.items()},
        "fit_first_sparse_checkpoint_masks": {parameter: record for parameter, (_path, record) in sparse_records.items()},
    }
    for parameter in PARAMETERS:
        fit = images[parameter]["fit_first"]
        _validate_effective_query_mask(effective[parameter], fit, parameter)
        _validate_sparse_checkpoint_mask(sparse_masks[parameter], parameter)
    endpoint_provenance["mask_geometry"] = {
        parameter: {
            "fit_first_endpoint_volume": _mask_geometry(images[parameter]["fit_first"]),
            "effective_query_mask": _mask_geometry(effective[parameter]),
            "sparse_checkpoint_mask": _mask_geometry(sparse_masks[parameter]),
        }
        for parameter in PARAMETERS
    }
    all_caches = {stack: _all_native_groups(native_root, sparse_manifest, stack) for stack in STACKS}
    all_groups = {stack: np.asarray(all_caches[stack]["group_idx"], int) for stack in STACKS}
    selected = evaluation_group_ids(condition, all_groups, _manifest_heldout(sparse))
    caches: dict[str, dict[str, Any]] = {}
    for stack, ids in selected.items():
        if condition == "L100":
            caches[stack] = all_caches[stack]
        elif ids:
            caches[stack] = select_verified_native_cache_groups(native_root, sparse_manifest, stack, ids)
            if [int(value) for value in caches[stack]["group_idx"]] != ids:
                raise ValueError(f"{stack}: held-out native selection does not match the sparse manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        per_plane: list[dict[str, Any]] = []
        pooled: dict[tuple[str, str, str], dict[str, Any]] = {}
        coverage: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for stack in STACKS:
            for index, group_value in enumerate(caches.get(stack, {}).get("group_idx", [])):
                group = int(group_value); affine = native_ras_affine(caches[stack]["affine_lps_rc"][index])
                plane: dict[str, Any] = {"stack": stack, "original_group_idx": group, "native_affine_ras": affine.tolist(), "parameters": {}}
                for parameter in PARAMETERS:
                    reference = np.asarray(caches[stack][f"{parameter}_ms"][index], float)
                    source = np.asarray(caches[stack]["source_support"][index], bool)
                    sparse = reproject_binary_mask_to_native_plane(sparse_masks[parameter].get_fdata(), sparse_masks[parameter].affine, affine, reference.shape)
                    proposed, pvalid = reproject_volume_to_native_plane(images[parameter]["proposed"].get_fdata(), images[parameter]["proposed"].affine, affine, reference.shape)
                    fit, fgeometry = reproject_volume_to_native_plane(images[parameter]["fit_first"].get_fdata(), images[parameter]["fit_first"].affine, affine, reference.shape)
                    query = reproject_binary_mask_to_native_plane(effective[parameter].get_fdata(), effective[parameter].affine, affine, reference.shape)
                    l100 = l100_region_mask(query)
                    manual_outer, manual_outer_provenance = None, None
                    if manual_outer_masks_root is not None and stack == "sax":
                        from .current_native_central_component_mask import _manual_outer_region
                        manual_outer, manual_outer_provenance = _manual_outer_region(manual_outer_masks_root, stack, group, reference.shape)
                    regions = native_grid_regions(source, reference, l100, sparse, manual_outer_region=manual_outer)
                    item = {"l100_mask_source": "fit_first_query_effective_mask", "l100_native_support": endpoint_provenance["fit_first_effective_query_masks"][parameter], "source_support_equation": SUPPORT_EQUATIONS["native"], "manual_outer_segment": manual_outer_provenance, "slice_region_mask": "manual outer segment AND source_support AND finite(native_reference) AND fit_first_query_effective_mask" if manual_outer is not None else "source_support AND finite(native_reference) AND fit_first_query_effective_mask", "regions": {}, "mask_mapping": "binary masks reprojected with nearest-neighbour order=0", "prediction_mapping": "central endpoint maps reprojected with linear order=1"}
                    mask_arrays = {"source_support": source, "l100_native": l100, "sparse_native": sparse,
                                   "proposed_valid": pvalid, "fit_first_geometry_valid": fgeometry,
                                   "fit_first_query_effective_mask": query, **regions}
                    if manual_outer is not None:
                        mask_arrays["manual_outer_region"] = manual_outer
                    _save_native_masks(staging / "native_masks" / parameter / stack / f"group{group:03d}.npz", affine, **mask_arrays)
                    for region_name in REGIONS:
                        region = regions[region_name]
                        support = exact_paired_support(region, reference, proposed, fit, pvalid, fgeometry)
                        proposed_metrics = metrics_on_exact_support(reference, proposed, support, int(region.sum()))
                        fit_metrics = metrics_on_exact_support(reference, fit, support, int(region.sum()))
                        details = {"region_equation": SUPPORT_EQUATIONS[region_name], "paired_support_equation": COMMON_SUPPORT_EQUATION, "native_region_voxel_count": int(region.sum()), "paired_support_voxel_count": int(support.sum()), "proposed": proposed_metrics, "fit_first": fit_metrics, "paired_deltas": _delta(proposed_metrics, fit_metrics)}
                        if region_name == "recovered_only":
                            details["proposed"].update(recovered_distribution(proposed, support)); details["fit_first"].update(recovered_distribution(fit, support))
                        item["regions"][region_name] = details
                        for pool_name in (stack, "all_stacks"):
                            key = (parameter, pool_name, region_name)
                            target = pooled.setdefault(key, {"reference": [], "proposed": [], "fit_first": [], "native_total": 0})
                            target["native_total"] += int(region.sum())
                            if support.any():
                                target["reference"].append(reference[support]); target["proposed"].append(proposed[support]); target["fit_first"].append(fit[support])
                    coverage.setdefault((parameter, stack), []).append(mask_coverage(regions["native"], regions["l100_full"], regions["sparse_region"]))
                    if render_qc:
                        render_plane_panels(parameter, reference, proposed, fit, exact_paired_support(regions["l100_full"], reference, proposed, fit, pvalid, fgeometry), sparse, staging / "visualization" / parameter / stack / f"group{group:03d}_l100_full_paired.png", title=f"{subject} {condition} {parameter.upper()} {stack} native group {group}; paired L100 sampling support", comparator_label="Fit-first")
                    plane["parameters"][parameter] = item
                per_plane.append(plane)
        aggregate = {"schema": "final5_native_grid_common_support_metrics/v2", "subject": subject, "condition": condition, "endpoint": endpoint_provenance["endpoint"], "regions": _aggregate(pooled), "mask_coverage": {parameter: {stack: pooled_mask_coverage(values) for (name, stack), values in coverage.items() if name == parameter} for parameter in PARAMETERS}, "support_equations": {**SUPPORT_EQUATIONS, "l100_mask": "fit_first_query_effective_mask mapped to the native plane with nearest-neighbour order=0", "slice_region_mask": "manual outer segment AND source_support AND finite(native_reference) AND fit_first_query_effective_mask when manual outer mask is supplied; otherwise historical native support", "paired": COMMON_SUPPORT_EQUATION}, "native_geometry": "original R0 native physical grid; affine_ras = diag([-1,-1,1,1]) @ affine_lps_rc", "l100_full_semantics": "native-plane Fit-first L100 query support restricted by the manual outer segment when supplied; never the myocardium quantitative ROI"}
        aggregate_staging = staging / "aggregate_metrics.json"; write_immutable_json(aggregate_staging, aggregate)
        write_immutable_json(staging / "per_plane_metrics.json", {"schema": "final5_native_grid_per_plane_metrics/v2", "planes": per_plane, "support_equations": aggregate["support_equations"]})
        write_immutable_json(staging / "manifest.json", {"schema": "final5_native_grid_metric_bundle/v2", "subject": subject, "condition": condition, "endpoint": endpoint_provenance["endpoint"], "endpoint_provenance": endpoint_provenance, "l100_mask_source": "fit_first_query_effective_mask", "manual_outer_masks_root": None if manual_outer_masks_root is None else str(Path(manual_outer_masks_root).resolve()), "sparse_manifest": _record(sparse_manifest), "aggregate_metrics": _published_record(aggregate_staging, output / "aggregate_metrics.json"), "support_equations": aggregate["support_equations"], "native_grid": aggregate["native_geometry"], "l100_full_semantics": aggregate["l100_full_semantics"], "myocardial_quantitative_roi": "AHA/myocardium_core_1px is separate and is not used for this slice-region metric", "no_connected_component_filtering": True, "no_training": True, "no_reconstruction": True, "legacy_current_native_mask_used": False})
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("proposed-endpoint-root", "fit-first-endpoint-root", "r0-native-maps-root", "sparse-manifest", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--manual-outer-masks-root")
    parser.add_argument("--no-render-qc", action="store_true")
    args = vars(parser.parse_args()); args["render_qc"] = not args.pop("no_render_qc")
    print(build_final_native_grid_metrics(**args))


if __name__ == "__main__":
    main()
