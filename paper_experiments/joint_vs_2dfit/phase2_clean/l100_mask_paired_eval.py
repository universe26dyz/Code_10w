"""Read-only L100-sampling-mask held-out evaluation for a sparse R1 run.

This is deliberately separate from the formal evaluator.  It compares the
existing Proposed volume and the existing fixed-L100-mask K128 fit-first
volume on native planes selected by the immutable sparse manifest.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import sha256, write_immutable_json
from .evaluation import _metrics, _world_sample, reproject_volume_to_native_plane
from .fit_first import select_verified_native_cache_groups
from .heldout import _heldout_ids
from .visualization import DISPLAY_RANGES_MS, RESIDUAL_RANGES_MS, colorbars


REGIONS = ("l100_full", "recovered_only", "sparse_region")
PARAMETERS = {"t1": "t1_ms", "t2": "t2_ms"}
SUPPORT_EQUATIONS = {
    "native": "source_support AND finite(native_reference)",
    "l100_full": "S_native AND M_l100_native",
    "recovered_only": "S_native AND M_l100_native AND NOT M_sparse_native",
    "sparse_region": "S_native AND M_sparse_native",
    "paired": "region AND proposed_reprojection_valid AND fitfirst_reprojection_valid AND finite(proposed) AND finite(fitfirst)",
}


def require_new_output(output: str | Path) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    return output


def _record(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def resolve_formal_proposed_volumes(
    r1_real_root: str | Path, r1_run_id: str, subject: str, condition: str,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Resolve exactly the paths passed by the formal sparse held-out wrapper."""
    root = Path(r1_real_root) / r1_run_id / "proposed" / subject / condition
    paths = {"t1": root / "T1_3D.nii.gz", "t2": root / "T2_3D.nii.gz"}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"formal Proposed quantitative export is missing: {path}")
    return paths, {
        "selection": "matching_quantitative_export",
        "selection_basis": (
            "exact T1_3D.nii.gz/T2_3D.nii.gz paths supplied to the formal held-out "
            "evaluator by phase2_clean/server_commands/evaluate_real_sparse_pilot.sh"
        ),
        "formal_evaluation_wrapper": "phase2_clean/server_commands/evaluate_real_sparse_pilot.sh",
        "resolved_root": str(root.resolve()),
    }


def reproject_binary_mask_to_native_plane(
    mask: np.ndarray, mask_affine: np.ndarray, native_affine: np.ndarray, plane_shape: tuple[int, int],
) -> np.ndarray:
    """Nearest-neighbour world reprojection for binary sampling-mask membership."""
    row, col = np.indices(plane_shape, dtype=float)
    indices = np.vstack((row.ravel(), col.ravel(), np.zeros(row.size), np.ones(row.size)))
    world = np.asarray(native_affine, float) @ indices
    values, inside = _world_sample(np.asarray(mask, np.uint8), mask_affine, world.reshape(4, *plane_shape), order=0)
    return np.asarray(inside, bool) & np.isfinite(values) & (values > .5)


def native_support_regions(
    source_support: np.ndarray, native_reference: np.ndarray, l100_mask: np.ndarray, sparse_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """The three mask-defined domains; never infer membership from volume FOV."""
    arrays = (source_support, native_reference, l100_mask, sparse_mask)
    if len({np.asarray(value).shape for value in arrays}) != 1:
        raise ValueError("native support arrays must share one plane shape")
    native = np.asarray(source_support, bool) & np.isfinite(native_reference)
    l100 = native & np.asarray(l100_mask, bool)
    sparse = native & np.asarray(sparse_mask, bool)
    return {"native": native, "l100_full": l100, "recovered_only": l100 & ~np.asarray(sparse_mask, bool), "sparse_region": sparse}


def paired_support(region: np.ndarray, proposed: np.ndarray, fit_first: np.ndarray, proposed_valid: np.ndarray, fit_first_valid: np.ndarray) -> np.ndarray:
    arrays = (region, proposed, fit_first, proposed_valid, fit_first_valid)
    if len({np.asarray(value).shape for value in arrays}) != 1:
        raise ValueError("paired native arrays must share one plane shape")
    return (np.asarray(region, bool) & np.asarray(proposed_valid, bool) & np.asarray(fit_first_valid, bool)
            & np.isfinite(proposed) & np.isfinite(fit_first))


def _empty_metrics() -> dict[str, Any]:
    return {"voxel_count": 0, "rmse": None, "mae": None, "bias": None, "reference_mean_ms": None,
            "prediction_mean_ms": None, "pearson_r": None, "ncc": None, "range_nrmse": None}


def metrics_on_exact_support(reference: np.ndarray, prediction: np.ndarray, support: np.ndarray, native_region_count: int) -> dict[str, Any]:
    support = np.asarray(support, bool)
    if not support.any():
        result = _empty_metrics()
    else:
        canonical = _metrics(reference, prediction, support)
        result = {
            "voxel_count": int(canonical["support_voxels"]), "rmse": canonical["rmse"], "mae": canonical["mae"],
            "bias": canonical["bias"], "reference_mean_ms": canonical["reference_mean_ms"],
            "prediction_mean_ms": canonical["prediction_mean_ms"], "pearson_r": canonical["pearson_r"],
            "ncc": canonical["ncc"], "range_nrmse": canonical["relative_nrmse"],
        }
    result["coverage_fraction"] = float(result["voxel_count"] / native_region_count) if native_region_count else 0.0
    result["native_region_voxel_count"] = int(native_region_count)
    return result


def recovered_distribution(prediction: np.ndarray, support: np.ndarray) -> dict[str, Any]:
    values = np.asarray(prediction, float)[np.asarray(support, bool)]
    if not values.size:
        return {"p5_prediction_ms": None, "median_prediction_ms": None, "p95_prediction_ms": None, "zero_fraction": None}
    return {"p5_prediction_ms": float(np.percentile(values, 5)), "median_prediction_ms": float(np.median(values)),
            "p95_prediction_ms": float(np.percentile(values, 95)), "zero_fraction": float(np.mean(values == 0))}


def mask_coverage(native: np.ndarray, l100: np.ndarray, sparse: np.ndarray) -> dict[str, Any]:
    native_count, l100_count, sparse_count = (int(np.count_nonzero(value)) for value in (native, l100, sparse))
    recovered = np.asarray(l100, bool) & ~np.asarray(sparse, bool)
    sparse_inside_l100 = np.asarray(sparse, bool) & np.asarray(l100, bool)
    return {"native_source_voxels": native_count, "l100_native_mask_voxels_within_source": l100_count,
            "sparse_native_mask_voxels_within_source": sparse_count, "recovered_only_voxels": int(recovered.sum()),
            "sparse_fraction_of_l100": float(sparse_inside_l100.sum() / l100_count) if l100_count else 0.0,
            "l100_coverage_fraction_of_native": float(l100_count / native_count) if native_count else 0.0,
            "sparse_not_l100_boundary_discrepancy_voxels": int((np.asarray(sparse, bool) & ~np.asarray(l100, bool)).sum())}


def _pooled_metrics(reference: list[np.ndarray], prediction: list[np.ndarray], native_total: int) -> dict[str, Any]:
    if not reference:
        return metrics_on_exact_support(np.empty(0), np.empty(0), np.empty(0, bool), native_total)
    ref, pred = np.concatenate(reference), np.concatenate(prediction)
    return metrics_on_exact_support(ref, pred, np.ones(ref.size, bool), native_total)


def _delta(proposed: dict[str, Any], fit_first: dict[str, Any]) -> dict[str, float | None]:
    return {f"delta_{key}_proposed_minus_fit_first": None if proposed[key] is None or fit_first[key] is None else float(proposed[key] - fit_first[key]) for key in ("rmse", "mae", "bias")}


def _save_native_masks(path: Path, native_affine_ras: np.ndarray, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, native_affine_ras=np.asarray(native_affine_ras, float), **{key: np.asarray(value, bool) for key, value in arrays.items()})


def pooled_mask_coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool native mask counts across planes; never average plane fractions."""
    names = ("native_source_voxels", "l100_native_mask_voxels_within_source", "sparse_native_mask_voxels_within_source", "recovered_only_voxels", "sparse_not_l100_boundary_discrepancy_voxels")
    result = {name: int(sum(int(record[name]) for record in records)) for name in names}
    l100 = result["l100_native_mask_voxels_within_source"]
    native = result["native_source_voxels"]
    sparse_inside_l100 = l100 - result["recovered_only_voxels"]
    result["sparse_fraction_of_l100"] = float(sparse_inside_l100 / l100) if l100 else 0.0
    result["l100_coverage_fraction_of_native"] = float(l100 / native) if native else 0.0
    return result


def _masked(values: np.ndarray, support: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_where(~np.asarray(support, bool), np.asarray(values, float))


def render_plane_panels(parameter: str, reference: np.ndarray, proposed: np.ndarray, fit_first: np.ndarray,
                        display_mask: np.ndarray, sparse_mask: np.ndarray, output: Path, *, title: str) -> None:
    """Render un-cropped same-grid panels; the sparse contour is an overlay only."""
    import matplotlib.pyplot as plt
    parameter_upper = parameter.upper()
    map_range, residual_range = DISPLAY_RANGES_MS[parameter_upper], RESIDUAL_RANGES_MS[parameter_upper]
    cmap = colorbars().mapping[parameter_upper]
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 5, figsize=(14, 3.1))
    entries = (("Native reference", reference, cmap, map_range), ("Proposed", proposed, cmap, map_range),
               ("Fit-first fixed-L100 K128", fit_first, cmap, map_range),
               ("|Proposed − Native|", np.abs(proposed-reference), "magma", residual_range),
               ("|Fit-first − Native|", np.abs(fit_first-reference), "magma", residual_range))
    for axis, (label, values, panel_cmap, limits) in zip(axes, entries):
        axis.imshow(_masked(values, display_mask), cmap=panel_cmap, vmin=limits[0], vmax=limits[1], origin="lower", interpolation="nearest", aspect="equal")
        axis.contour(np.asarray(sparse_mask, bool).astype(float), levels=(.5,), colors="white", linewidths=.55, linestyles="dashed", origin="lower")
        axis.set_title(label, fontsize=8); axis.axis("off")
    figure.suptitle(title, fontsize=10); figure.savefig(output, dpi=320, bbox_inches="tight"); plt.close(figure)


def render_summary_triptych(parameter: str, reference: np.ndarray, proposed: np.ndarray, fit_first: np.ndarray,
                            display_mask: np.ndarray, sparse_mask: np.ndarray, output: Path, *, title: str) -> None:
    """Representative un-cropped map summary on the same L100-defined domain."""
    import matplotlib.pyplot as plt
    parameter_upper = parameter.upper(); limits = DISPLAY_RANGES_MS[parameter_upper]; cmap = colorbars().mapping[parameter_upper]
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(8.3, 3.0))
    for axis, label, values in zip(axes, ("Native reference", "Proposed", "Fit-first fixed-L100 K128"), (reference, proposed, fit_first)):
        axis.imshow(_masked(values, display_mask), cmap=cmap, vmin=limits[0], vmax=limits[1], origin="lower", interpolation="nearest", aspect="equal")
        axis.contour(np.asarray(sparse_mask, bool).astype(float), levels=(.5,), colors="white", linewidths=.55, linestyles="dashed", origin="lower")
        axis.set_title(label, fontsize=8); axis.axis("off")
    figure.suptitle(title, fontsize=10); figure.savefig(output, dpi=320, bbox_inches="tight"); plt.close(figure)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_l100_mask_paired_evaluation(*, subject: str, condition: str, r0_run_id: str, r1_run_id: str,
                                    sparse_manifest: str | Path, r0_native_maps_root: str | Path,
                                    r1_real_root: str | Path, mask_audit_root: str | Path, output: str | Path) -> Path:
    """Evaluate existing volumes only, atomically publishing a new audit directory."""
    import nibabel as nib

    output = require_new_output(output)
    manifest_path, cache_root, audit_root = Path(sparse_manifest), Path(r0_native_maps_root), Path(mask_audit_root)
    sparse_manifest_payload = json.loads(manifest_path.read_text())
    if sparse_manifest_payload.get("subject") != subject or sparse_manifest_payload.get("condition") != condition.split("_")[0]:
        raise ValueError("subject/condition do not match the sparse manifest")
    proposed_paths, proposed_selection = resolve_formal_proposed_volumes(r1_real_root, r1_run_id, subject, condition)
    paths: dict[str, dict[str, Path]] = {}
    for parameter in PARAMETERS:
        paths[parameter] = {
            "l100_mask": audit_root / parameter / "k1_reference_grid" / "l100_effective_mask_1mm.nii.gz",
            "sparse_mask": audit_root / parameter / "k1_reference_grid" / "sparse_mask_on_l100_grid.nii.gz",
            "fit_first": audit_root / parameter / "l100_mask" / "volume_ms.nii.gz", "proposed": proposed_paths[parameter],
        }
        for path in paths[parameter].values():
            if not path.is_file(): raise FileNotFoundError(path)
    images = {parameter: {name: nib.load(str(path)) for name, path in values.items()} for parameter, values in paths.items()}
    selected = {stack: select_verified_native_cache_groups(cache_root, manifest_path, stack, _heldout_ids(sparse_manifest_payload, stack)) for stack in ("2ch", "4ch")}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    pooled: dict[tuple[str, str, str], dict[str, Any]] = {}
    coverage_pooled: dict[tuple[str, str], list[dict[str, Any]]] = {}
    representative_rendered: set[str] = set()
    try:
        per_plane: list[dict[str, Any]] = []
        for parameter, native_key in PARAMETERS.items():
            for stack, cache in selected.items():
                for index, group in enumerate(cache["group_idx"]):
                    affine_ras = np.diag((-1., -1., 1., 1.)) @ np.asarray(cache["affine_lps_rc"][index], float)
                    reference = np.asarray(cache[native_key][index], float); source = np.asarray(cache["source_support"][index], bool)
                    l100 = reproject_binary_mask_to_native_plane(images[parameter]["l100_mask"].get_fdata(), images[parameter]["l100_mask"].affine, affine_ras, reference.shape)
                    sparse = reproject_binary_mask_to_native_plane(images[parameter]["sparse_mask"].get_fdata(), images[parameter]["sparse_mask"].affine, affine_ras, reference.shape)
                    proposed, p_valid = reproject_volume_to_native_plane(images[parameter]["proposed"].get_fdata(), images[parameter]["proposed"].affine, affine_ras, reference.shape)
                    fit_first, f_valid = reproject_volume_to_native_plane(images[parameter]["fit_first"].get_fdata(), images[parameter]["fit_first"].affine, affine_ras, reference.shape)
                    regions = native_support_regions(source, reference, l100, sparse)
                    plane = {"parameter": parameter, "stack": stack, "original_group_idx": int(group), "coverage": mask_coverage(regions["native"], regions["l100_full"], regions["sparse_region"]), "regions": {}}
                    _save_native_masks(staging / "masks_native" / parameter / f"{stack}_group{int(group)}.npz", affine_ras, l100_native=l100, sparse_native=sparse, native_source=regions["native"], l100_full=regions["l100_full"], recovered_only=regions["recovered_only"], sparse_region=regions["sparse_region"])
                    for pool_name in (stack, "all_lax"):
                        coverage_pooled.setdefault((parameter, pool_name), []).append(plane["coverage"])
                    for region_name in REGIONS:
                        region = regions[region_name]; paired = paired_support(region, proposed, fit_first, p_valid, f_valid)
                        proposed_metrics = metrics_on_exact_support(reference, proposed, paired, int(region.sum()))
                        fit_metrics = metrics_on_exact_support(reference, fit_first, paired, int(region.sum()))
                        details = {"region_equation": SUPPORT_EQUATIONS[region_name], "paired_support_equation": SUPPORT_EQUATIONS["paired"], "native_region_voxel_count": int(region.sum()), "paired_support_voxel_count": int(paired.sum()), "proposed": proposed_metrics, "fit_first": fit_metrics, "paired_deltas": _delta(proposed_metrics, fit_metrics)}
                        if region_name == "recovered_only":
                            details["proposed"].update(recovered_distribution(proposed, paired)); details["fit_first"].update(recovered_distribution(fit_first, paired))
                        plane["regions"][region_name] = details
                        for pool_name in (stack, "all_lax"):
                            key = (parameter, pool_name, region_name)
                            item = pooled.setdefault(key, {"reference": [], "proposed": [], "fit_first": [], "native_total": 0})
                            item["native_total"] += int(region.sum())
                            if paired.any():
                                item["reference"].append(reference[paired]); item["proposed"].append(proposed[paired]); item["fit_first"].append(fit_first[paired])
                    display = regions["l100_full"]
                    root = staging / "visualization" / parameter
                    stem = f"{stack}_group{int(group)}"
                    render_plane_panels(parameter, reference, proposed, fit_first, display, sparse, root / f"{stem}_l100_full.png", title=f"{subject} {condition} {parameter.upper()} {stack} group {int(group)}; L100 sampling domain")
                    render_plane_panels(parameter, reference, proposed, fit_first, regions["recovered_only"], sparse, root / f"{stem}_recovered_only.png", title=f"{subject} {condition} {parameter.upper()} {stack} group {int(group)}; L100 minus sparse")
                    if parameter not in representative_rendered:
                        render_summary_triptych(parameter, reference, proposed, fit_first, display, sparse, root / "summary_representative_l100_full.png", title=f"{subject} {condition} {parameter.upper()} representative {stack} group {int(group)}; L100 sampling domain")
                        render_summary_triptych(parameter, reference, proposed, fit_first, regions["recovered_only"], sparse, root / "summary_representative_recovered_only.png", title=f"{subject} {condition} {parameter.upper()} representative {stack} group {int(group)}; L100 minus sparse")
                        representative_rendered.add(parameter)
                    per_plane.append(plane)
        aggregate: dict[str, Any] = {"statistical_unit": "subject; voxel-pooled held-out native-plane values, not per-plane means", "regions": {}, "mask_coverage": {}}
        for (parameter, stack, region), item in pooled.items():
            section = aggregate["regions"].setdefault(parameter, {}).setdefault(stack, {})
            proposed_metrics = _pooled_metrics(item["reference"], item["proposed"], item["native_total"])
            fit_metrics = _pooled_metrics(item["reference"], item["fit_first"], item["native_total"])
            section[region] = {"proposed": proposed_metrics, "fit_first": fit_metrics, "paired_deltas": _delta(proposed_metrics, fit_metrics)}
        for (parameter, stack), values in coverage_pooled.items():
            aggregate["mask_coverage"].setdefault(parameter, {})[stack] = pooled_mask_coverage(values)
        summary_rows = []
        for parameter, stacks in aggregate["regions"].items():
            for stack, regions in stacks.items():
                for region, metrics in regions.items():
                    for method in ("proposed", "fit_first"):
                        summary_rows.append({"scope": "subject_voxel_pooled", "parameter": parameter, "stack": stack, "region": region, "method": method, **metrics[method]})
        with (staging / "summary.csv").open("w", newline="") as handle:
            fields = sorted({key for row in summary_rows for key in row}); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(summary_rows)
        write_immutable_json(staging / "per_plane_metrics.json", {"subject": subject, "condition": condition, "planes": per_plane, "support_equations": SUPPORT_EQUATIONS})
        write_immutable_json(staging / "aggregate_metrics.json", aggregate)
        audit_manifest = {"git_sha": _git_sha(), "subject": subject, "condition": condition, "r0_run_id": r0_run_id, "r1_run_id": r1_run_id,
                          "native_reference": {stack: cache["provenance"] for stack, cache in selected.items()}, "sparse_manifest": _record(manifest_path),
                          "proposed": {parameter: _record(paths[parameter]["proposed"]) for parameter in PARAMETERS}, "proposed_selection": proposed_selection,
                          "fit_first_fixed_l100_k128": {parameter: _record(paths[parameter]["fit_first"]) for parameter in PARAMETERS},
                          "sampling_masks": {parameter: {name: _record(paths[parameter][name]) for name in ("l100_mask", "sparse_mask")} for parameter in PARAMETERS},
                          "mask_interpolation": "nearest-neighbor world-coordinate", "prediction_interpolation": "linear world-coordinate", "support_equations": SUPPORT_EQUATIONS,
                          "no_training": True, "no_refitting": True, "no_reconstruction": True, "k1_diagnostic_used_for_quantitative_comparison": False,
                          "boundary_discrepancy_policy": "record sparse-mask membership outside L100 after nearest projection; do not modify masks"}
        write_immutable_json(staging / "manifest.json", audit_manifest)
        output_exists = output.exists()
        if output_exists: raise FileExistsError(f"refusing overwrite: {output}")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True); raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True); parser.add_argument("--condition", required=True)
    parser.add_argument("--r0-run-id", required=True); parser.add_argument("--r1-run-id", required=True)
    parser.add_argument("--sparse-manifest", required=True); parser.add_argument("--r0-native-maps-root", required=True)
    parser.add_argument("--r1-real-root", required=True); parser.add_argument("--mask-audit-root", required=True); parser.add_argument("--output", required=True)
    print(run_l100_mask_paired_evaluation(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
