"""All-stack common-native LV/myocardial ROI metrics from the SAX mask volume."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..phase2_clean.contracts import sha256, write_immutable_json
from ..phase2_clean.evaluation import reproject_binary_mask_to_native_plane
from ..phase2_clean.final_native_grid_metrics import native_ras_affine
from ..phase2_clean.final_stratified_native_metrics import classify_stratified_groups
from ..phase2_clean.visualization import native_display_support, prepare_native_comparison_panels, render_native_plane_comparison
from .common_native import load_volume_images, sample_common_native_plane
from .common_native_metrics_qc import PARAMETERS, STACKS, SUPPORT_EQUATION, _native_stack_affine, _pooled, _volume_arrays
from .cyj11_adapter import load_cyj11_native_sax
from .cyj11_myocardial_roi_metrics import ROI_NAMES, _load_scmr_masks
from .final5_myocardial_roi_metrics import paired_roi_statistics
from .paired_final_pose_lv_metrics import load_outer_filled_masks


ALL_ROIS = ("lv_outer_filled", *ROI_NAMES)
ALL_SLICES = "all_slices"
HELDOUT_SLICES = "heldout_slices"


def _record(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def stack_sax_masks_as_volume(native_sax: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Represent all manual SAX planes as one physical nearest-neighbour mask volume."""
    affine = _native_stack_affine(native_sax)
    count = len(np.asarray(native_sax["affine_lps_rc"]))
    result: dict[str, np.ndarray] = {}
    for name, mask in masks.items():
        value = np.asarray(mask, bool)
        if value.ndim != 3 or value.shape[0] != count:
            raise ValueError(f"{name}: SAX mask stack does not match SAX geometry")
        result[name] = np.moveaxis(value, 0, 2)
    return result, affine


def project_sax_mask_volume(mask_volume: np.ndarray, mask_affine_ras: np.ndarray, native_affine_ras: np.ndarray,
                            shape_rc: tuple[int, int], *, slice_index: int = 0) -> np.ndarray:
    """Project the discrete SAX mask volume onto any immutable native stack plane."""
    return reproject_binary_mask_to_native_plane(mask_volume, mask_affine_ras, native_affine_ras, shape_rc, slice_index=slice_index)


def _strata_for_plane(stack: str, membership: str) -> tuple[str, ...]:
    values = [ALL_SLICES, f"stack_{stack}"]
    if membership == "held-out":
        values.extend((HELDOUT_SLICES, f"heldout_{stack}"))
    return tuple(values)


def _empty_accumulators() -> dict[str, dict[str, dict[str, Any]]]:
    names = (ALL_SLICES, HELDOUT_SLICES, *(f"stack_{stack}" for stack in STACKS), "heldout_2ch", "heldout_4ch")
    return {
        name: {roi: {"reference": [], "proposed": [], "fit_first": [],
                     "proposed_ssim": [], "fit_first_ssim": [], "mask_voxels": 0}
               for roi in ALL_ROIS}
        for name in names
    }


def _render_mask_qc(*, staging: Path, subject: str, case: str, stack: str, parameter: str, group: int,
                    plane: dict[str, np.ndarray], display_support: np.ndarray, metric_support: np.ndarray,
                    roi_name: str) -> None:
    if not metric_support.any():
        return
    panels, crop = prepare_native_comparison_panels(plane["reference"], plane["proposed"], plane["fit_first"], display_support)
    r0, r1, c0, c1 = crop
    render_native_plane_comparison(
        parameter, panels,
        staging / "metric_mask_qc" / roi_name / parameter / stack / f"group{group:03d}" / f"{parameter}_paired_mask.png",
        title=f"{subject} {case} {parameter.upper()} {roi_name} common metric ROI; {stack.upper()} group {group}",
        myocardium_contour=np.asarray(metric_support, bool)[r0:r1, c0:c1],
        metadata={"subject": subject, "case": case, "parameter": parameter, "stack": stack,
                  "native_group_idx": group, "roi": roi_name, "crop_bounds_rc": crop,
                  "metric_support": "nearest-neighbour SAX mask volume ROI AND finite reference AND both volume geometry-valid and finite"},
    )


def _aggregate(accumulators: dict[str, dict[str, Any]], *, parameter: str, heldout_available: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stratum, by_roi in accumulators.items():
        if stratum == HELDOUT_SLICES and not heldout_available:
            result[stratum] = {roi: {"status": "NOT_APPLICABLE_L100"} for roi in ALL_ROIS}
            continue
        if stratum.startswith("heldout_") and stratum not in (HELDOUT_SLICES,) and not heldout_available:
            result[stratum] = {roi: {"status": "NOT_APPLICABLE_L100"} for roi in ALL_ROIS}
            continue
        result[stratum] = {}
        for roi, values in by_roi.items():
            pooled = _pooled(values, parameter=parameter,
                             ssim_values={method: values[f"{method}_ssim"] for method in ("proposed", "fit_first")})
            result[stratum][roi] = {"mask_voxels": int(values["mask_voxels"]),
                                    "common_support_voxels": int(pooled["reference"]["n"]),
                                    "coverage_fraction": (float(pooled["reference"]["n"] / values["mask_voxels"])
                                                          if values["mask_voxels"] else 0.0),
                                    "statistics": pooled}
    return result


def build_common_native_all_stack_roi_metrics(*, subject: str, case: str, native_maps_root: str | Path,
                                               prepared_subject_root: str | Path, roi_bundle: str | Path,
                                               sparse_manifest: str | Path,
                                               proposed_t1: str | Path, proposed_t2: str | Path,
                                               fit_first_t1: str | Path, fit_first_t2: str | Path,
                                               output: str | Path) -> Path:
    """Evaluate manual SAX LV/myocardium masks on all SAX/2CH/4CH native planes."""
    native_maps_root, prepared_subject_root, roi_bundle, sparse_manifest, output = (Path(value) for value in
        (native_maps_root, prepared_subject_root, roi_bundle, sparse_manifest, output))
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    sparse = json.loads(sparse_manifest.read_text(encoding="utf-8"))
    images = load_volume_images(proposed_t1=str(proposed_t1), proposed_t2=str(proposed_t2),
                                fit_first_t1=str(fit_first_t1), fit_first_t2=str(fit_first_t2))
    native = {stack: load_cyj11_native_sax(native_maps_root / stack, prepared_subject_root / stack / "observations.npz")
              for stack in STACKS}
    sax = native["sax"]
    sax_groups = np.asarray(sax["group_idx"], int)
    sax_shape = tuple(map(int, np.asarray(sax["t1_ms"]).shape[1:]))
    myocardium = _load_scmr_masks(roi_bundle, expected_groups=sax_groups, expected_shape=sax_shape)
    masks = {"lv_outer_filled": load_outer_filled_masks(roi_bundle, expected_groups=sax_groups, expected_shape=sax_shape),
             **myocardium}
    mask_volumes, mask_affine = stack_sax_masks_as_volume(sax, masks)
    source_groups = {stack: [int(group) for group in native[stack]["group_idx"]] for stack in STACKS}
    _regions, statuses = classify_stratified_groups(sparse, source_groups)
    heldout_available = sparse.get("condition") != "L100"

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        np.savez_compressed(staging / "sax_mask_volumes.npz", affine_ras=mask_affine, group_idx=sax_groups,
                            **{name: value.astype(np.uint8) for name, value in mask_volumes.items()})
        accumulators = {parameter: _empty_accumulators() for parameter in PARAMETERS}
        per_plane: list[dict[str, Any]] = []
        for stack in STACKS:
            cache = native[stack]
            for index, group_value in enumerate(np.asarray(cache["group_idx"], int)):
                group = int(group_value)
                membership = statuses[(stack, group)]
                affine = native_ras_affine(np.asarray(cache["affine_lps_rc"][index], float))
                plane_record: dict[str, Any] = {"stack": stack, "native_group_idx": group,
                                                "membership": membership, "parameters": {}}
                for parameter in PARAMETERS:
                    proposed_volume, proposed_affine, fit_volume, fit_affine = _volume_arrays(images, parameter)
                    plane = sample_common_native_plane(
                        reference=np.asarray(cache[f"{parameter}_ms"][index], float), native_affine_ras=affine,
                        proposed_volume=proposed_volume, proposed_affine_ras=proposed_affine,
                        fit_first_volume=fit_volume, fit_first_affine_ras=fit_affine,
                    )
                    display = native_display_support(plane["reference"], plane["proposed"], plane["fit_first"],
                                                     plane["proposed_valid"], plane["fit_first_valid"])
                    roi_records: dict[str, Any] = {}
                    for roi_name in ALL_ROIS:
                        roi = project_sax_mask_volume(mask_volumes[roi_name], mask_affine, affine,
                                                      tuple(plane["reference"].shape))
                        stats = paired_roi_statistics(roi=roi, parameter=parameter, **plane)
                        roi_records[roi_name] = {
                            "mask_voxels": int(roi.sum()), "common_support_voxels": stats["common_support_voxels"],
                            "coverage_fraction": (float(stats["common_support_voxels"] / roi.sum()) if roi.any() else 0.0),
                            "statistics": {key: value for key, value in stats.items() if key != "common_support"},
                        }
                        _render_mask_qc(staging=staging, subject=subject, case=case, stack=stack,
                                        parameter=parameter, group=group, plane=plane,
                                        display_support=display, metric_support=stats["common_support"], roi_name=roi_name)
                        for stratum in _strata_for_plane(stack, membership):
                            target = accumulators[parameter][stratum][roi_name]
                            target["mask_voxels"] += int(roi.sum())
                            if stats["common_support"].any():
                                support = stats["common_support"]
                                for method in ("reference", "proposed", "fit_first"):
                                    target[method].append(np.asarray(plane[method])[support])
                                for method in ("proposed", "fit_first"):
                                    if stats[method].get("ssim") is not None:
                                        target[f"{method}_ssim"].append(float(stats[method]["ssim"]))
                    plane_record["parameters"][parameter] = roi_records
                per_plane.append(plane_record)
        aggregate = {parameter: _aggregate(accumulators[parameter], parameter=parameter,
                                            heldout_available=heldout_available)
                     for parameter in PARAMETERS}
        write_immutable_json(staging / "per_plane_statistics.json", {
            "schema": "common_native_all_stack_roi_per_plane/v1", "subject": subject, "case": case,
            "planes": per_plane,
        })
        write_immutable_json(staging / "aggregate_statistics.json", {
            "schema": "common_native_all_stack_roi_aggregate/v1", "subject": subject, "case": case,
            "aggregates": aggregate,
        })
        write_immutable_json(staging / "manifest.json", {
            "schema": "common_native_all_stack_roi_metrics/v1", "subject": subject, "case": case,
            "sampling": "common_acquisition_native_plane", "mask_volume": "all hand-drawn SAX masks stacked on the regular SAX physical lattice and nearest-neighbour projected to each target native plane",
            "support_equation": "nearest-neighbour projected SAX ROI AND finite(native reference) AND proposed-volume-geometry-valid AND fit-first-volume-geometry-valid AND finite(proposed, fit-first)",
            "strata": {"all_slices": "all SAX, 2CH and 4CH source planes", "heldout_slices": "all manifest-held-out 2CH/4CH planes; not applicable for L100", "per_stack": "all source planes separately for sax, 2ch and 4ch", "per_stack_heldout": "manifest-held-out planes separately for 2ch and 4ch"},
            "input_provenance": {"native_maps": {stack: {name: _record(native_maps_root / stack / name)
                                                              for name in ("native_maps.npz", "geometry.npz", "manifest.json")}
                                                   for stack in STACKS},
                                 "sparse_manifest": _record(sparse_manifest), "roi_bundle": _record(roi_bundle / "sax_myocardium_masks.npz"),
                                 "volumes": {parameter: {method: _record(image.get_filename()) for method, image in items.items()}
                                             for parameter, items in images.items()}},
            "metric_contract": "mean, sample SD, RMSE, NRMSE, bias, MAE, Pearson CC, NCC, fixed-range PSNR, and mean masked 2-D SSIM",
            "psnr_data_ranges_ms": {"t1": 2500.0, "t2": 200.0}, "metric_mask_qc": "all ROI supports are overlaid on Native/Proposed/Fit-first for every stack/plane",
            "no_reconstruction": True,
        })
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("subject", "case", "native-maps-root", "prepared-subject-root", "roi-bundle", "sparse-manifest",
                 "proposed-t1", "proposed-t2", "fit-first-t1", "fit-first-t2", "output"):
        parser.add_argument("--" + name, required=True)
    print(build_common_native_all_stack_roi_metrics(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
