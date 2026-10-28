"""Paired common-native metrics and QC for completed Proposed/Fit-first volumes.

Both final volumes are sampled onto each immutable acquisition-native physical
plane.  This is the sole geometry used for pair-slice displays and quantitative
ROI masks; the volumes already incorporate their respective optimized poses.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..phase2_clean.contracts import sha256, write_immutable_json
from ..phase2_clean.final_native_grid_metrics import native_ras_affine
from ..phase2_clean.visualization import (native_display_support, prepare_native_comparison_panels,
                                          render_native_plane_comparison)
from .common_native import load_volume_images, sample_common_native_plane
from .cyj11_adapter import load_cyj11_native_sax
from .cyj11_myocardial_roi_metrics import ROI_NAMES, _load_scmr_masks
from .final5_myocardial_roi_metrics import paired_roi_statistics
from .paired_final_pose_lv_metrics import load_outer_filled_masks


SCHEMA = "common_native_paired_metrics_qc/v1"
STACKS = ("sax", "2ch", "4ch")
PARAMETERS = ("t1", "t2")
SUPPORT_EQUATION = "one immutable acquisition-native physical plane AND finite non-zero native foreground AND proposed-volume-geometry-valid AND fit-first-volume-geometry-valid AND finite(proposed, fit-first)"


def _record(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _pooled(values: dict[str, list[np.ndarray]], *, parameter: str,
            ssim_values: dict[str, list[float]]) -> dict[str, Any]:
    reference = np.concatenate(values["reference"]) if values["reference"] else np.asarray([], float)
    proposed = np.concatenate(values["proposed"]) if values["proposed"] else np.asarray([], float)
    fit_first = np.concatenate(values["fit_first"]) if values["fit_first"] else np.asarray([], float)
    result = paired_roi_statistics(reference=reference, proposed=proposed, fit_first=fit_first,
                                   proposed_valid=np.ones(reference.shape, bool), fit_first_valid=np.ones(reference.shape, bool),
                                   roi=np.ones(reference.shape, bool), parameter=parameter)
    result.pop("common_support", None)
    for method in ("proposed", "fit_first"):
        scores = ssim_values[method]
        result[method]["ssim"] = float(np.mean(scores)) if scores else None
        result[method]["ssim_status"] = "mean_of_per_plane_common_support_ssim" if scores else "not_available_2d_support"
    return result


def _volume_arrays(images: dict[str, dict[str, Any]], parameter: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    proposed, fit_first = images[parameter]["proposed"], images[parameter]["fit_first"]
    return (np.asarray(proposed.get_fdata(), float), np.asarray(proposed.affine, float),
            np.asarray(fit_first.get_fdata(), float), np.asarray(fit_first.affine, float))


def _render_roi_qc(*, staging: Path, subject: str, case: str, parameter: str, group: int,
                   reference: np.ndarray, proposed: np.ndarray, fit_first: np.ndarray,
                   display_support: np.ndarray, metric_support: np.ndarray, roi_name: str) -> dict[str, str] | None:
    if not metric_support.any():
        return None
    panels, crop = prepare_native_comparison_panels(reference, proposed, fit_first, display_support)
    r0, r1, c0, c1 = crop
    target = staging / "metric_mask_qc" / roi_name / parameter / f"group{group:03d}" / f"{parameter}_paired_mask.png"
    return render_native_plane_comparison(
        parameter, panels, target,
        title=f"{subject} {case} {parameter.upper()} {roi_name} common metric ROI; group {group}",
        myocardium_contour=np.asarray(metric_support, bool)[r0:r1, c0:c1],
        metadata={"subject": subject, "case": case, "parameter": parameter, "native_group_idx": group,
                  "roi": roi_name, "crop_bounds_rc": crop,
                  "display_support": SUPPORT_EQUATION,
                  "metric_support": "named ROI AND finite(reference) AND both volume geometry-valid and finite"},
    )


def build_common_native_metrics_qc(*, subject: str, case: str, native_maps_root: str | Path,
                                   prepared_subject_root: str | Path, roi_bundle: str | Path,
                                   proposed_t1: str | Path, proposed_t2: str | Path,
                                   fit_first_t1: str | Path, fit_first_t2: str | Path,
                                   output: str | Path) -> Path:
    """Create all-stack common-plane pair QC plus SAX LV/myocardial metrics."""
    native_maps_root, prepared_subject_root, roi_bundle, output = (Path(value) for value in
        (native_maps_root, prepared_subject_root, roi_bundle, output))
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    images = load_volume_images(proposed_t1=str(proposed_t1), proposed_t2=str(proposed_t2),
                                fit_first_t1=str(fit_first_t1), fit_first_t2=str(fit_first_t2))
    native = {stack: load_cyj11_native_sax(native_maps_root / stack, prepared_subject_root / stack / "observations.npz")
              for stack in STACKS}
    sax = native["sax"]
    groups = np.asarray(sax["group_idx"], int)
    shape = tuple(map(int, np.asarray(sax["t1_ms"]).shape[1:]))
    myocardium = _load_scmr_masks(roi_bundle, expected_groups=groups, expected_shape=shape)
    outer_filled = load_outer_filled_masks(roi_bundle, expected_groups=groups, expected_shape=shape)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        pair_outputs: dict[str, Any] = {parameter: {} for parameter in PARAMETERS}
        lv_planes: list[dict[str, Any]] = []
        myo_planes: list[dict[str, Any]] = []
        lv_values = {parameter: {method: [] for method in ("reference", "proposed", "fit_first")} for parameter in PARAMETERS}
        lv_ssim = {parameter: {method: [] for method in ("proposed", "fit_first")} for parameter in PARAMETERS}
        myo_values = {parameter: {roi: {method: [] for method in ("reference", "proposed", "fit_first")}
                                  for roi in ROI_NAMES} for parameter in PARAMETERS}
        myo_ssim = {parameter: {roi: {method: [] for method in ("proposed", "fit_first")}
                                for roi in ROI_NAMES} for parameter in PARAMETERS}

        for stack in STACKS:
            cache = native[stack]
            for index, group_value in enumerate(np.asarray(cache["group_idx"], int)):
                group = int(group_value)
                affine = native_ras_affine(np.asarray(cache["affine_lps_rc"][index], float))
                for parameter in PARAMETERS:
                    pvol, paff, fvol, faff = _volume_arrays(images, parameter)
                    plane = sample_common_native_plane(
                        reference=np.asarray(cache[f"{parameter}_ms"][index], float), native_affine_ras=affine,
                        proposed_volume=pvol, proposed_affine_ras=paff, fit_first_volume=fvol, fit_first_affine_ras=faff,
                    )
                    display = native_display_support(plane["reference"], plane["proposed"], plane["fit_first"],
                                                     plane["proposed_valid"], plane["fit_first_valid"])
                    if display.any():
                        panels, crop = prepare_native_comparison_panels(plane["reference"], plane["proposed"], plane["fit_first"], display)
                        target = staging / "pair_slices" / parameter / stack / f"group{group:03d}" / f"{parameter}_paired_slice.png"
                        pair_outputs[parameter].setdefault(stack, {})[str(group)] = render_native_plane_comparison(
                            parameter, panels, target,
                            title=f"{subject} {case} {parameter.upper()} common-native paired {stack.upper()} group {group}",
                            metadata={"subject": subject, "case": case, "parameter": parameter, "stack": stack,
                                      "native_group_idx": group, "crop_bounds_rc": crop, "support_equation": SUPPORT_EQUATION},
                        )
                    else:
                        pair_outputs[parameter].setdefault(stack, {})[str(group)] = {"status": "EMPTY_COMMON_NATIVE_SUPPORT"}
                    if stack != "sax":
                        continue
                    roi_sets = {"lv_outer_filled": outer_filled[index], **myocardium}
                    metric_reports: dict[str, Any] = {}
                    for roi_name, roi in roi_sets.items():
                        stats = paired_roi_statistics(roi=roi, parameter=parameter, **plane)
                        metric_reports[roi_name] = stats
                        _render_roi_qc(staging=staging, subject=subject, case=case, parameter=parameter, group=group,
                                       reference=plane["reference"], proposed=plane["proposed"], fit_first=plane["fit_first"],
                                       display_support=display, metric_support=stats["common_support"], roi_name=roi_name)
                    lv_stat = metric_reports["lv_outer_filled"]
                    lv_planes.append({"native_group_idx": group, "outer_filled_voxels": int(outer_filled[index].sum()),
                                      "parameters": {parameter: {"common_support_voxels": lv_stat["common_support_voxels"],
                                      "coverage": None if not outer_filled[index].any() else lv_stat["common_support_voxels"] / int(outer_filled[index].sum()),
                                      "statistics": {key: value for key, value in lv_stat.items() if key != "common_support"}}}})
                    for method in ("reference", "proposed", "fit_first"):
                        if lv_stat["common_support"].any():
                            lv_values[parameter][method].append(np.asarray(plane[method])[lv_stat["common_support"]])
                    for method in ("proposed", "fit_first"):
                        if lv_stat[method].get("ssim") is not None:
                            lv_ssim[parameter][method].append(float(lv_stat[method]["ssim"]))
                    myo_planes.append({"native_group_idx": group, "roi_voxels": {name: int(mask[index].sum()) for name, mask in myocardium.items()},
                                       "parameters": {parameter: {name: {"common_support_voxels": stats["common_support_voxels"],
                                       "coverage": None if not myocardium[name][index].any() else stats["common_support_voxels"] / int(myocardium[name][index].sum()),
                                       "statistics": {key: value for key, value in stats.items() if key != "common_support"}}
                                       for name, stats in metric_reports.items() if name in myocardium}}})
                    for name in ROI_NAMES:
                        stats = metric_reports[name]; support = stats["common_support"]
                        for method in ("reference", "proposed", "fit_first"):
                            if support.any():
                                myo_values[parameter][name][method].append(np.asarray(plane[method])[support])
                        for method in ("proposed", "fit_first"):
                            if stats[method].get("ssim") is not None:
                                myo_ssim[parameter][name][method].append(float(stats[method]["ssim"]))

        # Merge the separately appended T1/T2 records by native group.
        def merge(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
            merged: dict[int, dict[str, Any]] = {}
            for item in records:
                group = int(item["native_group_idx"])
                target = merged.setdefault(group, {key: value for key, value in item.items() if key != "parameters"})
                target.setdefault("parameters", {}).update(item["parameters"])
            return [merged[group] for group in sorted(merged)]

        lv_planes, myo_planes = merge(lv_planes), merge(myo_planes)
        lv_pooled = {parameter: _pooled(values, parameter=parameter, ssim_values=lv_ssim[parameter])
                     for parameter, values in lv_values.items()}
        myo_pooled = {parameter: {roi: _pooled(values, parameter=parameter, ssim_values=myo_ssim[parameter][roi])
                                  for roi, values in by_roi.items()}
                       for parameter, by_roi in myo_values.items()}
        provenance = {"native_maps": {stack: {name: _record(native_maps_root / stack / name)
                                                for name in ("native_maps.npz", "geometry.npz", "manifest.json")}
                                      for stack in STACKS},
                      "prepared_subject_root": str(prepared_subject_root.resolve()), "roi_bundle": _record(roi_bundle / "sax_myocardium_masks.npz"),
                      "volumes": {parameter: {method: _record(image.get_filename()) for method, image in items.items()}
                                  for parameter, items in images.items()}}
        base = {"schema": SCHEMA, "subject": subject, "case": case, "sampling": "common_acquisition_native_plane",
                "coordinate_contract": "both final reconstructed volumes are linearly sampled on each immutable native physical RAS plane; no method-specific final output-slice plane is used for pairwise comparison",
                "support_equation": SUPPORT_EQUATION, "input_provenance": provenance,
                "metric_contract": "mean, sample SD, RMSE, NRMSE, bias, MAE, Pearson CC, NCC, fixed-range PSNR, and mean masked 2-D SSIM",
                "psnr_data_ranges_ms": {"t1": 2500.0, "t2": 200.0}, "downstream_only": True, "no_reconstruction": True}
        write_immutable_json(staging / "lv_outer_filled" / "per_plane_statistics.json", {**base, "region": "lv_outer_filled", "planes": lv_planes})
        write_immutable_json(staging / "lv_outer_filled" / "pooled_sax_statistics.json", {**base, "region": "lv_outer_filled", "pooled_sax": lv_pooled})
        write_immutable_json(staging / "myocardial_roi" / "per_plane_statistics.json", {**base, "region": "myocardium", "planes": myo_planes})
        write_immutable_json(staging / "myocardial_roi" / "pooled_sax_statistics.json", {**base, "region": "myocardium", "pooled_sax": myo_pooled})
        write_immutable_json(staging / "manifest.json", {**base, "pair_slices": pair_outputs,
                                  "metric_mask_qc": "outline overlays on Native/Proposed/Fit-first common-native planes"})
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("subject", "case", "native-maps-root", "prepared-subject-root", "roi-bundle",
                 "proposed-t1", "proposed-t2", "fit-first-t1", "fit-first-t2", "output"):
        parser.add_argument("--" + name, required=True)
    print(build_common_native_metrics_qc(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
