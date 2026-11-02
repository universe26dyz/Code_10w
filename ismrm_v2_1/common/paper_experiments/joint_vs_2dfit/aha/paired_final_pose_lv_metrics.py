"""Final-pose paired metrics inside the manually drawn filled LV outer contour.

This is deliberately separate from myocardial ROI reporting: ``outer_filled``
contains both myocardium and the enclosed LV blood pool.  It consumes native
final-pose bundles, never endpoint-volume reprojections.
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
from .cyj11_adapter import load_cyj11_native_sax
from .cyj11_runner import final_pose_sax_plane, load_cyj11_final_pose_bundles
from .final5_myocardial_roi_metrics import paired_roi_statistics


SCHEMA = "paired_final_pose_lv_outer_filled_metrics/v1"
SUPPORT_EQUATION = "manual_outer_filled (epicardial contour plus enclosed LV) AND finite(reference) AND proposed-final-pose-valid AND fit-first-final-pose-valid AND finite(proposed, fit-first)"


def _record(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def load_outer_filled_masks(bundle_root: str | Path, *, expected_groups: np.ndarray,
                            expected_shape: tuple[int, int]) -> np.ndarray:
    """Load the exact native-grid filled epicardial/LV masks without resampling."""
    archive_path = Path(bundle_root) / "sax_myocardium_masks.npz"
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    with np.load(archive_path, allow_pickle=False) as archive:
        if "group_idx" not in archive or "outer_filled" not in archive:
            raise ValueError("LV metrics require group_idx and outer_filled in sax_myocardium_masks.npz")
        groups = np.asarray(archive["group_idx"], int)
        masks = np.asarray(archive["outer_filled"], bool)
    expected = (len(expected_groups), *expected_shape)
    if not np.array_equal(groups, np.asarray(expected_groups, int)):
        raise ValueError("outer_filled group identity mismatches native SAX cache")
    if masks.shape != expected:
        raise ValueError(f"outer_filled shape {masks.shape} mismatches native SAX {expected}")
    return masks


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
        values_for_method = ssim_values[method]
        result[method]["ssim"] = float(np.mean(values_for_method)) if values_for_method else None
        result[method]["ssim_status"] = ("mean_of_per_plane_common_support_ssim" if values_for_method
                                          else "not_available_2d_support")
    return result


def build_paired_final_pose_lv_metrics(*, subject: str, case: str, native_sax_root: str | Path,
                                       prepared_sax: str | Path, outer_mask_bundle: str | Path,
                                       proposed_bundle: str | Path, fit_first_bundle: str | Path,
                                       output: str | Path) -> Path:
    """Write per-plane and pooled complete metrics on the filled LV outer contour."""
    native_sax_root, prepared_sax, outer_mask_bundle = (Path(value) for value in
                                                        (native_sax_root, prepared_sax, outer_mask_bundle))
    proposed_bundle, fit_first_bundle, output = (Path(value) for value in
                                                 (proposed_bundle, fit_first_bundle, output))
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    native = load_cyj11_native_sax(native_sax_root, prepared_sax)
    groups = np.asarray(native["group_idx"], int)
    shape = tuple(map(int, np.asarray(native["t1_ms"]).shape[1:]))
    outer_filled = load_outer_filled_masks(outer_mask_bundle, expected_groups=groups, expected_shape=shape)
    bundles = load_cyj11_final_pose_bundles(proposed_bundle, fit_first_bundle, subject=subject, case=case)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        reports: list[dict[str, Any]] = []
        pooled = {parameter: {method: [] for method in ("reference", "proposed", "fit_first")}
                  for parameter in ("t1", "t2")}
        pooled_ssim = {parameter: {method: [] for method in ("proposed", "fit_first")}
                       for parameter in ("t1", "t2")}
        for index, group in enumerate(groups):
            report: dict[str, Any] = {"native_group_idx": int(group), "outer_filled_voxels": int(outer_filled[index].sum()),
                                      "parameters": {}}
            for parameter in ("t1", "t2"):
                plane = final_pose_sax_plane(native=native, group=int(group), parameter=parameter, **bundles)
                stats = paired_roi_statistics(roi=outer_filled[index], parameter=parameter, **plane)
                report["parameters"][parameter] = {
                    "common_support_voxels": stats["common_support_voxels"],
                    "coverage": None if not outer_filled[index].any() else stats["common_support_voxels"] / int(outer_filled[index].sum()),
                    "statistics": {key: value for key, value in stats.items() if key != "common_support"},
                }
                support = stats["common_support"]
                if support.any():
                    for method in ("reference", "proposed", "fit_first"):
                        pooled[parameter][method].append(np.asarray(plane[method], float)[support])
                for method in ("proposed", "fit_first"):
                    value = stats[method].get("ssim")
                    if value is not None:
                        pooled_ssim[parameter][method].append(float(value))
            reports.append(report)
        pooled_report = {parameter: _pooled(values, parameter=parameter, ssim_values=pooled_ssim[parameter])
                         for parameter, values in pooled.items()}
        base = {"schema": SCHEMA, "subject": subject, "case": case, "sampling": "final_pose",
                "region": "lv_outer_filled", "support_equation": SUPPORT_EQUATION,
                "metric_contract": "mean, sample SD, RMSE, NRMSE, bias, MAE, Pearson CC, NCC, fixed-range PSNR, and mean masked 2-D SSIM",
                "psnr_data_ranges_ms": {"t1": 2500.0, "t2": 200.0},
                "outer_filled_policy": "manual epicardial contour filled to include myocardium and enclosed LV; empty manual planes are retained with zero support",
                "native_sax": {name: _record(native_sax_root / name) for name in ("native_maps.npz", "geometry.npz", "manifest.json")},
                "prepared_sax": _record(prepared_sax), "outer_mask_archive": _record(outer_mask_bundle / "sax_myocardium_masks.npz"),
                "proposed_final_pose_bundle": _record(proposed_bundle / "manifest.json"),
                "fit_first_final_pose_bundle": _record(fit_first_bundle / "manifest.json"),
                "downstream_only": True, "no_reconstruction": True, "immutable_output": True}
        write_immutable_json(staging / "per_plane_statistics.json", {**base, "planes": reports})
        write_immutable_json(staging / "pooled_sax_statistics.json", {**base, "pooled_sax": pooled_report})
        write_immutable_json(staging / "manifest.json", base)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("subject", "case", "native-sax-root", "prepared-sax", "outer-mask-bundle", "proposed-bundle", "fit-first-bundle", "output"):
        parser.add_argument("--" + name, required=True)
    args = vars(parser.parse_args())
    print(build_paired_final_pose_lv_metrics(
        subject=args["subject"], case=args["case"], native_sax_root=args["native_sax_root"],
        prepared_sax=args["prepared_sax"], outer_mask_bundle=args["outer_mask_bundle"],
        proposed_bundle=args["proposed_bundle"], fit_first_bundle=args["fit_first_bundle"], output=args["output"],
    ))


if __name__ == "__main__":
    main()
