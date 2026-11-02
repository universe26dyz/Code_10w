"""Build native SAX full/core myocardium masks from contours or filled masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes, distance_transform_edt

from evaluation.utils.common import (dump_json, geometry_equal, load_yaml, output_root,
                                     prepare_output, read_image, validate_config)


def build_masks(labels: np.ndarray, spacing_xy: tuple[float, float], erosion_mm: float) -> dict[str, np.ndarray]:
    blood = np.zeros_like(labels, dtype=bool)
    epi = np.zeros_like(labels, dtype=bool)
    for z in range(labels.shape[0]):
        endo_line = labels[z] == 1
        epi_line = labels[z] == 2
        if endo_line.any() and epi_line.any():
            blood[z] = binary_fill_holes(endo_line)
            epi[z] = binary_fill_holes(epi_line)
    full = epi & ~blood
    core = np.zeros_like(full)
    for z in range(full.shape[0]):
        if full[z].any():
            distance = distance_transform_edt(full[z], sampling=(spacing_xy[1], spacing_xy[0]))
            core[z] = distance > erosion_mm
    return {"blood_pool": blood, "epi_enclosed": epi, "myocardium_full": full,
            "myocardium_core": core, "landmark_anterior": labels == 3,
            "landmark_inferior": labels == 4}


def build_filled_masks(outer: np.ndarray, inner: np.ndarray, points: np.ndarray,
                       spacing_xy: tuple[float, float], erosion_mm: float,
                       anterior_labels: list[int], inferior_labels: list[int]) -> tuple[dict[str, np.ndarray], int]:
    epi = np.asarray(outer) > 0
    blood_raw = np.asarray(inner) > 0
    leakage = int(np.count_nonzero(blood_raw & ~epi))
    blood = blood_raw & epi
    full = epi & ~blood
    core = np.zeros_like(full)
    for z in range(full.shape[0]):
        if full[z].any():
            distance = distance_transform_edt(full[z], sampling=(spacing_xy[1], spacing_xy[0]))
            core[z] = distance > erosion_mm
    return ({"blood_pool": blood, "epi_enclosed": epi, "myocardium_full": full,
             "myocardium_core": core,
             "landmark_anterior": np.isin(points, anterior_labels),
             "landmark_inferior": np.isin(points, inferior_labels)}, leakage)


def run(cfg: dict, overwrite: bool = False) -> Path:
    validate_config(cfg)
    roi_cfg = cfg.get("roi", {})
    labelmap = roi_cfg.get("native_labelmap")
    outer_path, inner_path = roi_cfg.get("outer_filled_mask"), roi_cfg.get("inner_filled_mask")
    manifest = output_root(cfg) / "manifests" / "native_roi_status.json"
    filled_mode = bool(outer_path and inner_path and Path(outer_path).exists() and Path(inner_path).exists())
    contour_mode = bool(labelmap and Path(labelmap).exists())
    if not filled_mode and not contour_mode:
        dump_json({"subject_id": cfg["subject_id"], "status": "SKIPPED",
                   "reason": "myocardial ROI not available"}, manifest, overwrite)
        print("SKIPPED: myocardial ROI not available")
        return manifest
    erosion = float(roi_cfg.get("core_erosion_mm", 1.0))
    containment_leakage = 0
    if filled_mode:
        image, inner_image = read_image(outer_path), read_image(inner_path)
        if not geometry_equal(image, inner_image):
            raise ValueError("Outer and inner filled ROI geometries differ")
        point_path = roi_cfg.get("landmark_labelmap")
        if point_path and Path(point_path).exists():
            point_image = read_image(point_path)
            if not geometry_equal(image, point_image):
                raise ValueError("ROI point-label geometry differs from filled masks")
            points = sitk.GetArrayFromImage(point_image)
        else:
            points = np.zeros_like(sitk.GetArrayFromImage(image))
        outer, inner = sitk.GetArrayFromImage(image), sitk.GetArrayFromImage(inner_image)
        anterior_labels = [int(x) for x in roi_cfg.get("anterior_point_labels", [1, 3, 5])]
        inferior_labels = [int(x) for x in roi_cfg.get("inferior_point_labels", [2, 4, 6])]
        masks, containment_leakage = build_filled_masks(
            outer, inner, points, image.GetSpacing()[:2], erosion,
            anterior_labels, inferior_labels
        )
        labels = points
        source = {"mode": "filled_masks", "outer": str(Path(outer_path).resolve()),
                  "inner": str(Path(inner_path).resolve()),
                  "landmarks": str(Path(point_path).resolve()) if point_path else None,
                  "anterior_point_labels": anterior_labels,
                  "inferior_point_labels": inferior_labels}
    else:
        image = read_image(labelmap)
        labels = sitk.GetArrayFromImage(image)
        allowed = {0, 1, 2, 3, 4}
        unexpected = sorted(set(np.unique(labels).tolist()) - allowed)
        if unexpected:
            raise ValueError(f"Unexpected ROI labels: {unexpected}")
        masks = build_masks(labels, image.GetSpacing()[:2], erosion)
        source = {"mode": "contour_labelmap", "labelmap": str(Path(labelmap).resolve())}
    out_dir = output_root(cfg) / "masks" / "native_sax"
    records = {}
    for name, arr in masks.items():
        path = prepare_output(out_dir / f"{name}.nii.gz", overwrite)
        out = sitk.GetImageFromArray(arr.astype(np.uint8)); out.CopyInformation(image)
        sitk.WriteImage(out, str(path), True)
        records[name] = {"path": str(path), "voxel_count": int(arr.sum())}
    valid = []
    for z in range(labels.shape[0]):
        valid.append({"roi_volume_z_index": z, "has_inner": bool(masks["blood_pool"][z].any()),
                      "has_outer": bool(masks["epi_enclosed"][z].any()),
                      "full_pixels": int(masks["myocardium_full"][z].sum()),
                      "core_pixels": int(masks["myocardium_core"][z].sum()),
                      "anterior_points": int(masks["landmark_anterior"][z].sum()),
                      "inferior_points": int(masks["landmark_inferior"][z].sum())})
    dump_json({"subject_id": cfg["subject_id"], "status": "COMPLETE",
               "source": source, "core_erosion_mm": erosion,
               "inner_outside_outer_voxels_trimmed": containment_leakage,
               "outputs": records, "slices": valid}, manifest, overwrite)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True); p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(); print(run(load_yaml(a.config), a.overwrite))


if __name__ == "__main__": main()
