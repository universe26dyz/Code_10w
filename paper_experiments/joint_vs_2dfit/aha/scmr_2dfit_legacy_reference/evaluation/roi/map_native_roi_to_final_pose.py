"""Map native SAX ROI membership to independent T1 and T2 final poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from evaluation.utils.common import (PARAMETERS, apply_orientation, array2d, dump_json,
                                     load_yaml, output_root, read_csv, read_image,
                                     save_array_like, sha256_array, validate_config)


ROI_NAMES = ("blood_pool", "epi_enclosed", "myocardium_full", "myocardium_core",
             "landmark_anterior", "landmark_inferior")


def run(cfg: dict, overwrite: bool = False) -> Path:
    validate_config(cfg)
    status_path = output_root(cfg) / "manifests" / "native_roi_status.json"
    manifest_path = output_root(cfg) / "manifests" / "roi_final_pose_manifest.json"
    if not status_path.exists() or json.loads(status_path.read_text()).get("status") != "COMPLETE":
        dump_json({"subject_id": cfg["subject_id"], "status": "SKIPPED",
                   "reason": "myocardial ROI not available"}, manifest_path, overwrite)
        print("SKIPPED: myocardial ROI not available")
        return manifest_path
    native = {}
    for name in ROI_NAMES:
        path = output_root(cfg) / "masks" / "native_sax" / f"{name}.nii.gz"
        native[name] = read_image(path)
    rows = [r for r in read_csv(output_root(cfg) / "correspondence" / "slice_correspondence.csv")
            if r["stack"] == "sax"]
    records = []
    for row in rows:
        native_id = int(row["native_slice_id"])
        for parameter in PARAMETERS:
            key = parameter.lower(); output_id = int(row[f"{key}_output_id"])
            orientation = row[f"{key}_orientation_mapping"]
            reference = read_image(row[f"{key}_registered_path"])
            native_reference = read_image(row[f"{key}_native_path"])
            for name in ROI_NAMES:
                extracted = sitk.Resample(native[name], native_reference, sitk.Transform(),
                                          sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
                source = array2d(extracted).astype(np.uint8)
                mapped = apply_orientation(source, orientation).astype(np.uint8)
                if mapped.shape != array2d(reference).shape:
                    raise ValueError(f"ROI/registered shape mismatch: {parameter} SAX {native_id}")
                path = output_root(cfg) / "roi_final_pose" / parameter / f"sax_slice{native_id:02d}_output{output_id:03d}_{name}.nii.gz"
                save_array_like(mapped, reference, path, overwrite, sitk.sitkUInt8)
                records.append({"parameter": parameter, "native_slice_id": native_id,
                                "output_id": output_id, "roi": name, "path": str(path),
                                "orientation_mapping": orientation,
                                "input_voxel_count": int(source.sum()),
                                "output_voxel_count": int(mapped.sum()),
                                "input_array_hash": sha256_array(source),
                                "output_array_hash": sha256_array(mapped),
                                "roi_extraction": "physical_identity_nearest_neighbor",
                                "native_affine_geometry": row[f"{key}_native_geometry"],
                                "final_affine_geometry": row[f"{key}_final_geometry"],
                                "geometry_status": "PASS" if source.sum() == mapped.sum() else "FAIL"})
    status = "PASS" if all(x["geometry_status"] == "PASS" for x in records) else "FAIL"
    dump_json({"subject_id": cfg["subject_id"], "status": status, "mappings": records},
              manifest_path, overwrite)
    return manifest_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--config", required=True)
    p.add_argument("--overwrite", action="store_true"); a = p.parse_args()
    print(run(load_yaml(a.config), a.overwrite))


if __name__ == "__main__": main()
