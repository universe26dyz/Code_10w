"""Evaluate SAX core/full myocardial statistics after final-pose ROI mapping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.utils.common import (array2d, common_mask_manifest_path, dump_json,
                                     load_yaml, output_root, write_csv)
from evaluation.utils.metrics import distribution_stats, masked_metrics


def run(cfg, overwrite=False):
    root=output_root(cfg); roi_manifest=root/"manifests/roi_final_pose_manifest.json"
    status_path=root/"manifests/myocardial_metrics_status.json"
    if not roi_manifest.exists() or json.loads(roi_manifest.read_text()).get("status") == "SKIPPED":
        dump_json({"status":"SKIPPED","reason":"myocardial ROI not available"},status_path,overwrite)
        print("SKIPPED: myocardial ROI not available"); return status_path
    common=json.loads(common_mask_manifest_path(cfg).read_text())["slices"]
    common={(x["parameter"],x["stack"],int(x["native_slice_id"])):x for x in common}
    mappings=json.loads(roi_manifest.read_text())["mappings"]
    rois={(x["parameter"],int(x["native_slice_id"]),x["roi"]):x["path"] for x in mappings}
    min_pixels=int(cfg.get("roi",{}).get("minimum_pixels",32)); rows=[]
    for (parameter,stack,native_id),x in common.items():
        if stack != "sax": continue
        original,synthetic=array2d(x["original_ms"]),array2d(x["synthetic_ms"])
        base=array2d(x["common_mask"])>0
        for roi_kind in ("myocardium_core","myocardium_full"):
            roi=array2d(rois[(parameter,native_id,roi_kind)])>0; mask=base&roi
            included=int(mask.sum()>=min_pixels)
            os,ss=distribution_stats(original[mask]),distribution_stats(synthetic[mask])
            paired=masked_metrics(original,synthetic,mask,min_pixels=min_pixels)
            rows.append({"subject_id":cfg["subject_id"],"parameter":parameter,"stack":"sax",
                         "native_slice_id":native_id,"output_id":x["output_id"],"roi":roi_kind,
                         "included":included,"exclusion_reason":"" if included else "empty/insufficient supported ROI",
                         "roi_pixels":int(roi.sum()),"supported_roi_pixels":int(mask.sum()),
                         **{f"original_{k}":v for k,v in os.items()},
                         **{f"synthetic_{k}":v for k,v in ss.items()},
                         "mean_difference":ss["mean"]-os["mean"],
                         "absolute_mean_difference":abs(ss["mean"]-os["mean"]),
                         **{f"paired_{k}":v for k,v in paired.items()}})
    path=root/"metrics/myocardial_per_slice.csv"; write_csv(rows,path,overwrite)
    dump_json({"status":"COMPLETE","rows":len(rows),"csv":str(path)},status_path,overwrite)
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",required=True)
    p.add_argument("--overwrite",action="store_true"); a=p.parse_args(); print(run(load_yaml(a.config),a.overwrite))


if __name__ == "__main__": main()
