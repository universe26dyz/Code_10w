"""Compute original and PSF-reprojection statistics for AHA segments."""

from __future__ import annotations

import argparse
import json

import numpy as np

from evaluation.utils.common import (array2d, common_mask_manifest_path, dump_json,
                                     load_yaml, output_root, write_csv)
from evaluation.utils.metrics import distribution_stats


def run(cfg,overwrite=False):
    root=output_root(cfg); manifest=root/"manifests/aha16_mask_manifest.json"; status=root/"manifests/aha16_stats_status.json"
    if not manifest.exists() or json.loads(manifest.read_text()).get("status") not in ("COMPLETE","REVIEW"):
        dump_json({"status":"SKIPPED","reason":"AHA masks unavailable"},status,overwrite); print("SKIPPED: AHA masks unavailable"); return status
    common=json.loads(common_mask_manifest_path(cfg).read_text())["slices"]
    common={(x["parameter"],int(x["native_slice_id"])):x for x in common if x["stack"]=="sax"}
    rows=[]
    for rec in json.loads(manifest.read_text())["records"]:
        x=common[(rec["parameter"],int(rec["native_slice_id"]))]
        labels=array2d(rec["path"]).astype(int); support=array2d(x["common_mask"])>0
        orig,syn=array2d(x["original_ms"]),array2d(x["synthetic_ms"])
        for segment in rec["segments"]:
            idx=(labels==segment)&support; a,b=distribution_stats(orig[idx]),distribution_stats(syn[idx])
            rows.append({"subject_id":cfg["subject_id"],"parameter":rec["parameter"],"level":rec["level"],
                         "native_slice_id":rec["native_slice_id"],"segment":segment,"valid_voxel_count":int(idx.sum()),
                         "original_mean":a["mean"],"original_SD":a["SD"],"reprojection_mean":b["mean"],
                         "reprojection_SD":b["SD"],"signed_difference":b["mean"]-a["mean"],
                         "absolute_difference":abs(b["mean"]-a["mean"])})
    path=root/"aha16/aha16_stats.csv"; write_csv(rows,path,overwrite)
    dump_json({"status":"COMPLETE","csv":str(path),"rows":len(rows)},status,overwrite); return path


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",required=True)
    p.add_argument("--overwrite",action="store_true"); a=p.parse_args(); print(run(load_yaml(a.config),a.overwrite))


if __name__ == "__main__": main()
