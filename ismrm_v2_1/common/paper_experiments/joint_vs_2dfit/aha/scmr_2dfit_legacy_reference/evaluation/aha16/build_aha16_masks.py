"""Build basal/mid/apical AHA masks from myocardium and RV landmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from evaluation.utils.common import (array2d, dump_json, load_yaml, output_root,
                                     read_image, save_array_like)


LEVELS = {"basal": (6, 1), "mid": (6, 7), "apical": (4, 13)}


def _point(selection: dict, level: str, name: str, fallback_mask: np.ndarray | None):
    value = selection.get(level, {}).get(name)
    if value is not None:
        return np.asarray(value, float)
    if fallback_mask is not None and fallback_mask.any():
        y, x = np.where(fallback_mask); return np.array([x.mean(), y.mean()])
    return None


def sectorize(mask: np.ndarray, anterior_xy: np.ndarray, inferior_xy: np.ndarray,
              n_segments: int, first_segment: int) -> np.ndarray:
    y, x = np.indices(mask.shape); yy, xx = np.where(mask)
    center = np.array([xx.mean(), yy.mean()])
    a0 = np.arctan2(anterior_xy[1]-center[1], anterior_xy[0]-center[0])
    ai = np.arctan2(inferior_xy[1]-center[1], inferior_xy[0]-center[0])
    # Choose the traversal direction that places the inferior landmark nearest
    # the opposite half of the circle. Final QC is mandatory before Figure 3.
    plus = (ai-a0) % (2*np.pi); minus = (a0-ai) % (2*np.pi)
    sign = 1.0 if abs(plus-np.pi) <= abs(minus-np.pi) else -1.0
    angle = (sign*(np.arctan2(y-center[1], x-center[0])-a0) + np.pi/n_segments) % (2*np.pi)
    local = np.floor(angle/(2*np.pi/n_segments)).astype(int)
    out = np.zeros(mask.shape, np.uint8); out[mask] = first_segment + local[mask]
    return out


def run(cfg, overwrite=False):
    root=output_root(cfg); target=root/"manifests/aha16_mask_manifest.json"
    selection_path=cfg.get("aha",{}).get("selection_file")
    roi_manifest=root/"manifests/roi_final_pose_manifest.json"
    if not selection_path or not Path(selection_path).exists() or not roi_manifest.exists():
        dump_json({"status":"SKIPPED","reason":"AHA selection/ROI unavailable"},target,overwrite)
        print("SKIPPED: AHA landmarks not available"); return target
    selection=load_yaml(selection_path).get("bullseye",{})
    mappings=json.loads(roi_manifest.read_text())
    if mappings.get("status") == "SKIPPED":
        dump_json({"status":"SKIPPED","reason":"myocardial ROI unavailable"},target,overwrite); return target
    lookup={(x["parameter"],int(x["native_slice_id"]),x["roi"]):x["path"] for x in mappings["mappings"]}
    records=[]
    for parameter in ("T1","T2"):
        inherited=None
        for level,(nseg,first) in LEVELS.items():
            native_id=selection.get(f"{level}_slice")
            if native_id is None: continue
            native_id=int(native_id)
            roi_path=lookup.get((parameter,native_id,cfg.get("aha",{}).get("roi","myocardium_core")))
            if not roi_path: continue
            ant_path=lookup.get((parameter,native_id,"landmark_anterior"))
            inf_path=lookup.get((parameter,native_id,"landmark_inferior"))
            ant_mask=array2d(ant_path)>0 if ant_path else None
            inf_mask=array2d(inf_path)>0 if inf_path else None
            anterior=_point(selection,level,"anterior_rv_point",ant_mask)
            inferior=_point(selection,level,"inferior_rv_point",inf_mask)
            if anterior is None or inferior is None:
                if level=="apical" and inherited is not None: anterior,inferior=inherited
                else: continue
            else: inherited=(anterior,inferior)
            roi=array2d(roi_path)>0; labels=sectorize(roi,anterior,inferior,nseg,first)
            ref=read_image(roi_path); out=root/"aha16"/parameter/f"{level}_sax_slice{native_id:02d}_segments.nii.gz"
            save_array_like(labels,ref,out,overwrite,sitk.sitkUInt8)
            records.append({"parameter":parameter,"level":level,"native_slice_id":native_id,
                            "path":str(out),"segments":list(range(first,first+nseg)),
                            "anterior_rv_point":[float(x) for x in anterior],
                            "inferior_rv_point":[float(x) for x in inferior]})
    status="COMPLETE" if len(records)==6 else "REVIEW"
    dump_json({"status":status,"records":records,"qc_required":True},target,overwrite); return target


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",required=True)
    p.add_argument("--overwrite",action="store_true"); a=p.parse_args(); print(run(load_yaml(a.config),a.overwrite))


if __name__ == "__main__": main()
