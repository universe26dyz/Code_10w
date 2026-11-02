"""Validate that all 16 AHA segments are present and sufficiently supported."""

from __future__ import annotations

import argparse
from collections import defaultdict

from evaluation.utils.common import dump_json, load_yaml, output_root, read_csv


def run(cfg,overwrite=False):
    root=output_root(cfg); stats=root/"aha16/aha16_stats.csv"; target=root/"manifests/aha16_validation.json"
    if not stats.exists(): report={"status":"SKIPPED","reason":"AHA statistics unavailable"}
    else:
        rows=read_csv(stats); minimum=int(cfg.get("aha",{}).get("minimum_segment_pixels",10)); failures=[]
        groups=defaultdict(set)
        for r in rows:
            groups[r["parameter"]].add(int(r["segment"]))
            if int(r["valid_voxel_count"])<minimum: failures.append({"parameter":r["parameter"],"segment":r["segment"],"reason":"insufficient support"})
        for parameter in ("T1","T2"):
            missing=sorted(set(range(1,17))-groups[parameter])
            if missing: failures.append({"parameter":parameter,"missing_segments":missing})
        report={"status":"PASS" if not failures else "FAIL","minimum_segment_pixels":minimum,"failures":failures,
                "note":"Visual AHA orientation QC remains mandatory before final Figure 3."}
    dump_json(report,target,overwrite); return target


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--config",required=True)
    p.add_argument("--overwrite",action="store_true"); a=p.parse_args(); print(run(load_yaml(a.config),a.overwrite))


if __name__ == "__main__": main()
