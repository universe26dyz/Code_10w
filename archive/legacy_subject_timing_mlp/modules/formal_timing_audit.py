"""Write the required formal timing audit from a provenance-preserving pool."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def audit_formal_timing_pool(pool_path: str | Path, csv_path: str | Path, summary_path: str | Path) -> dict[str, object]:
    with np.load(pool_path, allow_pickle=False) as data:
        required = {"timing9_ms", "tr_ms", "vps", "subject_id", "stack", "group_id", "source_id", "functional_fixture"}
        if required.difference(data.files): raise ValueError("Formal timing audit requires a manifest-derived provenance pool.")
        timing = np.asarray(data["timing9_ms"], dtype=np.float64); subjects = data["subject_id"].astype(str); stacks = data["stack"].astype(str); groups = data["group_id"].astype(int); sources = data["source_id"].astype(str)
        tr, vps = float(np.asarray(data["tr_ms"]).reshape(-1)[0]), int(np.asarray(data["vps"]).reshape(-1)[0])
        if bool(np.asarray(data["functional_fixture"]).reshape(-1)[0]): raise ValueError("Functional fixture cannot be used for formal timing audit.")
    csv_path, summary_path = Path(csv_path), Path(summary_path)
    if csv_path.exists() or summary_path.exists(): raise FileExistsError("Formal timing audit outputs must not be overwritten.")
    csv_path.parent.mkdir(parents=True, exist_ok=True); summary_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "subject_id",
            "stack",
            "group_id",
            "source_id",
            "TR",
            "VPS",
            *[f"timing{i}" for i in range(2, 11)],
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(timing.shape[0]): writer.writerow({"subject_id": subjects[i], "stack": stacks[i], "group_id": groups[i], "source_id": sources[i], "TR": tr, "VPS": vps, **{f"timing{j+2}": timing[i, j] for j in range(9)}})
    summary = {"n_subjects": int(np.unique(subjects).size), "n_groups": int(timing.shape[0]), "tr_unique": [tr], "tr_range": [tr, tr], "vps_unique": [vps], "timing_min_ms": timing.min(0).tolist(), "timing_max_ms": timing.max(0).tolist(), "timing_mean_ms": timing.mean(0).tolist(), "timing_std_ms": timing.std(0).tolist()}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--pool", required=True); parser.add_argument("--csv", required=True); parser.add_argument("--summary", required=True)
    args = parser.parse_args(); print(json.dumps(audit_formal_timing_pool(args.pool, args.csv, args.summary), indent=2))


if __name__ == "__main__": main()
