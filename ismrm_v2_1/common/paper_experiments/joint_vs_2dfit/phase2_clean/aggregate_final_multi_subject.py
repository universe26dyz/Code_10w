"""Subject-level aggregation for immutable final five-subject paired summaries."""
from __future__ import annotations

import csv, json
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
from .contracts import sha256, write_immutable_json

SCHEMA = "phase4r_final_multi_subject_statistics/v1"
SUBJECTS = ("CYJ", "DYL_2", "DYZ", "HHZ", "HJL")


def aggregate_final_multi_subject(*, subject_summaries: list[str | Path], output: str | Path) -> Path:
    target = Path(output)
    if target.exists(): raise FileExistsError(f"refusing overwrite: {target}")
    records: list[dict[str, Any]] = []; evidence: dict[str, dict[str, str]] = {}
    for source in subject_summaries:
        path = Path(source); payload = json.loads(path.read_text(encoding="utf-8")); subject = payload.get("subject")
        if subject not in SUBJECTS or subject in evidence: raise ValueError("one immutable final summary is required for each canonical subject")
        rows = payload.get("paired_subject_metrics", payload.get("final_metrics"))
        if not isinstance(rows, list) or not rows: raise ValueError(f"{subject}: final summary lacks paired_subject_metrics")
        evidence[subject] = {"path": str(path.resolve()), "sha256": sha256(path)}
        for row in rows:
            if not isinstance(row, dict) or row.get("paired_support_status") != "AUDITED_IDENTICAL": raise ValueError(f"{subject}: PAIRED_SUPPORT_MISMATCH")
            if any(key not in row for key in ("endpoint", "parameter", "metric", "region", "proposed", "fit_first", "support_identity")): raise ValueError(f"{subject}: incomplete paired metric")
            records.append({"subject": subject, **row, "difference_proposed_minus_fit_first": float(row["proposed"]) - float(row["fit_first"])})
    if set(evidence) != set(SUBJECTS): raise ValueError("final aggregation is INCOMPLETE: all five subjects are required")
    target.mkdir(parents=True); fields = list(records[0])
    for name in ("per_subject.csv", "paired_subject_differences.csv"):
        with (target / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records: grouped[(row["endpoint"], row["parameter"], row["metric"], row["region"])].append(row)
    groups = [{"endpoint": key[0], "parameter": key[1], "metric": key[2], "region": key[3], "subject_count": len(rows), "proposed_mean": float(np.mean([row["proposed"] for row in rows])), "fit_first_mean": float(np.mean([row["fit_first"] for row in rows])), "paired_difference_mean": float(np.mean([row["difference_proposed_minus_fit_first"] for row in rows]))} for key, rows in grouped.items()]
    write_immutable_json(target / "cohort_summary.json", {"schema": SCHEMA, "unit_of_analysis": "subject", "inference": "descriptive_only_no_default_asymptotic_test", "groups": groups})
    with (target / "paper_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(groups[0])); writer.writeheader(); writer.writerows(groups)
    write_immutable_json(target / "statistics_manifest.json", {"schema": SCHEMA, "subjects": list(SUBJECTS), "subject_summaries": evidence, "unit_of_analysis": "subject", "paired_support_required": "AUDITED_IDENTICAL", "immutable": True})
    return target
