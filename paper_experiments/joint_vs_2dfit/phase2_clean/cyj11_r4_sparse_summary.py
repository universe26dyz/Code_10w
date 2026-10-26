"""Summarize immutable r4 metrics by protocol and sparse acquisition case.

Each subject contributes one value per sparse case: when a case has several
folds, its fold metrics are averaged within that subject before five-subject
mean and sample-SD statistics are calculated.  This keeps fold count from
changing a subject's statistical weight.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import sha256, write_immutable_json
from .cyj11_r4_reporting import PROTOCOL_CASES, R4_SOURCE, SUBJECTS


SUMMARY_SCHEMA = "cyj11_r4_sparse_case_summary/v1"
METRICS = {
    "rmse": "rmse",
    "nrmse": "relative_nrmse",
    "ncc": "ncc",
    "mae": "mae",
}
ERROR_METRICS = frozenset(("rmse", "nrmse", "mae"))


def sparse_case_id(case: str) -> str:
    """Remove only a terminal explicit fold suffix from an r4 case name."""
    prefix, marker, suffix = case.rpartition("_fold")
    return prefix if marker and suffix.isdigit() else case


def _require_metric_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"r4 aggregate metrics does not exist: {path}")


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sample_sd(values: Sequence[float]) -> float | None:
    return float(statistics.stdev(values)) if len(values) > 1 else None


def _case_metric_rows(
    path: Path,
    *,
    subject: str,
    protocol: str,
    case: str,
) -> list[dict[str, Any]]:
    _require_metric_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "final_stratified_native_common_support_metrics/v1":
        raise ValueError(f"r4 aggregate metrics has unsupported schema: {path}")
    regions = payload.get("regions")
    if not isinstance(regions, Mapping):
        raise ValueError(f"r4 aggregate metrics lacks regions: {path}")
    rows: list[dict[str, Any]] = []
    for parameter in ("t1", "t2"):
        parameter_regions = regions.get(parameter)
        if not isinstance(parameter_regions, Mapping):
            raise ValueError(f"r4 aggregate metrics lacks {parameter} regions: {path}")
        for region, record in sorted(parameter_regions.items()):
            if not isinstance(record, Mapping) or record.get("status") != "COMPLETE":
                continue
            proposed, fit_first = record.get("proposed"), record.get("fit_first")
            if not isinstance(proposed, Mapping) or not isinstance(fit_first, Mapping):
                raise ValueError(f"complete r4 region lacks paired methods: {path} ({parameter}/{region})")
            for name, source_name in METRICS.items():
                proposed_value, fit_first_value = _numeric(proposed.get(source_name)), _numeric(fit_first.get(source_name))
                # A constant reference makes NCC intentionally unavailable; do
                # not manufacture a correlation statistic from that case.
                if proposed_value is None or fit_first_value is None:
                    continue
                rows.append({
                    "subject": subject,
                    "protocol": protocol,
                    "case": case,
                    "sparse_case": sparse_case_id(case),
                    "parameter": parameter,
                    "region": region,
                    "metric": name,
                    "proposed": proposed_value,
                    "fit_first": fit_first_value,
                })
    return rows


def _subject_case_rows(fold_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Pool folds inside an individual subject, never across subjects."""
    grouped: dict[tuple[str, str, str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        grouped[(
            str(row["subject"]), str(row["protocol"]), str(row["sparse_case"]),
            str(row["parameter"]), str(row["region"]), str(row["metric"]),
        )].append(row)
    output: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        subject, protocol, sparse_case, parameter, region, metric = key
        ordered = sorted(rows, key=lambda item: str(item["case"]))
        output.append({
            "subject": subject,
            "protocol": protocol,
            "sparse_case": sparse_case,
            "parameter": parameter,
            "region": region,
            "metric": metric,
            "fold_cases": [str(item["case"]) for item in ordered],
            "fold_count": len(ordered),
            "proposed": float(statistics.fmean(float(item["proposed"]) for item in ordered)),
            "fit_first": float(statistics.fmean(float(item["fit_first"]) for item in ordered)),
        })
    return output


def _summary_rows(subject_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in subject_rows:
        grouped[(
            str(row["protocol"]), str(row["sparse_case"]), str(row["parameter"]),
            str(row["region"]), str(row["metric"]),
        )].append(row)
    output: list[dict[str, Any]] = []
    for (protocol, sparse_case, parameter, region, metric), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda item: str(item["subject"]))
        proposed = [float(item["proposed"]) for item in ordered]
        fit_first = [float(item["fit_first"]) for item in ordered]
        proposed_mean, fit_first_mean = float(statistics.fmean(proposed)), float(statistics.fmean(fit_first))
        if metric in ERROR_METRICS:
            denominator, direction = fit_first_mean, "lower_is_better"
            improvement = None if denominator == 0.0 else (fit_first_mean - proposed_mean) / denominator * 100.0
        else:
            denominator, direction = abs(fit_first_mean), "higher_is_better"
            improvement = None if denominator == 0.0 else (proposed_mean - fit_first_mean) / denominator * 100.0
        output.append({
            "protocol": protocol,
            "sparse_case": sparse_case,
            "parameter": parameter,
            "region": region,
            "metric": metric,
            "metric_source_field": METRICS[metric],
            "n_subjects": len(ordered),
            "n_subject_case_folds": int(sum(int(item["fold_count"]) for item in ordered)),
            "subjects": [str(item["subject"]) for item in ordered],
            "fold_cases_by_subject": {str(item["subject"]): list(item["fold_cases"]) for item in ordered},
            "proposed_mean": proposed_mean,
            "proposed_sample_sd": _sample_sd(proposed),
            "fit_first_mean": fit_first_mean,
            "fit_first_sample_sd": _sample_sd(fit_first),
            "proposed_percent_improvement_vs_fit_first": None if improvement is None else float(improvement),
            "improvement_direction": direction,
            "improvement_status": "UNDEFINED_ZERO_FIT_FIRST_MEAN" if improvement is None else "COMPLETE",
        })
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            for key, value in tuple(flat.items()):
                if isinstance(value, (dict, list)):
                    flat[key] = json.dumps(value, sort_keys=True)
            writer.writerow(flat)


def summary_plan(
    analysis_root: str | Path,
    output: str | Path,
    *,
    subjects: Sequence[str] = SUBJECTS,
    protocols: Mapping[str, Sequence[str]] = PROTOCOL_CASES,
    metric_source: str = R4_SOURCE,
) -> dict[str, Any]:
    """Validate every required immutable input before creating the summary."""
    root, target = Path(analysis_root).resolve(), Path(output).resolve()
    chosen_subjects = tuple(subjects)
    if not chosen_subjects or len(chosen_subjects) != len(set(chosen_subjects)) or any(subject not in SUBJECTS for subject in chosen_subjects):
        raise ValueError(f"subjects must be a unique non-empty subset of {', '.join(SUBJECTS)}")
    if not protocols or any(protocol not in PROTOCOL_CASES for protocol in protocols):
        raise ValueError("protocols must be a non-empty mapping limited to profile-v2 and legacy-density")
    if target.exists():
        raise FileExistsError(f"refusing existing r4 sparse summary destination: {target}")
    inputs: list[dict[str, str]] = []
    for protocol, cases in protocols.items():
        allowed = PROTOCOL_CASES[protocol]
        if not cases or len(cases) != len(set(cases)) or any(case not in allowed for case in cases):
            raise ValueError(f"cases must be a unique non-empty subset of {protocol}")
        for subject in chosen_subjects:
            for case in cases:
                path = root / "metrics" / "central" / subject / case / metric_source / "aggregate_metrics.json"
                _require_metric_file(path)
                inputs.append({"subject": subject, "protocol": protocol, "case": case, "path": str(path.resolve()), "sha256": sha256(path)})
    return {
        "schema": "cyj11_r4_sparse_case_summary_plan/v1",
        "analysis_root": str(root),
        "output": str(target),
        "metric_source": metric_source,
        "subjects": list(chosen_subjects),
        "protocol_cases": {protocol: list(cases) for protocol, cases in protocols.items()},
        "inputs": inputs,
        "fold_aggregation": "mean each subject's available folds within one protocol/sparse_case before cross-subject statistics",
        "cross_subject_standard_deviation": "sample standard deviation (ddof=1)",
        "percent_improvement": {
            "rmse_nrmse_mae": "100 * (fit_first_mean - proposed_mean) / fit_first_mean; positive is better",
            "ncc": "100 * (proposed_mean - fit_first_mean) / abs(fit_first_mean); positive is better",
        },
    }


def summarize_r4_sparse_cases(
    analysis_root: str | Path,
    output: str | Path,
    *,
    subjects: Sequence[str] = SUBJECTS,
    protocols: Mapping[str, Sequence[str]] = PROTOCOL_CASES,
    metric_source: str = R4_SOURCE,
) -> Path:
    """Write a no-overwrite protocol-separated sparse-case summary bundle."""
    plan = summary_plan(analysis_root, output, subjects=subjects, protocols=protocols, metric_source=metric_source)
    target = Path(plan["output"])
    fold_rows: list[dict[str, Any]] = []
    for item in plan["inputs"]:
        fold_rows.extend(_case_metric_rows(
            Path(item["path"]), subject=item["subject"], protocol=item["protocol"], case=item["case"],
        ))
    if not fold_rows:
        raise ValueError("required r4 aggregate metrics have no complete paired metric rows")
    subject_rows = _subject_case_rows(fold_rows)
    summaries = _summary_rows(subject_rows)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        _write_csv(staging / "summary.csv", summaries)
        write_immutable_json(staging / "summary.json", {
            "schema": SUMMARY_SCHEMA,
            "analysis_root": plan["analysis_root"],
            "metric_source": metric_source,
            "subjects": plan["subjects"],
            "protocol_cases": plan["protocol_cases"],
            "fold_aggregation": plan["fold_aggregation"],
            "cross_subject_standard_deviation": plan["cross_subject_standard_deviation"],
            "percent_improvement": plan["percent_improvement"],
            "input_metric_files": plan["inputs"],
            "subject_fold_aggregates": subject_rows,
            "summaries": summaries,
        })
        write_immutable_json(staging / "manifest.json", {
            "schema": "cyj11_r4_sparse_case_summary_bundle/v1",
            "summary": {"path": str((target / "summary.json").resolve()), "sha256": sha256(staging / "summary.json")},
            "csv": {"path": str((target / "summary.csv").resolve()), "sha256": sha256(staging / "summary.csv")},
            "input_metric_files": plan["inputs"],
            "no_metric_files_modified": True,
        })
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subject", action="append", choices=SUBJECTS)
    parser.add_argument("--protocol", action="append", choices=tuple(PROTOCOL_CASES))
    parser.add_argument("--metric-source", default=R4_SOURCE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    subjects = tuple(args.subject or SUBJECTS)
    protocols = {protocol: PROTOCOL_CASES[protocol] for protocol in (args.protocol or tuple(PROTOCOL_CASES))}
    plan = summary_plan(args.analysis_root, args.output, subjects=subjects, protocols=protocols, metric_source=args.metric_source)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    print(summarize_r4_sparse_cases(args.analysis_root, args.output, subjects=subjects, protocols=protocols, metric_source=args.metric_source))


if __name__ == "__main__":
    main()
