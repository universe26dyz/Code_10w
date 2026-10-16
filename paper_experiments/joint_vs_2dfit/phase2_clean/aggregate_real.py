"""Subject-first Phase-4R cohort summaries; pixels and planes are never cohort samples."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def bland_altman(reference_means: list[float], prediction_means: list[float]) -> dict[str, object]:
    reference, prediction = np.asarray(reference_means, float), np.asarray(prediction_means, float)
    if reference.shape != prediction.shape or reference.ndim != 1 or not len(reference):
        raise ValueError("Bland–Altman requires paired non-empty subject means")
    differences = prediction - reference
    sd = float(np.std(differences, ddof=1)) if len(differences) > 1 else float("nan")
    mean = float(np.mean(differences))
    return {"statistical_unit": "subject", "n_subjects": int(len(differences)), "paired_difference_ms": differences.tolist(), "mean_difference_ms": mean, "sd_difference_ms": sd, "loa_lower_ms": mean - 1.96 * sd, "loa_upper_ms": mean + 1.96 * sd}


def summarize_r1_subjects(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, object, object, object], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["subject"], row["method"], row["level"], row.get("parameter", "combined"))].append(float(row["rmse"]))
    return [{"subject": key[0], "method": key[1], "level": key[2], "parameter": key[3], "fold_count": len(values), "rmse": float(np.mean(values)), "statistical_unit": "subject; folds aggregated before cohort comparison"} for key, values in sorted(grouped.items())]


def heldout_plane_rows(root: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(Path(root).rglob("heldout_quantitative_metrics.json")):
        report = json.loads(path.read_text())
        for stack, stack_data in report["stacks"].items():
            for plane in stack_data["planes"]:
                for parameter, endpoint in plane["parameters"].items():
                    for method in ("proposed", "fit_first"):
                        metric = endpoint[method]
                        rows.append({"subject": report["subject"], "level": report["condition"], "phase": report["phase"], "stack": stack, "original_group_idx": plane["original_group_idx"], "parameter": parameter, "method": method, **{key: metric.get(key) for key in ("rmse", "bias", "mae", "relative_nrmse", "valid_mape", "pearson_r", "ncc", "coverage_fraction", "reference_mean_ms", "prediction_mean_ms")}, "endpoint": endpoint["endpoint"], "primary_region": endpoint["primary_region"]})
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]], fallback: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else fallback); writer.writeheader(); writer.writerows(rows)


def cohort_summary(subject_rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[object, object, object], list[float]] = defaultdict(list)
    for row in subject_rows:
        grouped[(row["method"], row["level"], row.get("parameter", "combined"))].append(float(row["rmse"]))
    return {"statistical_unit": "subject", "groups": [{"method": key[0], "level": key[1], "parameter": key[2], "n_subjects": len(values), "mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"), "median": float(np.median(values)), "iqr": float(np.percentile(values, 75) - np.percentile(values, 25))} for key, values in sorted(grouped.items())]}


def summarize_r0_evaluations(records: list[tuple[str, str, dict[str, object]]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Return subject rows and subject-level Bland–Altman without pixel inference."""
    rows: list[dict[str, object]] = []
    for subject, method, report in records:
        for parameter, value in report["metrics"].items():
            metric = value["subject_level"]
            rows.append({"subject": subject, "method": method, "parameter": parameter, **{key: metric.get(key) for key in ("rmse", "mae", "bias", "relative_nrmse", "valid_mape", "pearson_r", "ncc", "coverage_fraction", "reference_mean_ms", "prediction_mean_ms", "mean_difference_ms")}})
    ba = {}
    for method in sorted({row["method"] for row in rows}):
        for parameter in ("t1_ms", "t2_ms"):
            matched = [row for row in rows if row["method"] == method and row["parameter"] == parameter]
            ba[f"{method}_{parameter}"] = bland_altman([float(row["reference_mean_ms"]) for row in matched], [float(row["prediction_mean_ms"]) for row in matched])
    return rows, {"endpoint": "native-reference agreement/consistency, not true 3-D accuracy", "statistical_unit": "subject", "aha_status": "OPTIONAL_NOT_IMPLEMENTED", "bland_altman": ba}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r1-plane-metrics"); parser.add_argument("--r1-evaluation-root"); parser.add_argument("--r0-evaluation-root"); parser.add_argument("--output-root", required=True)
    args = parser.parse_args(); output = Path(args.output_root)
    if output.exists(): raise FileExistsError(f"refusing overwrite: {output}")
    if sum(bool(value) for value in (args.r1_plane_metrics, args.r1_evaluation_root, args.r0_evaluation_root)) != 1: raise ValueError("provide exactly one input mode")
    if args.r0_evaluation_root:
        records = []
        for path in sorted(Path(args.r0_evaluation_root).glob("*_*.json")):
            subject, separator, method = path.stem.partition("_")
            if not separator or method not in ("proposed", "fit_first"): raise ValueError(f"unexpected R0 evaluation filename: {path.name}")
            records.append((subject, method, json.loads(path.read_text())))
        rows, summary = summarize_r0_evaluations(records); output.mkdir(parents=True)
        _write_csv(output / "R0_real_full_subject_table.csv", rows, ["subject", "method", "parameter"])
        (output / "R0_real_full_cohort_summary.json").write_text(json.dumps(summary, indent=2) + "\n"); return
    rows = heldout_plane_rows(args.r1_evaluation_root) if args.r1_evaluation_root else list(csv.DictReader(Path(args.r1_plane_metrics).open()))
    subjects = summarize_r1_subjects(rows); output.mkdir(parents=True)
    if args.r1_evaluation_root:
        _write_csv(output / "R1_real_sparse_plane_metrics.csv", rows, ["subject", "method", "level", "phase", "stack", "original_group_idx", "parameter", "rmse"])
        folds = []
        groups: dict[tuple[object, object, object, object], list[float]] = defaultdict(list)
        for row in rows: groups[(row["subject"], row["method"], row["level"], row["parameter"], row["phase"])].append(float(row["rmse"]))
        for key, values in sorted(groups.items()): folds.append({"subject": key[0], "method": key[1], "level": key[2], "parameter": key[3], "phase": key[4], "plane_count": len(values), "rmse": float(np.mean(values)), "statistical_unit": "fold; held-out planes descriptive only"})
        _write_csv(output / "R1_real_sparse_fold_metrics.csv", folds, ["subject", "method", "level", "parameter", "phase", "rmse"])
    _write_csv(output / "R1_real_sparse_subject_summary.csv", subjects, ["subject", "method", "level", "parameter", "fold_count", "rmse", "statistical_unit"])
    (output / "R1_real_sparse_cohort_summary.json").write_text(json.dumps(cohort_summary(subjects), indent=2) + "\n")


if __name__ == "__main__":
    main()
