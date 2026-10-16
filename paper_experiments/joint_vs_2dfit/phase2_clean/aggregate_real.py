"""Subject-first Phase-4R cohort summaries; pixels and planes are never cohort samples."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from .contracts import ELIGIBLE_REAL_SUBJECTS


def bland_altman(reference_means: list[float], prediction_means: list[float]) -> dict[str, object]:
    reference, prediction = np.asarray(reference_means, float), np.asarray(prediction_means, float)
    if reference.shape != prediction.shape or reference.ndim != 1 or not len(reference):
        raise ValueError("Bland–Altman requires paired non-empty subject means")
    differences = prediction - reference
    sd = float(np.std(differences, ddof=1)) if len(differences) > 1 else float("nan")
    mean = float(np.mean(differences))
    return {"statistical_unit": "subject", "n_subjects": int(len(differences)), "paired_difference_ms": differences.tolist(), "mean_difference_ms": mean, "sd_difference_ms": sd, "loa_lower_ms": mean - 1.96 * sd, "loa_upper_ms": mean + 1.96 * sd}


EXPECTED_FOLDS = {"L100": 1, "L50": 2, "L33": 3, "L25": 4}


def pooled_r1_fold_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, object, object, object, object], list[dict[str, object]]] = defaultdict(list)
    for row in rows: grouped[(row["subject"], row["method"], row["level"], row["parameter"], row["phase"])].append(row)
    result = []
    for key, values in sorted(grouped.items()):
        support = sum(int(row["support_voxels"]) for row in values)
        if support <= 0: raise ValueError(f"empty fold support: {key}")
        sse = sum(float(row["sum_squared_error"]) for row in values); absolute = sum(float(row["sum_absolute_error"]) for row in values); signed = sum(float(row["sum_signed_error"]) for row in values)
        result.append({"subject": key[0], "method": key[1], "level": key[2], "parameter": key[3], "phase": key[4], "heldout_plane_count": len(values), "support_voxels": support, "pooled_rmse": float(np.sqrt(sse / support)), "pooled_mae": absolute / support, "pooled_bias": signed / support, "statistical_unit": "fold; planes pooled by pixel error only within fold"})
    return result


def summarize_r1_subjects(folds: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, object, object, object], list[dict[str, object]]] = defaultdict(list)
    for row in folds: grouped[(row["subject"], row["method"], row["level"], row["parameter"])].append(row)
    result = []
    for key, values in sorted(grouped.items()):
        phases = sorted(int(row["phase"]) for row in values); expected = list(range(EXPECTED_FOLDS[key[2]])); missing = sorted(set(expected).difference(phases))
        scores = [float(row["pooled_rmse"]) for row in sorted(values, key=lambda row: int(row["phase"]))]
        result.append({"subject": key[0], "method": key[1], "level": key[2], "parameter": key[3], "fold_count": len(values), "expected_fold_count": len(expected), "observed_phases": phases, "missing_phases": missing, "aggregation_status": "COMPLETE" if not missing else "INCOMPLETE_MISSING_OR_INELIGIBLE_FOLDS", "fold_values": scores, "rmse_mean_over_folds": float(np.mean(scores)), "rmse_sd_over_folds": float(np.std(scores, ddof=1)) if len(scores) > 1 else float("nan"), "rmse": float(np.mean(scores)), "statistical_unit": "subject; eligible phase folds equally weighted"})
    return result


def heldout_plane_rows(root: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(Path(root).rglob("heldout_quantitative_metrics.json")):
        report = json.loads(path.read_text())
        for stack, stack_data in report["stacks"].items():
            for plane in stack_data["planes"]:
                for parameter, endpoint in plane["parameters"].items():
                    for method in ("proposed", "fit_first"):
                        metric = endpoint[method]
                        rows.append({"subject": report["subject"], "level": report["condition"], "phase": report["phase"], "stack": stack, "original_group_idx": plane["original_group_idx"], "parameter": parameter, "method": method, **{key: metric.get(key) for key in ("rmse", "bias", "mae", "relative_nrmse", "valid_mape", "pearson_r", "ncc", "coverage_fraction", "reference_mean_ms", "prediction_mean_ms", "support_voxels", "sum_squared_error", "sum_absolute_error", "sum_signed_error")}, "endpoint": endpoint["endpoint"], "primary_region": endpoint["primary_region"]})
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]], fallback: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else fallback); writer.writeheader(); writer.writerows(rows)


def cohort_summary(subject_rows: list[dict[str, object]]) -> dict[str, object]:
    incomplete = [row for row in subject_rows if row.get("aggregation_status", "COMPLETE") != "COMPLETE"]
    if incomplete: raise ValueError(f"incomplete R1 fold coverage: {[(row['subject'], row['level'], row['missing_phases']) for row in incomplete]}")
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
    paired, paired_summary = paired_r0_comparison(rows)
    return rows, {"endpoint": "native-reference agreement/consistency, not true 3-D accuracy", "statistical_unit": "subject", "aha_status": "OPTIONAL_NOT_IMPLEMENTED", "method_vs_reference_bland_altman": ba, "paired_method_comparison": paired_summary, "paired_subject_rows": paired}


def paired_r0_comparison(rows: list[dict[str, object]], tolerance: float = 1e-6) -> tuple[list[dict[str, object]], dict[str, object]]:
    grouped: dict[tuple[object, object], dict[object, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        key = (row["subject"], row["parameter"]); method = row["method"]
        if method in grouped[key]: raise ValueError(f"duplicate subject/method record: {key}/{method}")
        grouped[key][method] = row
    paired = []
    for (subject, parameter), methods in sorted(grouped.items()):
        if set(methods) != {"proposed", "fit_first"}: raise ValueError(f"missing paired method result: {subject}/{parameter}")
        proposed, baseline = methods["proposed"], methods["fit_first"]
        if abs(float(proposed["reference_mean_ms"]) - float(baseline["reference_mean_ms"])) > tolerance: raise ValueError(f"reference mean mismatch: {subject}/{parameter}")
        paired.append({"subject": subject, "parameter": parameter, "proposed_rmse": proposed["rmse"], "fit_first_rmse": baseline["rmse"], "delta_rmse": float(proposed["rmse"]) - float(baseline["rmse"]), "proposed_mae": proposed["mae"], "fit_first_mae": baseline["mae"], "delta_mae": float(proposed["mae"]) - float(baseline["mae"]), "proposed_bias": proposed["bias"], "fit_first_bias": baseline["bias"], "proposed_reference_mean_ms": proposed["reference_mean_ms"], "proposed_prediction_mean_ms": proposed["prediction_mean_ms"], "fit_first_reference_mean_ms": baseline["reference_mean_ms"], "fit_first_prediction_mean_ms": baseline["prediction_mean_ms"]})
    summary = {}
    for parameter in sorted({row["parameter"] for row in paired}):
        parameter_rows = [row for row in paired if row["parameter"] == parameter]
        if {row["subject"] for row in parameter_rows} != set(ELIGIBLE_REAL_SUBJECTS): raise ValueError(f"missing formal paired subject: {parameter}")
        values = np.asarray([row["delta_rmse"] for row in parameter_rows], float)
        summary[parameter] = {"interpretation": "delta < 0 favors Proposed; delta > 0 favors fit-first", "n_subjects": len(values), "individual_subject_deltas": values.tolist(), "mean_delta": float(values.mean()), "sd_delta": float(values.std(ddof=1)) if len(values) > 1 else float("nan"), "median_delta": float(np.median(values)), "iqr_delta": float(np.percentile(values, 75)-np.percentile(values, 25))}
    return paired, summary


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
        _write_csv(output / "R0_real_full_paired_subject_table.csv", summary.pop("paired_subject_rows"), ["subject", "parameter", "delta_rmse"])
        (output / "R0_real_full_cohort_summary.json").write_text(json.dumps(summary, indent=2) + "\n"); return
    rows = heldout_plane_rows(args.r1_evaluation_root) if args.r1_evaluation_root else list(csv.DictReader(Path(args.r1_plane_metrics).open()))
    folds = pooled_r1_fold_metrics(rows); subjects = summarize_r1_subjects(folds); output.mkdir(parents=True)
    _write_csv(output / "R1_real_sparse_plane_metrics.csv", rows, ["subject", "method", "level", "phase", "stack", "original_group_idx", "parameter", "rmse"])
    _write_csv(output / "R1_real_sparse_fold_metrics.csv", folds, ["subject", "method", "level", "parameter", "phase", "pooled_rmse"])
    _write_csv(output / "R1_real_sparse_subject_summary.csv", subjects, ["subject", "method", "level", "parameter", "fold_count", "rmse", "statistical_unit"])
    (output / "R1_real_sparse_cohort_summary.json").write_text(json.dumps(cohort_summary(subjects), indent=2) + "\n")


if __name__ == "__main__":
    main()
