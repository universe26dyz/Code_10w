"""Local-only CYJ1.1 result summaries and AHA16 SD-figure backfill.

This module consumes completed immutable downstream AHA/ROI outputs.  It does
not create endpoints, run reconstruction, or write a formal metric artifact.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ..phase2_clean.contracts import sha256, write_immutable_json
from .cyj11_myocardial_roi_metrics import ROI_NAMES, _load_scmr_masks
from .cyj11_runner import load_cyj11_run_inputs, native_sax_plane
from .legacy_aha16 import render_single_bullseye


RESULT_TAG_DEFAULT = "cyj11_scmr_aha_roi_20261026"
SUMMARY_SCHEMA = "cyj11_sparse_myocardial_summary/v1"
SD_SCHEMA = "cyj11_aha16_sd_backfill/v1"
METHODS = ("proposed", "fit_first")
METRICS = ("rmse", "nrmse", "ncc", "mae")
FOLD_SUFFIX = re.compile(r"_fold\\d+$")


def sparse_case_name(case: str) -> str:
    """Group repeated fold realizations as one sparse acquisition setting."""
    return FOLD_SUFFIX.sub("", str(case))


def _finite_summary(values: Iterable[float | None]) -> dict[str, float | int | None]:
    array = np.asarray([value for value in values if value is not None and np.isfinite(value)], float)
    return {"n": int(array.size), "mean": None if not array.size else float(array.mean()),
            "sample_sd": None if not array.size else float(array.std(ddof=1)) if array.size > 1 else 0.0}


def summarize_sparsity_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate subject/fold ROI rows, retaining paired method improvements."""
    rows = [dict(row) for row in rows]
    grouped: dict[tuple[str, str, str, str], list[float | None]] = defaultdict(list)
    paired: dict[tuple[str, str, str, str, str], dict[str, float | None]] = defaultdict(dict)
    for row in rows:
        sparse = str(row.get("sparse_case") or sparse_case_name(str(row["case"])))
        identity = (str(row["subject"]), str(row["case"]), str(row["parameter"]), str(row["roi"]))
        for metric in METRICS:
            value = row.get(metric)
            value = None if value is None or not np.isfinite(value) else float(value)
            grouped[(sparse, str(row["parameter"]), str(row["roi"]), str(row["method"]), metric)].append(value)
            paired[(*identity, metric)][str(row["method"])] = value
    metric_rows: list[dict[str, Any]] = []
    for (sparse, parameter, roi, method, metric), values in sorted(grouped.items()):
        metric_rows.append({"sparse_case": sparse, "parameter": parameter, "roi": roi,
                            "method": method, "metric": metric, **_finite_summary(values)})
    improvements: dict[tuple[str, str, str, str], list[float | None]] = defaultdict(list)
    for (_subject, case, parameter, roi, metric), values in paired.items():
        proposed, baseline = values.get("proposed"), values.get("fit_first")
        if proposed is None or baseline is None or baseline == 0:
            improvement = None
        elif metric == "ncc":
            improvement = 100.0 * (proposed - baseline) / abs(baseline)
        else:
            improvement = 100.0 * (baseline - proposed) / abs(baseline)
        improvements[(sparse_case_name(case), parameter, roi, metric)].append(improvement)
    improvement_rows = []
    for (sparse, parameter, roi, metric), values in sorted(improvements.items()):
        summary = _finite_summary(values)
        improvement_rows.append({"sparse_case": sparse, "parameter": parameter, "roi": roi, "metric": metric,
                                 "baseline_method": "fit_first", "comparison_method": "proposed",
                                 "direction": "higher_is_better" if metric == "ncc" else "lower_is_better",
                                 "formula": "100 * (proposed - fit_first) / abs(fit_first)" if metric == "ncc"
                                 else "100 * (fit_first - proposed) / abs(fit_first)",
                                 "mean_percent": summary["mean"], "sample_sd_percent": summary["sample_sd"], "n": summary["n"]})
    return metric_rows, improvement_rows


def _paired_metrics(reference: np.ndarray, prediction: np.ndarray, support: np.ndarray) -> dict[str, float | None]:
    reference, prediction, support = np.asarray(reference, float), np.asarray(prediction, float), np.asarray(support, bool)
    values, estimate = reference[support], prediction[support]
    if not values.size:
        return {metric: None for metric in METRICS}
    difference = estimate - values
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    reference_range = float(values.max() - values.min())
    centered_reference, centered_estimate = values - values.mean(), estimate - estimate.mean()
    denominator = float(np.linalg.norm(centered_reference) * np.linalg.norm(centered_estimate))
    return {"rmse": rmse, "nrmse": None if reference_range == 0 else rmse / reference_range,
            "ncc": None if denominator == 0 else float(np.dot(centered_reference, centered_estimate) / denominator),
            "mae": float(np.abs(difference).mean())}


def _metric_manifests(root: Path, result_tag: str) -> list[Path]:
    return sorted((root / "metrics" / "central").glob(f"*/*/{result_tag}/myocardial_roi/manifest.json"))


def collect_myocardial_rows(*, analysis_root: str | Path, result_tag: str) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Recompute the requested four metrics from completed local ROI inputs."""
    root = Path(analysis_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    manifests = _metric_manifests(root, result_tag)
    if not manifests:
        raise FileNotFoundError(f"no completed myocardial ROI manifests found for result tag {result_tag!r}")
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "cyj11_local_myocardial_roi_metrics/v1":
            raise ValueError(f"unexpected myocardial ROI schema: {manifest_path}")
        subject, case = str(manifest["subject"]), str(manifest["case"])
        expected = root / "metrics" / "central" / subject / case / result_tag / "myocardial_roi" / "manifest.json"
        if manifest_path.resolve() != expected.resolve():
            raise ValueError(f"myocardial ROI path/identity mismatch: {manifest_path}")
        bundle = Path(str(manifest["scmr_myocardium_bundle"]["archive"]["path"])).parent
        native, endpoints = load_cyj11_run_inputs(root, subject, case)
        groups = np.asarray(native["group_idx"], int)
        masks = _load_scmr_masks(bundle, expected_groups=groups, expected_shape=tuple(map(int, native["t1_ms"].shape[1:])))
        values = {parameter: {roi: {method: [] for method in METHODS} for roi in ROI_NAMES} for parameter in ("t1", "t2")}
        for group_index, group in enumerate(groups):
            for parameter in ("t1", "t2"):
                plane = native_sax_plane(native=native, group=int(group), endpoint_images=endpoints, parameter=parameter)
                support_base = (np.isfinite(plane["reference"]) & np.isfinite(plane["proposed"]) & np.isfinite(plane["fit_first"])
                                & plane["proposed_valid"] & plane["fit_first_valid"])
                for roi in ROI_NAMES:
                    support = masks[roi][group_index] & support_base
                    for method in METHODS:
                        if support.any():
                            values[parameter][roi][method].append((plane["reference"][support], plane[method][support]))
        for parameter in ("t1", "t2"):
            for roi in ROI_NAMES:
                for method in METHODS:
                    pairs = values[parameter][roi][method]
                    if pairs:
                        reference, prediction = (np.concatenate(item) for item in zip(*pairs))
                        scores = _paired_metrics(reference, prediction, np.ones(reference.shape, bool))
                        support_voxels = int(reference.size)
                    else:
                        scores, support_voxels = {metric: None for metric in METRICS}, 0
                    rows.append({"subject": subject, "case": case, "sparse_case": sparse_case_name(case),
                                 "parameter": parameter, "roi": roi, "method": method,
                                 "common_support_voxels": support_voxels, **scores})
        sources.append({"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)})
    return rows, sources


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def build_sparse_myocardial_summary(*, analysis_root: str | Path, result_tag: str, output: str | Path) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    case_rows, sources = collect_myocardial_rows(analysis_root=analysis_root, result_tag=result_tag)
    metric_rows, improvement_rows = summarize_sparsity_rows(case_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        _write_csv(staging / "case_metric_rows.csv", case_rows,
                   ("subject", "case", "sparse_case", "parameter", "roi", "method", "common_support_voxels", *METRICS))
        _write_csv(staging / "sparse_case_metric_summary.csv", metric_rows,
                   ("sparse_case", "parameter", "roi", "method", "metric", "n", "mean", "sample_sd"))
        _write_csv(staging / "sparse_case_improvement_summary.csv", improvement_rows,
                   ("sparse_case", "parameter", "roi", "metric", "baseline_method", "comparison_method", "direction", "formula", "n", "mean_percent", "sample_sd_percent"))
        write_immutable_json(staging / "manifest.json", {
            "schema": SUMMARY_SCHEMA, "result_tag": result_tag, "source_myocardial_roi_manifests": sources,
            "case_row_count": len(case_rows), "sparse_case_names": sorted({row["sparse_case"] for row in case_rows}),
            "metrics": {"rmse": "root mean squared error (ms)", "nrmse": "RMSE / pooled native-reference range",
                        "ncc": "zero-mean normalized cross-correlation", "mae": "mean absolute error (ms)"},
            "common_support": "manual anatomical ROI AND finite(reference) AND proposed-valid AND fit-first-valid AND finite(proposed, fit-first)",
            "fold_grouping": "terminal _foldN suffix is removed; subject/fold pairs are aggregation observations",
            "improvement": {"lower_is_better": "100 * (fit_first - proposed) / abs(fit_first)",
                            "higher_is_better": "100 * (proposed - fit_first) / abs(fit_first)"},
            "downstream_only": True, "no_reconstruction": True, "no_formal_metric": True, "immutable_output": True,
        })
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True); raise
    return output


def write_aha_sd_bullseyes(rows: Iterable[Mapping[str, Any]], output: str | Path, *,
                           manifest_updates: Mapping[str, Any] | None = None) -> Path:
    """Write one reference plus both method SD AHA16 bullseyes per case/parameter."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    grouped: dict[tuple[str, str, str, str], dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    all_sd: dict[str, list[float]] = defaultdict(list)
    for raw in rows:
        row = dict(raw); parameter = str(row["parameter"]).lower(); method = str(row["method"])
        if parameter not in ("t1", "t2") or method not in METHODS:
            raise ValueError("AHA SD rows require t1/t2 and proposed/fit_first methods")
        segment = int(row["segment"])
        if segment not in range(1, 17):
            raise ValueError("AHA SD rows require segments 1 through 16")
        key = (str(row["subject"]), str(row["case"]), parameter, method)
        prediction_sd, reference_sd = float(row["prediction_sd"]), float(row["reference_sd"])
        previous = grouped[key][segment].get("reference")
        if previous is not None and not np.isclose(previous, reference_sd, rtol=0, atol=1e-10):
            raise ValueError("reference SD differs between paired AHA methods")
        grouped[key][segment].update({"reference": reference_sd, method: prediction_sd})
        all_sd[parameter].extend((reference_sd, prediction_sd))
    if not grouped:
        raise ValueError("AHA SD backfill requires at least one row")
    ranges = {parameter: (0.0, max(values) if max(values) > 0 else 1.0) for parameter, values in all_sd.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        processed: set[tuple[str, str, str]] = set()
        for subject, case, parameter, _method in sorted(grouped):
            case_key = (subject, case, parameter)
            if case_key in processed:
                continue
            processed.add(case_key)
            by_method = {method: grouped[(subject, case, parameter, method)] for method in METHODS}
            if any(set(values) != set(range(1, 17)) for values in by_method.values()):
                raise ValueError(f"AHA SD rows must contain all 16 segments for {subject}/{case}/{parameter}")
            reference = {segment: by_method["proposed"][segment]["reference"] for segment in range(1, 17)}
            for segment in range(1, 17):
                if not np.isclose(reference[segment], by_method["fit_first"][segment]["reference"], rtol=0, atol=1e-10):
                    raise ValueError("reference SD differs between paired AHA methods")
            display_range = ranges[parameter]
            root = staging / subject / case
            render_single_bullseye(root / "reference" / f"{parameter}_aha16_sd.png", reference, parameter=parameter,
                                   title=f"{subject} {case} native reference {parameter.upper()} AHA16 SD",
                                   display_range=display_range, colorbar_label=f"{parameter.upper()} SD (ms)")
            for method in METHODS:
                prediction = {segment: by_method[method][segment][method] for segment in range(1, 17)}
                render_single_bullseye(root / method / f"{parameter}_aha16_sd.png", prediction, parameter=parameter,
                                       title=f"{subject} {case} {method.replace('_', '-')} {parameter.upper()} AHA16 SD",
                                       display_range=display_range, colorbar_label=f"{parameter.upper()} SD (ms)")
        manifest = {"schema": SD_SCHEMA,
            "display_ranges_ms": {key: list(value) for key, value in ranges.items()},
            "source": "completed AHA16 aha16_segment_statistics.csv prediction_sd/reference_sd columns",
            "algorithm": "historical AHA16 bullseye geometry; SD-only color scale shared by parameter",
            "downstream_only": True, "no_reconstruction": True, "no_formal_metric": True, "immutable_output": True}
        if manifest_updates:
            manifest.update(dict(manifest_updates))
        write_immutable_json(staging / "manifest.json", manifest)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True); raise
    return output


def build_aha_sd_backfill(*, analysis_root: str | Path, result_tag: str, output: str | Path) -> Path:
    root = Path(analysis_root)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    paths = sorted((root / "metrics" / "central").glob(f"*/*/{result_tag}/aha16/aha16_segment_statistics.csv"))
    if not paths:
        raise FileNotFoundError(f"no completed AHA16 statistics found for result tag {result_tag!r}")
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
        sources.append({"path": str(path.resolve()), "sha256": sha256(path)})
    return write_aha_sd_bullseyes(rows, output, manifest_updates={
        "result_tag": result_tag, "source_aha16_statistics": sources,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("summary", "aha-sd"), required=True)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--result-tag", default=RESULT_TAG_DEFAULT)
    args = parser.parse_args()
    if args.mode == "summary":
        print(build_sparse_myocardial_summary(analysis_root=args.analysis_root, result_tag=args.result_tag, output=args.output))
    else:
        print(build_aha_sd_backfill(analysis_root=args.analysis_root, result_tag=args.result_tag, output=args.output))


if __name__ == "__main__":
    main()
