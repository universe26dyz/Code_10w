"""Q001 metric helpers: explicit true-pixel pooling alongside legacy D2 macro rows."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from trad.evaluation.scmr.metrics import agreement_metrics, signal_agreement_metrics


TRUE_POOLED = "true pooled pixels: concatenate all valid finite pixels before metric calculation"
LEGACY_MACRO = "N-weighted mean of per-observation metrics; retained only as a secondary diagnostic"


def true_pooled_rows(samples: Iterable[dict[str, Any]], keys: tuple[str, ...], *, domain: str) -> list[dict[str, Any]]:
    """Calculate metrics on concatenated support, never an average of nonlinear metrics."""

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[tuple(sample[key] for key in keys)].append(sample)
    result: list[dict[str, Any]] = []
    metric_fn = signal_agreement_metrics if domain == "signal" else agreement_metrics
    for identifier, members in sorted(grouped.items()):
        reference = np.concatenate([np.asarray(item["reference"])[np.asarray(item["support"], bool)] for item in members])
        prediction = np.concatenate([np.asarray(item["prediction"])[np.asarray(item["support"], bool)] for item in members])
        metric = metric_fn(reference, prediction, np.ones(reference.shape, dtype=bool), min_pixels=16)
        result.append(dict(zip(keys, identifier)) | {"metric_semantics": TRUE_POOLED, **metric})
    return result


def legacy_macro_rows(samples: Iterable[dict[str, Any]], keys: tuple[str, ...], *, domain: str) -> list[dict[str, Any]]:
    metric_fn = signal_agreement_metrics if domain == "signal" else agreement_metrics
    rows = [dict((key, sample[key]) for key in keys) | metric_fn(sample["reference"], sample["prediction"], sample["support"], min_pixels=16) for sample in samples]
    metric_names = ("bias_signal", "MAE_signal", "RMSE_signal", "NRMSE", "Pearson_r", "NCC", "SSIM") if domain == "signal" else ("bias_ms", "MAE_ms", "RMSE_ms", "NRMSE", "Pearson_r", "NCC", "SSIM")
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    result = []
    for identifier, members in sorted(grouped.items()):
        weights = np.asarray([float(row["N"]) for row in members])
        item = dict(zip(keys, identifier)) | {"N": int(weights.sum()), "aggregation": LEGACY_MACRO}
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in members])
            valid = np.isfinite(values) & (weights > 0)
            item[metric] = float(np.average(values[valid], weights=weights[valid])) if valid.any() else float("nan")
        result.append(item)
    for row in result:
        row["metric_semantics"] = LEGACY_MACRO
    return result


def strict_common_support(observed: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, baseline_mask: np.ndarray, candidate_mask: np.ndarray) -> np.ndarray:
    """Strict B6/Q001B support used for every primary matched signal comparison."""
    shapes = {np.shape(value) for value in (observed, baseline, candidate, baseline_mask, candidate_mask)}
    if len(shapes) != 1:
        raise ValueError("Strict Q001B support requires identical observed/predicted/mask shapes.")
    return np.asarray(baseline_mask, bool) & np.asarray(candidate_mask, bool) & np.isfinite(observed) & np.isfinite(baseline) & np.isfinite(candidate)
