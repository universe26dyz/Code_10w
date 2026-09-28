"""Deterministic D1 agreement and map-detail metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from trad.evaluation.scmr.metrics import agreement_metrics


DETAIL_METRIC_COLUMNS = (
    "gradient_magnitude_ratio",
    "gradient_correlation",
    "edge_RMSE_ms",
    "high_frequency_energy_ratio",
)


def strict_paired_support(
    native_valid: np.ndarray,
    prediction_supports: Iterable[np.ndarray],
    prediction_arrays: Iterable[np.ndarray],
) -> np.ndarray:
    """Return the explicit four-prediction D1 strict paired support."""

    result = np.asarray(native_valid, dtype=bool).copy()
    for support, values in zip(prediction_supports, prediction_arrays):
        support = np.asarray(support, dtype=bool)
        values = np.asarray(values)
        if support.shape != result.shape or values.shape != result.shape:
            raise ValueError("D1 strict paired support requires equal 2-D shapes.")
        result &= support & np.isfinite(values)
    return result


def _interior_gradient(values: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference gradient magnitude where all four neighbours are valid."""

    values = np.asarray(values, dtype=np.float64)
    support = np.asarray(support, dtype=bool) & np.isfinite(values)
    if values.ndim != 2 or support.shape != values.shape:
        raise ValueError("Gradient metrics require matching 2-D values and support.")
    gradient = np.full(values.shape, np.nan, dtype=np.float64)
    interior = np.zeros(values.shape, dtype=bool)
    if min(values.shape) < 3:
        return gradient, interior
    interior[1:-1, 1:-1] = (
        support[1:-1, 1:-1]
        & support[:-2, 1:-1]
        & support[2:, 1:-1]
        & support[1:-1, :-2]
        & support[1:-1, 2:]
    )
    row_gradient = (values[2:, 1:-1] - values[:-2, 1:-1]) / 2.0
    col_gradient = (values[1:-1, 2:] - values[1:-1, :-2]) / 2.0
    gradient[1:-1, 1:-1] = np.hypot(row_gradient, col_gradient)
    return gradient, interior


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 2:
        return float("nan")
    first, second = first - first.mean(), second - second.mean()
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator else float("nan")


def _high_frequency_energy(values: np.ndarray, support: np.ndarray) -> float:
    """Windowed high-frequency energy using the fixed r >= 0.25 cycles/pixel band."""

    rows, cols = np.where(support)
    if not rows.size:
        return float("nan")
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    roi_values, roi_support = np.asarray(values, dtype=np.float64)[r0:r1, c0:c1], support[r0:r1, c0:c1]
    if min(roi_values.shape) < 2:
        return float("nan")
    fill = float(roi_values[roi_support].mean())
    filled = np.where(roi_support, roi_values, fill) - fill
    window = np.outer(np.hanning(roi_values.shape[0]), np.hanning(roi_values.shape[1]))
    power = np.abs(np.fft.fft2(filled * window)) ** 2
    frequencies = np.meshgrid(np.fft.fftfreq(roi_values.shape[0]), np.fft.fftfreq(roi_values.shape[1]), indexing="ij")
    radial_frequency = np.hypot(frequencies[0], frequencies[1])
    return float(power[radial_frequency >= 0.25].sum())


def d1_metrics(native: np.ndarray, prediction: np.ndarray, support: np.ndarray) -> dict[str, float | int]:
    """Agreement plus D1 detail metrics on one 2-D paired support."""

    native, prediction = np.asarray(native, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    support = np.asarray(support, dtype=bool) & np.isfinite(native) & np.isfinite(prediction)
    if native.ndim != 2 or prediction.shape != native.shape or support.shape != native.shape:
        raise ValueError("D1 metrics require matching 2-D native/prediction/support arrays.")
    result = agreement_metrics(native, prediction, support)
    native_gradient, native_interior = _interior_gradient(native, support)
    prediction_gradient, prediction_interior = _interior_gradient(prediction, support)
    interior = native_interior & prediction_interior
    native_detail, prediction_detail = native_gradient[interior], prediction_gradient[interior]
    if native_detail.size == 0:
        result.update({name: float("nan") for name in DETAIL_METRIC_COLUMNS})
        return result
    native_mean = float(native_detail.mean())
    edge_threshold = float(np.percentile(native_detail, 75.0))
    edge = interior & (native_gradient >= edge_threshold)
    difference = prediction - native
    result.update({
        "gradient_magnitude_ratio": float(prediction_detail.mean() / native_mean) if native_mean > 0 else float("nan"),
        "gradient_correlation": _correlation(native_detail, prediction_detail),
        "edge_RMSE_ms": float(np.sqrt(np.mean(difference[edge] ** 2))) if edge.any() else float("nan"),
        "high_frequency_energy_ratio": _high_frequency_ratio(native, prediction, support),
    })
    return result


def _high_frequency_ratio(native: np.ndarray, prediction: np.ndarray, support: np.ndarray) -> float:
    native_energy, prediction_energy = _high_frequency_energy(native, support), _high_frequency_energy(prediction, support)
    if not np.isfinite(native_energy) or native_energy <= 0 or not np.isfinite(prediction_energy):
        return float("nan")
    return float(prediction_energy / native_energy)
