"""Masked quantitative-agreement metrics with no background-zero leakage."""

from __future__ import annotations

import numpy as np


METRIC_COLUMNS = ("N", "bias_ms", "MAE_ms", "RMSE_ms", "NRMSE", "Pearson_r", "NCC", "SSIM")


def agreement_metrics(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray, *, min_pixels: int = 16) -> dict[str, float | int]:
    """Compute paired metrics only over valid finite pixels in ``mask``.

    NRMSE is RMSE divided by the in-mask reference range.  NCC uses the
    conventional mean-centred definition, while Pearson r is reported
    explicitly to make the intended interpretation unambiguous.
    """

    valid = np.asarray(mask, dtype=bool) & np.isfinite(reference) & np.isfinite(prediction)
    a = np.asarray(reference, dtype=float)[valid]
    b = np.asarray(prediction, dtype=float)[valid]
    n = int(a.size)
    empty = {name: float("nan") for name in METRIC_COLUMNS if name != "N"}
    if n < min_pixels:
        return {"N": n, **empty}
    difference = b - a
    a_centered, b_centered = a - a.mean(), b - b.mean()
    denominator = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    correlation = float(np.dot(a_centered, b_centered) / denominator) if denominator else float("nan")
    reference_range = float(np.ptp(a))
    rmse = float(np.sqrt(np.mean(difference**2)))
    ssim = _roi_bounded_ssim(np.asarray(reference, dtype=float), np.asarray(prediction, dtype=float), valid, a, b) \
        if valid.ndim == 2 else float("nan")
    return {
        "N": n,
        "bias_ms": float(difference.mean()),
        "MAE_ms": float(np.abs(difference).mean()),
        "RMSE_ms": rmse,
        "NRMSE": rmse / reference_range if reference_range > 0 else float("nan"),
        "Pearson_r": correlation,
        "NCC": correlation,
        "SSIM": ssim,
    }


def _roi_bounded_ssim(reference: np.ndarray, prediction: np.ndarray, valid: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Evaluate SSIM in the support bounding box without zero-filled background."""

    try:
        from skimage.metrics import structural_similarity
    except ImportError:  # SSIM remains optional when scikit-image is unavailable.
        return float("nan")
    rows, cols = np.where(valid)
    if not rows.size:
        return float("nan")
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    ref = np.full(reference.shape, float(a.mean()), dtype=float)
    pred = np.full(prediction.shape, float(b.mean()), dtype=float)
    ref[valid], pred[valid] = a, b
    ref, pred = ref[r0:r1, c0:c1], pred[r0:r1, c0:c1]
    data_range = float(max(np.ptp(a), np.ptp(b)))
    window = min(7, min(ref.shape))
    if window % 2 == 0:
        window -= 1
    if data_range <= 0 or window < 3:
        return float("nan")
    return float(structural_similarity(ref, pred, data_range=data_range, win_size=window))
