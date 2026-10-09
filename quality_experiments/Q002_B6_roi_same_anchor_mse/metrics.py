"""Strict common-support metric helpers for B6-versus-Q002 evaluation."""

from __future__ import annotations

import numpy as np

from quality_experiments.Q001_recon.metrics import true_pooled_rows


def strict_q002_common_support(reference: np.ndarray, b6: np.ndarray, q002: np.ndarray, b6_mask: np.ndarray, q002_mask: np.ndarray, reference_mask: np.ndarray) -> np.ndarray:
    shapes = {item.shape for item in (reference, b6, q002, b6_mask, q002_mask, reference_mask)}
    if len(shapes) != 1:
        raise ValueError("Q002 strict common support requires identical native-plane shapes.")
    return np.asarray(reference_mask, bool) & np.asarray(b6_mask, bool) & np.asarray(q002_mask, bool) & np.isfinite(reference) & np.isfinite(b6) & np.isfinite(q002)


def strict_q002_three_way_common_support(reference: np.ndarray, b6: np.ndarray, q002_64: np.ndarray, q002_s640: np.ndarray, b6_mask: np.ndarray, q002_64_mask: np.ndarray, q002_s640_mask: np.ndarray, reference_mask: np.ndarray) -> np.ndarray:
    """Support for a fair B6/Q002-64/Q002-S640 comparison on one native plane."""

    shapes = {item.shape for item in (reference, b6, q002_64, q002_s640, b6_mask, q002_64_mask, q002_s640_mask, reference_mask)}
    if len(shapes) != 1:
        raise ValueError("Q002 three-way common support requires identical native-plane shapes.")
    return np.asarray(reference_mask, bool) & np.asarray(b6_mask, bool) & np.asarray(q002_64_mask, bool) & np.asarray(q002_s640_mask, bool) & np.isfinite(reference) & np.isfinite(b6) & np.isfinite(q002_64) & np.isfinite(q002_s640)


def strict_fingerprint_common_support(b6_fingerprint: np.ndarray, q002_fingerprint: np.ndarray, b6_masks: np.ndarray, q002_masks: np.ndarray) -> np.ndarray:
    """Return pixels where every one of the native ten-weight vectors is valid."""

    shapes = {item.shape for item in (b6_fingerprint, q002_fingerprint, b6_masks, q002_masks)}
    if len(shapes) != 1 or np.asarray(b6_fingerprint).ndim != 3 or np.asarray(b6_fingerprint).shape[0] != 10:
        raise ValueError("Fingerprint common support requires matching [10,H,W] arrays.")
    return (np.asarray(b6_masks, bool) & np.asarray(q002_masks, bool) & np.isfinite(b6_fingerprint) & np.isfinite(q002_fingerprint)).all(axis=0)


def strict_fingerprint_three_way_common_support(b6_fingerprint: np.ndarray, q002_64_fingerprint: np.ndarray, q002_s640_fingerprint: np.ndarray, b6_masks: np.ndarray, q002_64_masks: np.ndarray, q002_s640_masks: np.ndarray) -> np.ndarray:
    """Pixel support shared by every ordered ten-weight vector from all methods."""

    shapes = {item.shape for item in (b6_fingerprint, q002_64_fingerprint, q002_s640_fingerprint, b6_masks, q002_64_masks, q002_s640_masks)}
    if len(shapes) != 1 or np.asarray(b6_fingerprint).ndim != 3 or np.asarray(b6_fingerprint).shape[0] != 10:
        raise ValueError("Three-way fingerprint support requires matching [10,H,W] arrays.")
    return (np.asarray(b6_masks, bool) & np.asarray(q002_64_masks, bool) & np.asarray(q002_s640_masks, bool) & np.isfinite(b6_fingerprint) & np.isfinite(q002_64_fingerprint) & np.isfinite(q002_s640_fingerprint)).all(axis=0)


def fingerprint_cosine_summary(predicted: np.ndarray, observed: np.ndarray, support: np.ndarray, *, epsilon: float = 1e-8) -> dict[str, float | int]:
    """Observed fingerprint-shape fidelity for ordered [10,H,W] vectors."""
    if predicted.shape != observed.shape or predicted.ndim != 3 or predicted.shape[0] != 10 or support.shape != predicted.shape[1:]:
        raise ValueError("Fingerprint cosine requires matching [10,H,W] prediction/observed and [H,W] support.")
    dot = (predicted * observed).sum(axis=0)
    pred_norm_raw = np.linalg.norm(predicted, axis=0)
    obs_norm_raw = np.linalg.norm(observed, axis=0)
    near_zero = (pred_norm_raw < epsilon) | (obs_norm_raw < epsilon)
    cosine = dot / (np.maximum(pred_norm_raw, epsilon) * np.maximum(obs_norm_raw, epsilon))
    values = cosine[np.asarray(support, bool)]
    return {"mean_cosine": float(values.mean()) if values.size else float("nan"), "median_cosine": float(np.median(values)) if values.size else float("nan"), "support_N": int(values.size), "near_zero_norm_count": int((near_zero & np.asarray(support, bool)).sum())}


__all__ = ["strict_q002_common_support", "strict_q002_three_way_common_support", "strict_fingerprint_common_support", "strict_fingerprint_three_way_common_support", "fingerprint_cosine_summary", "true_pooled_rows"]
