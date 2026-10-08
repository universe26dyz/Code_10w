"""Strict common-support metric helpers for B6-versus-Q002 evaluation."""

from __future__ import annotations

import numpy as np

from quality_experiments.Q001_recon.metrics import true_pooled_rows


def strict_q002_common_support(reference: np.ndarray, b6: np.ndarray, q002: np.ndarray, b6_mask: np.ndarray, q002_mask: np.ndarray, reference_mask: np.ndarray) -> np.ndarray:
    shapes = {item.shape for item in (reference, b6, q002, b6_mask, q002_mask, reference_mask)}
    if len(shapes) != 1:
        raise ValueError("Q002 strict common support requires identical native-plane shapes.")
    return np.asarray(reference_mask, bool) & np.asarray(b6_mask, bool) & np.asarray(q002_mask, bool) & np.isfinite(reference) & np.isfinite(b6) & np.isfinite(q002)


def strict_fingerprint_common_support(b6_fingerprint: np.ndarray, q002_fingerprint: np.ndarray, b6_masks: np.ndarray, q002_masks: np.ndarray) -> np.ndarray:
    """Return pixels where every one of the native ten-weight vectors is valid."""

    shapes = {item.shape for item in (b6_fingerprint, q002_fingerprint, b6_masks, q002_masks)}
    if len(shapes) != 1 or np.asarray(b6_fingerprint).ndim != 3 or np.asarray(b6_fingerprint).shape[0] != 10:
        raise ValueError("Fingerprint common support requires matching [10,H,W] arrays.")
    return (np.asarray(b6_masks, bool) & np.asarray(q002_masks, bool) & np.isfinite(b6_fingerprint) & np.isfinite(q002_fingerprint)).all(axis=0)


__all__ = ["strict_q002_common_support", "strict_fingerprint_common_support", "true_pooled_rows"]
