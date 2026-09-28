"""Optional G0 adapter for D2-compatible observed/predicted weight-0 signals."""

from __future__ import annotations

import numpy as np


def extract_weight_zero_signal(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Select one observed/predicted weight-0 row per complete zero-based group."""

    required = {"group_idx", "weight_idx", "masks", "observed", "predicted"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"G0 signal artifact lacks {sorted(missing)}")
    groups, weights = np.asarray(data["group_idx"], dtype=np.int64), np.asarray(data["weight_idx"], dtype=np.int64)
    observed, predicted, masks = np.asarray(data["observed"]), np.asarray(data["predicted"]), np.asarray(data["masks"], bool)
    if observed.ndim != 3 or predicted.shape != observed.shape or masks.shape != observed.shape or groups.shape != (observed.shape[0],) or weights.shape != (observed.shape[0],):
        raise ValueError("G0 signal artifact has invalid observation-level schema.")
    expected = np.arange(int(groups.max()) + 1) if groups.size else np.empty(0, dtype=np.int64)
    if not np.array_equal(np.unique(groups), expected):
        raise ValueError("G0 signal group_idx must be contiguous zero-based.")
    indices = []
    for group in expected:
        rows = np.flatnonzero(groups == group)
        selected = rows[weights[rows] == 0]
        if selected.size != 1:
            raise ValueError(f"G0 signal group {group} must contain exactly one weight-0 row.")
        indices.append(int(selected[0]))
    selected = np.asarray(indices, dtype=int)
    return {"observed": observed[selected], "predicted": predicted[selected], "support": masks[selected] & np.isfinite(observed[selected]) & np.isfinite(predicted[selected]), "group_idx": expected}
