"""Discrete-orientation, one-to-one diagnostic matching for G0."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


def _orientations(values: np.ndarray) -> tuple[tuple[str, np.ndarray], ...]:
    return (
        ("IDENTITY", values),
        ("LR_FLIP", np.fliplr(values)),
        ("UD_FLIP", np.flipud(values)),
        ("ROT180", np.rot90(values, 2)),
        ("ROT90", np.rot90(values, 1)),
        ("ROT270", np.rot90(values, 3)),
        ("TRANSPOSE", values.T),
        ("ANTI_TRANSPOSE", np.rot90(values.T, 2)),
    )


def apply_orientation(values: np.ndarray, name: str) -> np.ndarray:
    """Apply a named G0 discrete orientation candidate."""

    options = dict(_orientations(values))
    if name not in options:
        raise ValueError(f"Unknown G0 orientation {name!r}.")
    return options[name]


@dataclass(frozen=True)
class StackAudit:
    assignment: np.ndarray
    score_matrix: np.ndarray
    orientation_matrix: np.ndarray
    rows: list[dict[str, object]]


def _pair_metric(native: np.ndarray, candidate: np.ndarray, native_support: np.ndarray, candidate_support: np.ndarray) -> tuple[float, float, int]:
    support = native_support & candidate_support & np.isfinite(native) & np.isfinite(candidate)
    count = int(support.sum())
    if count < 2:
        return float("-inf"), float("inf"), count
    reference, prediction = native[support].astype(float), candidate[support].astype(float)
    centered_reference, centered_prediction = reference - reference.mean(), prediction - prediction.mean()
    denominator = float(np.linalg.norm(centered_reference) * np.linalg.norm(centered_prediction))
    correlation = float(np.dot(centered_reference, centered_prediction) / denominator) if denominator else -1.0
    rmse = float(np.sqrt(np.mean((prediction - reference) ** 2)))
    reference_range = float(np.ptp(reference))
    relative_rmse = rmse / reference_range if reference_range > 0 else (0.0 if rmse == 0 else float("inf"))
    return correlation, relative_rmse, count


def audit_stack(
    native: np.ndarray, reconstructed: np.ndarray, native_support: np.ndarray, reconstructed_support: np.ndarray,
    *, low_margin_threshold: float = 0.05,
) -> StackAudit:
    """Score every native/reconstruction pair and preserve expected group indices in output rows."""

    native, reconstructed = np.asarray(native), np.asarray(reconstructed)
    native_support, reconstructed_support = np.asarray(native_support, bool), np.asarray(reconstructed_support, bool)
    if native.ndim != 3 or reconstructed.ndim != 3 or native_support.shape != native.shape or reconstructed_support.shape != reconstructed.shape:
        raise ValueError("G0 requires matching [group,row,column] values/support arrays within each source.")
    score = np.full((native.shape[0], reconstructed.shape[0]), -np.inf, dtype=float)
    rrmse = np.full_like(score, np.inf)
    counts = np.zeros_like(score, dtype=int)
    orientations = np.full(score.shape, "INCOMPATIBLE", dtype=object)
    for expected_group, reference in enumerate(native):
        for candidate_group, candidate in enumerate(reconstructed):
            best: tuple[float, float, int, str] | None = None
            for name, oriented in _orientations(candidate):
                oriented_support = dict(_orientations(reconstructed_support[candidate_group]))[name]
                if oriented.shape != reference.shape:
                    continue
                correlation, relative_rmse, count = _pair_metric(reference, oriented, native_support[expected_group], oriented_support)
                candidate_metric = (correlation, relative_rmse, count, name)
                if best is None or correlation > best[0] or (correlation == best[0] and relative_rmse < best[1]):
                    best = candidate_metric
            if best is not None:
                score[expected_group, candidate_group], rrmse[expected_group, candidate_group], counts[expected_group, candidate_group], orientations[expected_group, candidate_group] = best
    safe_cost = np.where(np.isfinite(score), -score, 1e9)
    assigned_rows, assigned_columns = linear_sum_assignment(safe_cost)
    assignment = np.full(native.shape[0], -1, dtype=int)
    assignment[assigned_rows] = assigned_columns
    rows: list[dict[str, object]] = []
    for expected_group, matched_group in enumerate(assignment):
        row_scores = score[expected_group]
        finite_scores = np.sort(row_scores[np.isfinite(row_scores)])[::-1]
        margin = float(finite_scores[0] - finite_scores[1]) if finite_scores.size > 1 else float("inf")
        best_orientation = str(orientations[expected_group, matched_group]) if matched_group >= 0 else "INCOMPATIBLE"
        if margin < low_margin_threshold:
            status = "LOW_MARGIN_AMBIGUOUS"
        elif matched_group != expected_group:
            status = "NONDIAGONAL_AUDIT_MATCH"
        elif best_orientation != "IDENTITY":
            status = "NONIDENTITY_ORIENTATION"
        else:
            status = "EXPECTED_DIAGONAL_IDENTITY"
        rows.append({
            "expected_group": expected_group,
            "content_matched_group": int(matched_group),
            "best_orientation": best_orientation,
            "correlation": float(score[expected_group, matched_group]),
            "relative_rmse": float(rrmse[expected_group, matched_group]),
            "valid_common_support_count": int(counts[expected_group, matched_group]),
            "assignment_margin": margin,
            "status": status,
        })
    return StackAudit(assignment=assignment, score_matrix=score, orientation_matrix=orientations, rows=rows)
