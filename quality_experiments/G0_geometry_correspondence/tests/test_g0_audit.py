"""Synthetic contracts for the read-only G0 geometry audit."""

from __future__ import annotations

import numpy as np
import pytest
import subprocess
import sys
from pathlib import Path

from quality_experiments.G0_geometry_correspondence.correspondence import audit_stack
from quality_experiments.G0_geometry_correspondence.pose_audit import pose_delta_rows
from quality_experiments.G0_geometry_correspondence.signal_adapter import extract_weight_zero_signal


def _maps(count: int = 3, shape: tuple[int, int] = (5, 7)) -> np.ndarray:
    rows, cols = np.indices(shape)
    return np.stack([100.0 * group + 3.0 * rows + 7.0 * cols + group * rows * cols for group in range(count)]).astype(np.float32)


def test_identity_assignment_is_diagonal_and_keeps_official_expected_pairing() -> None:
    """Removing the diagonal correspondence must fail this test."""

    reference = _maps()
    audit = audit_stack(reference, reference, np.ones_like(reference, bool), np.ones_like(reference, bool))

    assert np.array_equal(audit.assignment, np.arange(3))
    assert all(row["expected_group"] == index and row["content_matched_group"] == index for index, row in enumerate(audit.rows))


def test_shuffled_candidates_are_recovered_by_one_to_one_assignment() -> None:
    """Replacing Hungarian matching with row order must fail this test."""

    reference = _maps()
    candidates = reference[[2, 0, 1]]
    audit = audit_stack(reference, candidates, np.ones_like(reference, bool), np.ones_like(candidates, bool))

    assert np.array_equal(audit.assignment, np.array([1, 2, 0]))
    assert len(set(audit.assignment.tolist())) == 3


def test_reversed_order_and_discrete_flip_are_diagnosed_without_rewriting_expected_group() -> None:
    """Silently applying the diagnostic pairing or orientation must fail this test."""

    reference = _maps(2)
    candidates = np.flip(reference[::-1], axis=2)
    audit = audit_stack(reference, candidates, np.ones_like(reference, bool), np.ones_like(candidates, bool))

    assert np.array_equal(audit.assignment, np.array([1, 0]))
    assert all(row["expected_group"] == index for index, row in enumerate(audit.rows))
    assert all(row["best_orientation"] == "LR_FLIP" for row in audit.rows)


def test_90_degree_orientation_is_detected_when_shape_is_transpose_compatible() -> None:
    """Dropping 90-degree candidates must fail this test."""

    reference = _maps(1, (5, 5))
    candidates = np.rot90(reference, 1, axes=(1, 2))
    audit = audit_stack(reference, candidates, np.ones_like(reference, bool), np.ones_like(candidates, bool))

    assert audit.rows[0]["best_orientation"] == "ROT270"


def test_low_assignment_margin_is_flagged() -> None:
    """Reporting high confidence for tied candidates must fail this test."""

    reference = _maps(2)
    candidates = np.repeat(reference[:1], 2, axis=0)
    audit = audit_stack(reference, candidates, np.ones_like(reference, bool), np.ones_like(candidates, bool), low_margin_threshold=0.05)

    assert any(row["status"] == "LOW_MARGIN_AMBIGUOUS" for row in audit.rows)


def test_weight_zero_signal_adapter_selects_only_weight_zero_rows() -> None:
    """Using an arbitrary signal weight instead of HB1/weight 0 must fail this test."""

    groups = np.repeat(np.arange(2), 10)
    weights = np.tile(np.arange(10), 2)
    observed = np.stack([np.full((2, 3), 100 * group + weight) for group, weight in zip(groups, weights)]).astype(np.float32)
    result = extract_weight_zero_signal({"group_idx": groups, "weight_idx": weights, "masks": np.ones_like(observed, bool), "observed": observed, "predicted": observed + 1})

    assert np.array_equal(result["observed"][:, 0, 0], np.array([0.0, 100.0]))


def test_known_pose_translation_rotation_and_neighbors_are_measured() -> None:
    """Ignoring physical pose deltas must fail this test."""

    initial = np.zeros((2, 6), dtype=float)
    final = np.array([[1, 0, 0, 0, 0, np.pi / 2], [0, 3, 0, 0, 0, 0]], dtype=float)
    rows = pose_delta_rows("sax", initial, final)

    assert rows[0]["translation_magnitude_mm"] == pytest.approx(1.0)
    assert rows[0]["rotation_magnitude_deg"] == pytest.approx(90.0)
    assert rows[0]["neighbor_center_distance_mm"] == pytest.approx(2.0)


def test_overwrite_refusal_and_heterogeneous_stack_counts(tmp_path) -> None:
    """Allowing overwrite or forcing cross-stack dimensions to agree must fail this test."""

    from quality_experiments.G0_geometry_correspondence.run_g0_geometry_audit import ensure_empty_output

    output = tmp_path / "existing"; output.mkdir(); (output / "old").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        ensure_empty_output(output)
    assert audit_stack(_maps(2, (4, 5)), _maps(2, (4, 5)), np.ones((2, 4, 5), bool), np.ones((2, 4, 5), bool)).assignment.size == 2
    assert audit_stack(_maps(3, (6, 4)), _maps(3, (6, 4)), np.ones((3, 6, 4), bool), np.ones((3, 6, 4), bool)).assignment.size == 3


def test_direct_runner_cli_help_is_importable_from_repository_root() -> None:
    """A direct required CLI invocation must not depend on package launch syntax."""

    root = Path(__file__).resolve().parents[3]
    script = root / "quality_experiments/G0_geometry_correspondence/run_g0_geometry_audit.py"
    result = subprocess.run([sys.executable, str(script), "--help"], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
