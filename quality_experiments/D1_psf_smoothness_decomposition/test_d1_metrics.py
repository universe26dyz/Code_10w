"""Targeted synthetic contracts for the D1 read-only diagnostic."""

from __future__ import annotations

import numpy as np
import pytest

from quality_experiments.D1_psf_smoothness_decomposition.io import ensure_empty_output
from quality_experiments.D1_psf_smoothness_decomposition.metrics import (
    d1_metrics,
    strict_paired_support,
)


def test_identical_maps_have_perfect_agreement_and_detail_metrics() -> None:
    """A regression that changes a perfect-map metric must fail this test."""

    native = np.arange(64, dtype=np.float64).reshape(8, 8)
    support = np.ones_like(native, dtype=bool)

    result = d1_metrics(native, native.copy(), support)

    assert result["RMSE_ms"] == pytest.approx(0.0)
    assert result["gradient_magnitude_ratio"] == pytest.approx(1.0)
    assert result["gradient_correlation"] == pytest.approx(1.0)
    assert result["edge_RMSE_ms"] == pytest.approx(0.0)
    assert result["high_frequency_energy_ratio"] == pytest.approx(1.0)


def test_blurred_prediction_loses_gradient_and_high_frequency_energy() -> None:
    """Removing the detail calculation or reversing its ratio must fail."""

    rows, cols = np.indices((33, 33))
    native = 0.5 + 0.5 * np.sin(2.0 * np.pi * (rows + cols) / 5.0)
    prediction = np.full_like(native, 0.5)
    support = np.ones_like(native, dtype=bool)

    result = d1_metrics(native, prediction, support)

    assert result["gradient_magnitude_ratio"] < 1.0
    assert result["high_frequency_energy_ratio"] < 1.0


def test_strict_paired_support_requires_every_input_support_and_finite_value() -> None:
    """Dropping one of the four prediction supports must fail this test."""

    native_valid = np.array([[True, True], [True, False]])
    central_bloch = np.array([[True, True], [False, True]])
    psf_bloch = np.array([[True, False], [True, True]])
    central_mlp = np.array([[True, True], [True, True]])
    psf_mlp = np.array([[True, True], [True, True]])
    arrays = [
        np.array([[1.0, 1.0], [1.0, 1.0]]),
        np.array([[1.0, 1.0], [1.0, 1.0]]),
        np.array([[1.0, np.nan], [1.0, 1.0]]),
        np.array([[1.0, 1.0], [1.0, 1.0]]),
    ]

    result = strict_paired_support(
        native_valid,
        (central_bloch, psf_bloch, central_mlp, psf_mlp),
        arrays,
    )

    assert np.array_equal(result, np.array([[True, False], [False, False]]))


def test_nonempty_output_directory_is_refused(tmp_path) -> None:
    """Removing output overwrite protection must fail this test."""

    output = tmp_path / "D1"
    output.mkdir()
    (output / "existing.txt").write_text("preserve me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty"):
        ensure_empty_output(output)


def test_heterogeneous_stack_shapes_are_kept_separate() -> None:
    """Accidentally stacking SAX/2CH/4CH arrays must fail this test."""

    from quality_experiments.D1_psf_smoothness_decomposition.io import validate_stack_shapes

    stacks = {
        "sax": np.zeros((2, 8, 9), dtype=np.float32),
        "2ch": np.zeros((1, 7, 11), dtype=np.float32),
        "4ch": np.zeros((3, 6, 10), dtype=np.float32),
    }
    validate_stack_shapes(stacks)


def test_render_comparison_uses_existing_parameter_display_ranges(tmp_path) -> None:
    """Using a lower-case range key instead of the baseline contract must fail."""

    from quality_experiments.D1_psf_smoothness_decomposition.figures import render_comparison

    native = np.arange(16, dtype=float).reshape(4, 4)
    target = tmp_path / "comparison.png"
    render_comparison(
        parameter="T1", stack="sax", group=0, native=native, central_bloch=native,
        psf_bloch=native, central_mlp=native, psf_mlp=native,
        support=np.ones_like(native, dtype=bool), target=target,
    )
    assert target.is_file()


def test_increment_summary_weights_psf_minus_central_values_by_support_size() -> None:
    """Treating increment rows as ordinary metrics must fail this test."""

    from quality_experiments.D1_psf_smoothness_decomposition.run_d1_psf_smoothness_decomposition import _grouped_summaries

    rows = [
        {"parameter": "T1", "method": "Bloch", "N": 10, "delta_psf_minus_central_RMSE_ms": 2.0},
        {"parameter": "T1", "method": "Bloch", "N": 30, "delta_psf_minus_central_RMSE_ms": 6.0},
    ]
    result = _grouped_summaries(rows, ("parameter", "method"), ("delta_psf_minus_central_RMSE_ms",))

    assert result[0]["N"] == 40
    assert result[0]["delta_psf_minus_central_RMSE_ms"] == pytest.approx(5.0)


def test_output_layout_records_a_command_and_required_result_directories(tmp_path) -> None:
    """Omitting a required D1 result-schema artifact must fail this test."""

    from quality_experiments.D1_psf_smoothness_decomposition.io import prepare_output_layout

    output = prepare_output_layout(tmp_path / "D1", ["python", "runner.py", "--output", "D1"])

    assert (output / "commands" / "command.txt").is_file()
    assert (output / "logs" / "run.log").is_file()
    for relative in ("metrics", "figures/representative", "figures/all_slices", "figures/myocardium", "artifacts"):
        assert (output / relative).is_dir()
