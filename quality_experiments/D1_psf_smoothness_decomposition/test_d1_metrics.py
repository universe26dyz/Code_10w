"""Targeted synthetic contracts for the D1 read-only diagnostic."""

from __future__ import annotations

import numpy as np
import pytest

from quality_experiments.D1_psf_smoothness_decomposition.io import ensure_empty_output
from quality_experiments.D1_psf_smoothness_decomposition.metrics import (
    d1_metrics,
    strict_paired_support,
)


def _observation_level_archive(tmp_path, *, groups: int = 2, shape: tuple[int, int] = (3, 4)):
    """Create the real exporter schema: one row per (spatial group, weight)."""

    group_idx = np.repeat(np.arange(groups, dtype=np.int64), 10)
    weight_idx = np.tile(np.arange(10, dtype=np.int64), groups)
    t1 = np.stack([np.full(shape, 100.0 * group + weight, dtype=np.float32) for group, weight in zip(group_idx, weight_idx)])
    t2 = t1 + 1000.0
    masks = np.ones_like(t1, dtype=bool)
    masks[weight_idx == 7, 0, 0] = False
    order = np.array([12, 1, 19, 5, 10, 8, 2, 17, 0, 15, 4, 11, 7, 14, 3, 18, 6, 13, 9, 16])[: group_idx.size]
    path = tmp_path / "t1_t2_native_plane_sax.npz"
    np.savez_compressed(path, t1_ms=t1[order], t2_ms=t2[order], masks=masks[order], group_idx=group_idx[order], weight_idx=weight_idx[order])
    return path


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


@pytest.mark.parametrize(("parameter", "offset"), [("T1", 0.0), ("T2", 1000.0)])
def test_observation_level_central_loader_selects_shuffled_unique_weight_zero_and_ands_all_masks(tmp_path, parameter, offset) -> None:
    """Treating the ten observation rows as one group row must fail this test."""

    from quality_experiments.D1_psf_smoothness_decomposition.io import _load_central_plane

    _observation_level_archive(tmp_path)
    plane = _load_central_plane(tmp_path, "sax", parameter, np.zeros((2, 3, 4), dtype=np.float32))

    assert np.array_equal(plane.values[:, 1, 1], np.array([0.0, 100.0]) + offset)
    assert not plane.support[:, 0, 0].any()
    assert plane.support[:, 1, 1].all()


def test_observation_level_central_loader_rejects_missing_weight(tmp_path) -> None:
    """Silently accepting nine weights for a group must fail this test."""

    path = _observation_level_archive(tmp_path)
    with np.load(path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    keep = np.arange(data["group_idx"].size) != 0
    np.savez_compressed(path, **{key: value[keep] if value.ndim else value for key, value in data.items()})

    from quality_experiments.D1_psf_smoothness_decomposition.io import _load_central_plane
    with pytest.raises(ValueError, match="10 observation|weight"):
        _load_central_plane(tmp_path, "sax", "T1", np.zeros((2, 3, 4), dtype=np.float32))


def test_observation_level_central_loader_rejects_duplicate_weight(tmp_path) -> None:
    """Replacing one required weight with a duplicate must fail this test."""

    path = _observation_level_archive(tmp_path)
    with np.load(path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    duplicate_row = np.flatnonzero((data["group_idx"] == 0) & (data["weight_idx"] == 9))[0]
    data["weight_idx"][duplicate_row] = 8
    np.savez_compressed(path, **data)

    from quality_experiments.D1_psf_smoothness_decomposition.io import _load_central_plane
    with pytest.raises(ValueError, match="weight"):
        _load_central_plane(tmp_path, "sax", "T1", np.zeros((2, 3, 4), dtype=np.float32))


def test_observation_level_central_loader_rejects_missing_or_extra_group_and_shape_mismatch(tmp_path) -> None:
    """Inferring groups from row order or resizing a map must fail this test."""

    path = _observation_level_archive(tmp_path)
    with np.load(path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    data["group_idx"][data["group_idx"] == 1] = 2
    np.savez_compressed(path, **data)

    from quality_experiments.D1_psf_smoothness_decomposition.io import _load_central_plane
    with pytest.raises(ValueError, match="zero-based"):
        _load_central_plane(tmp_path, "sax", "T1", np.zeros((2, 3, 4), dtype=np.float32))

    _observation_level_archive(tmp_path)
    with pytest.raises(ValueError, match="no resize"):
        _load_central_plane(tmp_path, "sax", "T1", np.zeros((2, 4, 4), dtype=np.float32))
