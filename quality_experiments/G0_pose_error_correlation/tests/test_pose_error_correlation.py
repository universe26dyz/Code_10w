"""Synthetic contracts for the read-only G0 pose-error association diagnostic."""

from __future__ import annotations

import pytest
from pathlib import Path

from quality_experiments.G0_pose_error_correlation.analysis import bh_fdr, join_pose_map_rows, leave_one_out_sensitivity, primary_statistics


def _maps() -> list[dict[str, object]]:
    return [
        {"subject_id": "CYJ", "stack": stack, "parameter": parameter, "expected_group": group,
         "content_matched_group": group, "best_orientation": "IDENTITY", "correlation": 0.9 - 0.1 * group,
         "relative_rmse": 0.1 + 0.1 * group, "valid_common_support_count": 100, "assignment_margin": 0.2}
        for stack in ("sax", "2ch") for parameter in ("T1", "T2") for group in range(3)
    ]


def _poses() -> list[dict[str, object]]:
    return [
        {"stack": stack, "group_idx": group, "global_group_idx": (0 if stack == "sax" else 10) + group,
         "translation_magnitude_mm": 0.5 + group, "rotation_magnitude_deg": 0.2 + group,
         "final_center_ras_mm": "[0,0,0]", "slice_normal_ras": "[0,0,1]", "neighbor_center_distance_mm": 1.0}
        for stack in ("sax", "2ch") for group in range(3)
    ]


def test_join_uses_stack_and_expected_group_not_global_group() -> None:
    """Joining map expected_group to global_group_idx must fail this test."""

    joined = join_pose_map_rows(_maps(), _poses())

    assert len(joined) == 12
    assert {row["global_group_idx"] for row in joined if row["stack"] == "2ch"} == {10, 11, 12}


def test_duplicate_or_missing_pose_group_is_rejected() -> None:
    """Accepting duplicate or absent pose records must fail this test."""

    with pytest.raises(ValueError, match="duplicate"):
        join_pose_map_rows(_maps(), _poses() + [_poses()[0]])
    with pytest.raises(ValueError, match="missing"):
        join_pose_map_rows(_maps(), _poses()[1:])


def test_non_diagonal_or_nonidentity_correspondence_is_rejected() -> None:
    """Using audit-best rather than official same-group metrics must fail this test."""

    maps = _maps(); maps[0]["content_matched_group"] = 2
    with pytest.raises(ValueError, match="content_matched_group"):
        join_pose_map_rows(maps, _poses())
    maps = _maps(); maps[0]["best_orientation"] = "LR_FLIP"
    with pytest.raises(ValueError, match="IDENTITY"):
        join_pose_map_rows(maps, _poses())


def test_missing_or_nonfinite_map_measurements_are_rejected() -> None:
    """The formal association must never silently drop an invalid map record."""

    maps = _maps(); del maps[0]["assignment_margin"]
    with pytest.raises(ValueError, match="lacks"):
        join_pose_map_rows(maps, _poses())
    maps = _maps(); maps[0]["relative_rmse"] = "nan"
    with pytest.raises(ValueError, match="finite"):
        join_pose_map_rows(maps, _poses())


def test_one_minus_correlation_and_monotonic_statistics_have_expected_direction() -> None:
    """Reversing either error endpoint or Spearman direction must fail this test."""

    joined = join_pose_map_rows(_maps(), _poses())
    assert joined[0]["one_minus_correlation"] == pytest.approx(1.0 - joined[0]["correlation"])
    stats = primary_statistics([row for row in joined if row["stack"] == "sax" and row["parameter"] == "T1"])
    translation_rrmse = next(row for row in stats if row["pose_metric"] == "translation_magnitude_mm" and row["error_metric"] == "relative_rmse")
    translation_one_minus = next(row for row in stats if row["pose_metric"] == "translation_magnitude_mm" and row["error_metric"] == "one_minus_correlation")
    assert translation_rrmse["spearman_rho"] == pytest.approx(1.0)
    assert translation_one_minus["spearman_rho"] == pytest.approx(1.0)


def test_inverse_monotonic_data_has_negative_spearman_rho() -> None:
    """Returning unsigned rank association must fail this test."""

    rows = join_pose_map_rows(_maps(), _poses())
    selected = [dict(row) for row in rows if row["stack"] == "sax" and row["parameter"] == "T1"]
    for index, row in enumerate(selected):
        row["relative_rmse"] = 3 - index
    stats = primary_statistics(selected)
    value = next(row for row in stats if row["pose_metric"] == "translation_magnitude_mm" and row["error_metric"] == "relative_rmse")
    assert value["spearman_rho"] == pytest.approx(-1.0)


def test_leave_one_out_identifies_the_most_influential_group() -> None:
    """Ignoring group-level sensitivity must fail this test."""

    rows = [
        {"group_idx": index, "global_group_idx": 20 + index, "translation_magnitude_mm": float(index), "relative_rmse": value}
        for index, value in enumerate((0.0, 1.0, 3.0, 2.0))
    ]
    result = leave_one_out_sensitivity(rows, "translation_magnitude_mm", "relative_rmse")

    assert result["full_rho"] == pytest.approx(0.8)
    assert result["most_influential_removed_group_idx"] == 0
    assert result["most_influential_removed_global_group_idx"] == 20


def test_bh_fdr_matches_a_known_toy_case() -> None:
    """A non-monotone Benjamini-Hochberg correction must fail this test."""

    assert bh_fdr([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])


def test_t1_t2_pose_fields_are_consistent_after_join() -> None:
    """Divergent pose values across T1/T2 rows must fail this test."""

    joined = join_pose_map_rows(_maps(), _poses())
    assert all(row["translation_magnitude_mm"] == 0.5 + row["group_idx"] for row in joined)
    maps = _maps(); maps[-1]["translation_magnitude_mm"] = 999.0
    # Map-side pose-like fields are ignored; canonical pose values must come only from the pose table.
    assert join_pose_map_rows(maps, _poses())[-1]["translation_magnitude_mm"] != 999.0


def test_discovery_requires_a_complete_corrected_v2_bundle(tmp_path) -> None:
    """Treating a stray G0 CSV or v1 bundle as formal v2 input must fail this test."""

    from quality_experiments.G0_pose_error_correlation.run_g0_pose_error_correlation import discover_local_v2_bundles

    stray = tmp_path / "G0_per_slice.csv"; stray.write_text("x\n", encoding="utf-8")
    assert discover_local_v2_bundles([tmp_path]) == []


def test_analysis_requires_all_six_g0_stack_parameter_pairs() -> None:
    """Figures and FDR family must not silently omit a G0 stack or parameter."""

    from quality_experiments.G0_pose_error_correlation.run_g0_pose_error_correlation import validate_analysis_design

    with pytest.raises(ValueError, match="six G0"):
        validate_analysis_design({("sax", "T1"): [{}]})
    validate_analysis_design({(stack, parameter): [{}] for stack in ("sax", "2ch", "4ch") for parameter in ("T1", "T2")})
