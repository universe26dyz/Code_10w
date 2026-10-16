from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.phase2_clean.aggregate_real import bland_altman, summarize_r1_subjects
from paper_experiments.joint_vs_2dfit.phase2_clean.evaluation import heldout_quantitative_metrics, real_native_agreement
from paper_experiments.joint_vs_2dfit.phase2_clean.sparse import build_sparse_condition


def _observations(path: Path, group_ids: tuple[int, ...], *, missing_weight: bool = False) -> None:
    rows = [(group, weight) for group in group_ids for weight in range(10)]
    if missing_weight:
        rows.pop()
    count = len(rows)
    groups = np.asarray([row[0] for row in rows], dtype=np.int64)
    weights = np.asarray([row[1] for row in rows], dtype=np.int64)
    affine = np.repeat(np.eye(4)[None], count, axis=0)
    for index, group in enumerate(groups):
        affine[index, 2, 3] = float(group * 5)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        images=np.ones((count, 3, 4), np.float32), masks=np.ones((count, 3, 4), bool),
        group_idx=groups, weight_idx=weights, stack_idx=np.zeros(count, np.int64),
        acquisition_time_ms=np.arange(count, dtype=float), timing9_ms=np.tile(np.arange(1.0, 10.0), (count, 1)),
        affine_lps_rc=affine, pixel_spacing_rc_mm=np.ones((count, 2), float),
        slice_thickness_mm=np.full(count, 8.0), tr_ms=np.full(count, 2.61), vps=np.full(count, 87, np.int64),
    )


def _prepared(root: Path, *, missing_weight: bool = False) -> Path:
    for stack in ("sax", "2ch", "4ch"):
        _observations(root / "CYJ" / stack / "observations.npz", (4, 5, 6, 7), missing_weight=missing_weight and stack == "2ch")
    return root


def test_sparse_builder_keeps_atomic_groups_and_original_provenance(tmp_path: Path) -> None:
    source = _prepared(tmp_path / "prepared")
    result = build_sparse_condition(source, "CYJ", "L50", 0, tmp_path / "runs", git_sha="abc", audit_stack=lambda _: {"regular_gap": True, "gap_mm": 10.0, "group_count": 2})
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert manifest["status"] == "ELIGIBLE"
    assert manifest["stacks"]["2ch"]["retained_original_group_ids"] == [4, 6]
    assert manifest["stacks"]["2ch"]["heldout_original_group_ids"] == [5, 7]
    with np.load(result["prepared_root"] / "2ch" / "observations.npz") as data:
        assert np.unique(data["group_idx"]).tolist() == [0, 1]
        for group in (0, 1):
            assert np.sort(data["weight_idx"][data["group_idx"] == group]).tolist() == list(range(10))
    assert manifest["leakage_gate"] is True
    assert manifest["preprocessing_leakage_statement"].startswith("MIND operates")


def test_sparse_builder_rejects_incomplete_fingerprint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="weights 0..9"):
        build_sparse_condition(_prepared(tmp_path / "prepared", missing_weight=True), "CYJ", "L50", 0, tmp_path / "runs")


def test_sparse_builder_marks_too_few_retained_groups_ineligible(tmp_path: Path) -> None:
    source = _prepared(tmp_path / "prepared")
    result = build_sparse_condition(source, "CYJ", "L25", 3, tmp_path / "runs")
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert manifest["status"] == "INELIGIBLE"
    assert "fewer than two retained" in manifest["reason"]


def test_heldout_metrics_use_common_method_neutral_support() -> None:
    reference = np.array([[1000.0, 1100.0], [np.nan, 1200.0]])
    proposed = np.array([[1010.0, 1110.0], [999.0, np.nan]])
    baseline = np.array([[990.0, 1090.0], [999.0, 1210.0]])
    source_support = np.array([[True, True], [True, True]])
    reprojection = np.array([[True, True], [True, True]])
    report = heldout_quantitative_metrics(reference, proposed, baseline, source_support, reprojection)
    assert report["common_support_voxels"] == 2
    assert report["proposed"]["rmse"] == pytest.approx(10.0)
    assert report["fit_first"]["rmse"] == pytest.approx(10.0)
    shifted = heldout_quantitative_metrics(reference, proposed + 25.0, baseline, source_support, reprojection)
    assert shifted["common_support_voxels"] == report["common_support_voxels"]


def test_r0_metrics_means_bland_altman_and_subject_aggregation() -> None:
    reference = np.array([[1000.0, 1100.0]])
    prediction = np.array([[1010.0, 1110.0]])
    metric = real_native_agreement(reference, prediction, np.ones_like(reference, bool))
    assert metric["reference_mean_ms"] == 1050.0
    assert metric["prediction_mean_ms"] == 1060.0
    assert metric["mean_difference_ms"] == 10.0
    ba = bland_altman([1000.0, 1100.0, 1200.0, 1300.0], [1010.0, 1090.0, 1210.0, 1290.0])
    assert ba["n_subjects"] == 4 and ba["mean_difference_ms"] == 0.0
    summary = summarize_r1_subjects([
        {"subject": "CYJ", "method": "proposed", "level": "L50", "rmse": 10.0},
        {"subject": "CYJ", "method": "proposed", "level": "L50", "rmse": 14.0},
    ])
    assert summary[0]["rmse"] == 12.0 and summary[0]["fold_count"] == 2
