from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.phase2_clean.aggregate_real import (
    bland_altman, load_r0_paired_records, paired_r0_comparison, pooled_r1_fold_metrics, summarize_r1_subjects,
)
from paper_experiments.joint_vs_2dfit.phase2_clean.evaluation import heldout_quantitative_metrics, paired_per_plane_records, real_native_agreement, real_native_paired_agreement
from paper_experiments.joint_vs_2dfit.phase2_clean.sparse import build_sparse_condition
from paper_experiments.joint_vs_2dfit.phase2_clean import proposed
from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import APPROVED_DECODER_SHA256
from paper_experiments.joint_vs_2dfit.phase2_clean.fit_first import build_independent_nesvor_commands


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


def test_l100_manifest_reuses_original_observations_without_copying_them(tmp_path: Path) -> None:
    source = _prepared(tmp_path / "prepared")
    result = build_sparse_condition(source, "CYJ", "L100", 0, tmp_path / "runs", git_sha="abc")
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert manifest["identity_source"] == "R0_full_prepared_observations"
    assert Path(result["prepared_root"]) == source / "CYJ"
    assert not (tmp_path / "runs" / "CYJ" / "L100_fold0" / "2ch" / "observations.npz").exists()


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
        {"subject": "CYJ", "method": "proposed", "level": "L50", "parameter": "t1_ms", "phase": 0, "pooled_rmse": 10.0},
        {"subject": "CYJ", "method": "proposed", "level": "L50", "parameter": "t1_ms", "phase": 1, "pooled_rmse": 14.0},
    ])
    assert summary[0]["rmse"] == 12.0 and summary[0]["fold_count"] == 2


def test_r0_paired_evaluation_uses_identical_common_myocardium_support() -> None:
    reference = np.array([[1000.0, 1100.0], [1200.0, 1300.0]])
    proposed = reference + 10.0
    baseline = reference - 10.0
    myocardium = np.ones_like(reference, bool)
    proposed_valid = np.array([[True, True], [True, False]])
    baseline_valid = np.array([[True, True], [False, True]])
    report = real_native_paired_agreement(reference, proposed, baseline, myocardium, proposed_valid, baseline_valid)
    assert report["common_support_voxels"] == 2
    assert report["proposed"]["support_voxels"] == report["fit_first"]["support_voxels"] == 2
    assert report["proposed"]["reference_mean_ms"] == report["fit_first"]["reference_mean_ms"] == 1050.0


def test_r0_zero_common_support_plane_is_retained_without_metric_key_error() -> None:
    rows = [
        {"group_index": 0, "proposed": {"rmse": 1.0}, "fit_first": {"rmse": 2.0}},
        {"group_index": 1, "status": "no_common_myocardium_support", "support_voxels": 0, "coverage_fraction": 0.0},
    ]
    proposed = paired_per_plane_records(rows, "proposed")
    baseline = paired_per_plane_records(rows, "fit_first")
    assert proposed[1]["status"] == baseline[1]["status"] == "no_common_myocardium_support"
    assert proposed[1]["support_voxels"] == baseline[1]["support_voxels"] == 0


def test_r0_aggregation_rejects_legacy_individual_method_json(tmp_path: Path) -> None:
    (tmp_path / "CYJ_proposed.json").write_text("{}")
    with pytest.raises(ValueError, match="paired common-support"):
        load_r0_paired_records(tmp_path)


def test_proposed_command_uses_current_adapter_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proposed, "sha256", lambda _: APPROVED_DECODER_SHA256)
    command = proposed.build_proposed_command("/prepared", "/decoder.pth", "/b6.pt", "/output")
    assert shlex.split(command)[0] == sys.executable


def test_fit_first_commands_disable_default_700_rescaling_and_save_models() -> None:
    commands = build_independent_nesvor_commands("/inputs", "/output", nesvor_executable="/historical/nesvor")
    assert len(commands) == 2
    for parameter, scale, command in zip(("t1", "t2"), ("2500", "200"), commands):
        assert "--output-intensity-mean 0" in command
        assert f"--output-model /output/{parameter}/model.pt" in command
        assert "--output-intensity-mean 700" not in command
        assert f"--scale {scale}" in command
        assert "--registration stack" in command and "--n-iter 10000" in command


def test_r1_subject_uses_equal_fold_weighting_not_plane_weighting() -> None:
    planes = [
        {"subject": "CYJ", "method": "proposed", "level": "L50", "parameter": "t1_ms", "phase": 0, "support_voxels": 1, "sum_squared_error": 100.0, "sum_absolute_error": 10.0, "sum_signed_error": 10.0},
        *[{"subject": "CYJ", "method": "proposed", "level": "L50", "parameter": "t1_ms", "phase": 1, "support_voxels": 1, "sum_squared_error": 900.0, "sum_absolute_error": 30.0, "sum_signed_error": 30.0} for _ in range(3)],
    ]
    folds = pooled_r1_fold_metrics(planes)
    assert [row["pooled_rmse"] for row in folds] == [10.0, 30.0]
    subject = summarize_r1_subjects(folds)
    assert subject[0]["rmse_mean_over_folds"] == 20.0
    assert subject[0]["fold_values"] == [10.0, 30.0]


def test_r0_paired_deltas_require_matched_subjects_and_reference_means() -> None:
    records = []
    for subject, proposed, baseline in (("CYJ", 10.0, 12.0), ("DYZ", 20.0, 24.0), ("HHZ", 30.0, 29.0), ("HJL", 40.0, 45.0)):
        for method, rmse in (("proposed", proposed), ("fit_first", baseline)):
            records.append({"subject": subject, "method": method, "parameter": "t1_ms", "rmse": rmse, "mae": rmse / 2, "bias": 0.0, "reference_mean_ms": 1000.0, "prediction_mean_ms": 1000.0 + rmse})
    paired, summary = paired_r0_comparison(records)
    assert [row["delta_rmse"] for row in paired] == [-2.0, -4.0, 1.0, -5.0]
    assert summary["t1_ms"]["mean_delta"] == pytest.approx(-2.5)
    records[-1]["reference_mean_ms"] = 1001.0
    with pytest.raises(ValueError, match="reference mean mismatch"):
        paired_r0_comparison(records)
