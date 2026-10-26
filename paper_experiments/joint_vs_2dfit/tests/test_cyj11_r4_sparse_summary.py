"""Contracts for five-subject r4 sparse-condition summaries."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _metrics(proposed: float, fit_first: float) -> dict[str, float]:
    return {
        "rmse": proposed,
        "relative_nrmse": proposed / 10.0,
        "ncc": proposed / 10.0,
        "mae": proposed / 2.0,
    }, {
        "rmse": fit_first,
        "relative_nrmse": fit_first / 10.0,
        "ncc": fit_first / 10.0,
        "mae": fit_first / 2.0,
    }


def _write_case(root: Path, subject: str, case: str, proposed: float, fit_first: float) -> None:
    p, f = _metrics(proposed, fit_first)
    payload = {
        "schema": "final_stratified_native_common_support_metrics/v1",
        "regions": {
            parameter: {"all_lax": {"status": "COMPLETE", "proposed": p, "fit_first": f}}
            for parameter in ("t1", "t2")
        },
    }
    path = root / "metrics" / "central" / subject / case / "r4_00d88b45c75f" / "aggregate_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _row(summary: dict, *, protocol: str, sparse_case: str, parameter: str, metric: str) -> dict:
    return next(
        row for row in summary["summaries"]
        if row["protocol"] == protocol
        and row["sparse_case"] == sparse_case
        and row["parameter"] == parameter
        and row["region"] == "all_lax"
        and row["metric"] == metric
    )


def test_summary_averages_folds_within_subject_before_five_subject_statistics(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_r4_sparse_summary import summarize_r4_sparse_cases

    root = tmp_path / "analysis"
    # Two folds each: per-subject proposed RMSE means are 3 and 5, not four
    # independent sample values.  This is the intended no-fold-overweighting rule.
    _write_case(root, "CYJ", "S2CH50_4CH100_fold0", 4.0, 8.0)
    _write_case(root, "CYJ", "S2CH50_4CH100_fold1", 2.0, 6.0)
    _write_case(root, "DYZ", "S2CH50_4CH100_fold0", 6.0, 10.0)
    _write_case(root, "DYZ", "S2CH50_4CH100_fold1", 4.0, 8.0)

    output = tmp_path / "summary"
    result = summarize_r4_sparse_cases(
        root,
        output,
        subjects=("CYJ", "DYZ"),
        protocols={"profile-v2": ("S2CH50_4CH100_fold0", "S2CH50_4CH100_fold1")},
    )

    assert result == output
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    rmse = _row(summary, protocol="profile-v2", sparse_case="S2CH50_4CH100", parameter="t1", metric="rmse")
    assert rmse["n_subjects"] == 2
    assert rmse["n_subject_case_folds"] == 4
    assert rmse["proposed_mean"] == pytest.approx(4.0)
    assert rmse["proposed_sample_sd"] == pytest.approx(2**0.5)
    assert rmse["fit_first_mean"] == pytest.approx(8.0)
    assert rmse["fit_first_sample_sd"] == pytest.approx(2**0.5)
    assert rmse["proposed_percent_improvement_vs_fit_first"] == pytest.approx(50.0)
    assert rmse["improvement_direction"] == "lower_is_better"

    ncc = _row(summary, protocol="profile-v2", sparse_case="S2CH50_4CH100", parameter="t1", metric="ncc")
    assert ncc["proposed_percent_improvement_vs_fit_first"] == pytest.approx(-50.0)
    assert ncc["improvement_direction"] == "higher_is_better"
    assert (output / "summary.csv").is_file()
    assert (output / "manifest.json").is_file()


def test_summary_keeps_legacy_and_profile_sparse_cases_separate(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_r4_sparse_summary import summarize_r4_sparse_cases

    root = tmp_path / "analysis"
    _write_case(root, "CYJ", "L50_fold0", 2.0, 4.0)
    _write_case(root, "DYZ", "L50_fold0", 4.0, 8.0)
    _write_case(root, "CYJ", "SAX100_2CH0_4CH0", 3.0, 6.0)
    _write_case(root, "DYZ", "SAX100_2CH0_4CH0", 5.0, 10.0)

    output = tmp_path / "summary"
    summarize_r4_sparse_cases(
        root,
        output,
        subjects=("CYJ", "DYZ"),
        protocols={"legacy-density": ("L50_fold0",), "profile-v2": ("SAX100_2CH0_4CH0",)},
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert _row(summary, protocol="legacy-density", sparse_case="L50", parameter="t2", metric="mae")["n_subject_case_folds"] == 2
    assert _row(summary, protocol="profile-v2", sparse_case="SAX100_2CH0_4CH0", parameter="t2", metric="mae")["n_subject_case_folds"] == 2


def test_summary_refuses_missing_expected_metric_input(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_r4_sparse_summary import summarize_r4_sparse_cases

    with pytest.raises(FileNotFoundError, match="aggregate metrics"):
        summarize_r4_sparse_cases(
            tmp_path / "analysis",
            tmp_path / "summary",
            subjects=("CYJ",),
            protocols={"legacy-density": ("L100_fold0",)},
        )
