from __future__ import annotations

import pytest


def test_sparse_summary_groups_folds_and_reports_paired_improvement() -> None:
    from paper_experiments.joint_vs_2dfit.aha.cyj11_postprocess import summarize_sparsity_rows

    rows = []
    for subject, case, proposed, baseline in (
        ("CYJ", "L25_fold0", 2.0, 4.0),
        ("CYJ", "L25_fold1", 4.0, 8.0),
        ("DYZ", "L25_fold0", 6.0, 12.0),
    ):
        for method, rmse in (("proposed", proposed), ("fit_first", baseline)):
            rows.append({
                "subject": subject, "case": case, "sparse_case": "L25", "parameter": "t1",
                "roi": "myocardium_full", "method": method, "rmse": rmse,
                "nrmse": rmse / 10.0, "ncc": 0.8 if method == "proposed" else 0.4,
                "mae": rmse / 2.0,
            })

    metric_rows, improvement_rows = summarize_sparsity_rows(rows)
    proposed_rmse = next(row for row in metric_rows if row["method"] == "proposed" and row["metric"] == "rmse")
    rmse_improvement = next(row for row in improvement_rows if row["metric"] == "rmse")
    ncc_improvement = next(row for row in improvement_rows if row["metric"] == "ncc")

    assert proposed_rmse["sparse_case"] == "L25"
    assert proposed_rmse["n"] == 3
    assert proposed_rmse["mean"] == pytest.approx(4.0)
    assert proposed_rmse["sample_sd"] == pytest.approx(2.0)
    assert rmse_improvement["mean_percent"] == pytest.approx(50.0)
    assert ncc_improvement["mean_percent"] == pytest.approx(100.0)


def test_aha_sd_writer_outputs_reference_and_method_bullseyes(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.cyj11_postprocess import write_aha_sd_bullseyes

    rows = []
    for method, value in (("proposed", 12.0), ("fit_first", 18.0)):
        for segment in range(1, 17):
            rows.append({
                "subject": "CYJ", "case": "L50_fold0", "method": method,
                "parameter": "t1", "segment": segment,
                "reference_sd": 10.0, "prediction_sd": value,
            })

    output = write_aha_sd_bullseyes(rows, tmp_path / "sd")
    assert (output / "CYJ/L50_fold0/reference/t1_aha16_sd.png").is_file()
    assert (output / "CYJ/L50_fold0/proposed/t1_aha16_sd.png").is_file()
    assert (output / "CYJ/L50_fold0/fit_first/t1_aha16_sd.pdf").is_file()
