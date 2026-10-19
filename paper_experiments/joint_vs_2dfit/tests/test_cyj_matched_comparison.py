"""CPU contracts for the CYJ Q002S640/Fit-first matched comparison table."""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.phase2_clean import cyj_matched_comparison as comparison


def _metrics(proposed_rmse: float, fit_first_rmse: float) -> dict[str, object]:
    proposed_mae, fit_first_mae = proposed_rmse / 2, fit_first_rmse / 2
    proposed = {"voxel_count": 10, "coverage_fraction": 1.0, "rmse": proposed_rmse, "mae": proposed_mae, "bias": -1.0, "reference_mean_ms": 1000.0, "prediction_mean_ms": 999.0}
    fit_first = {"voxel_count": 10, "coverage_fraction": 1.0, "rmse": fit_first_rmse, "mae": fit_first_mae, "bias": 1.0, "reference_mean_ms": 1000.0, "prediction_mean_ms": 1001.0}
    return {"proposed": proposed, "fit_first": fit_first, "paired_deltas": {"delta_rmse_proposed_minus_fit_first": proposed_rmse - fit_first_rmse, "delta_mae_proposed_minus_fit_first": proposed_mae - fit_first_mae, "delta_bias_proposed_minus_fit_first": -2.0}}


def _volume(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.ones((1, 1, 1), np.float32), np.eye(4)), path)


def _result(root: Path, condition: str, endpoint: str, proposed_rmse: float, fit_first_rmse: float) -> None:
    output = root / f"{condition}_{endpoint}"
    output.mkdir(parents=True)
    regions = {parameter: {stack: {"l100_full": _metrics(proposed_rmse, fit_first_rmse)} for stack in ("2ch", "4ch", "all_lax")} for parameter in ("t1", "t2")}
    (output / "manifest.json").write_text(json.dumps({"subject": "CYJ", "condition": condition, "endpoint": endpoint}), encoding="utf-8")
    (output / "aggregate_metrics.json").write_text(json.dumps({"regions": regions}), encoding="utf-8")


def test_cyj_scope_has_l100_once_l50_twice_and_l25_four_times() -> None:
    assert comparison.CYJ_CASES == ("L100_fold0", "L50_fold0", "L50_fold1", "L25_fold0", "L25_fold1", "L25_fold2", "L25_fold3")


def test_l100_requires_the_r0_fit_first_root_to_be_reused(tmp_path: Path) -> None:
    r0, sparse = tmp_path / "r0", tmp_path / "r1"; r0.mkdir(); sparse.mkdir()
    with pytest.raises(ValueError, match="L100"):
        comparison.require_l100_r0_reuse("L100_fold0", r0, sparse)
    comparison.require_l100_r0_reuse("L100_fold0", r0, r0)


def test_q002s640_contract_preserves_training_k8_separately_from_output_endpoint(tmp_path: Path) -> None:
    root = tmp_path / "q002"; root.mkdir()
    (root / "experiment_manifest.json").write_text(json.dumps({"experiment_id": "Q002S640_B6_roi_same_anchor_mse", "training_psf_samples": 8}), encoding="utf-8")
    central = comparison.q002s640_provenance(root, "central")
    psf128 = comparison.q002s640_provenance(root, "psf128")
    assert central["training_route"] == "Q002S640_B6_roi_same_anchor_mse"
    assert central["training_psf"]["samples"] == 8
    assert central["export_psf"]["n_inference_samples"] == 1
    assert psf128["export_psf"]["n_inference_samples"] == 128
    assert psf128["export_psf"]["post_hoc_map_domain"] is True


def test_q002s640_contract_accepts_existing_route_nested_k8_provenance(tmp_path: Path) -> None:
    root = tmp_path / "q002"; root.mkdir()
    (root / "experiment_manifest.json").write_text(json.dumps({"experiment_id": "Q002S640_B6_roi_same_anchor_mse", "route": {"sampling": {"shared_training_psf_samples": 8}}}), encoding="utf-8")
    assert comparison.q002s640_provenance(root, "central")["training_psf"]["samples"] == 8


def test_q002s640_alias_carries_nested_k8_provenance_into_formal_output(tmp_path: Path) -> None:
    root = tmp_path / "q002"; root.mkdir(); (root / "model.pt").write_bytes(b"checkpoint")
    (root / "experiment_manifest.json").write_text(json.dumps({"experiment_id": "Q002S640_B6_roi_same_anchor_mse", "route": {"sampling": {"shared_training_psf_samples": 8}}}), encoding="utf-8")
    _volume(root / "T1_3D.nii.gz"); _volume(root / "T2_3D.nii.gz")
    output = comparison.formalize_q002s640_alias(subject="CYJ", condition="L100_fold0", source_root=root, endpoint="central", output=tmp_path / "alias")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["training_route"] == "Q002S640_B6_roi_same_anchor_mse"
    assert manifest["training_psf"]["training_psf_samples"] == 8
    assert manifest["export_psf"]["n_inference_samples"] == 1


def test_aggregation_uses_sample_sd_and_counts_proposed_wins(tmp_path: Path) -> None:
    _result(tmp_path / "results", "L50_fold0", "central", 1.0, 2.0)
    _result(tmp_path / "results", "L50_fold1", "central", 3.0, 2.0)
    output = comparison.aggregate_cyj_matched_results(tmp_path / "results", tmp_path / "summary")
    payload = json.loads((output / "summary.json").read_text())
    row = next(row for row in payload["summaries"] if row["endpoint"] == "central" and row["level"] == "L50" and row["parameter"] == "T1" and row["stack"] == "ALL_LAX")
    assert row["proposed_rmse_mean"] == pytest.approx(2.0)
    assert row["proposed_rmse_sample_sd"] == pytest.approx(2**0.5)
    assert row["fit_first_rmse_sample_sd"] == pytest.approx(0.0)
    assert row["proposed_rmse_win_count"] == 1
    assert row["observed_fold_count"] == row["expected_fold_count"] == 2


def test_aggregation_reports_missing_folds_without_fabricating_them(tmp_path: Path) -> None:
    _result(tmp_path / "results", "L25_fold0", "psf128", 1.0, 2.0)
    output = comparison.aggregate_cyj_matched_results(tmp_path / "results", tmp_path / "summary")
    payload = json.loads((output / "summary.json").read_text())
    row = next(row for row in payload["summaries"] if row["endpoint"] == "psf128" and row["level"] == "L25" and row["parameter"] == "T1" and row["stack"] == "2CH")
    assert row["aggregation_status"] == "INCOMPLETE_MISSING_FOLDS"
    assert row["observed_folds"] == [0]
    assert row["missing_folds"] == [1, 2, 3]
    assert row["observed_fold_count"] == 1


def test_aggregation_keeps_endpoints_separate_and_refuses_overwrite(tmp_path: Path) -> None:
    _result(tmp_path / "results", "L100_fold0", "central", 1.0, 2.0)
    _result(tmp_path / "results", "L100_fold0", "psf128", 4.0, 2.0)
    output = comparison.aggregate_cyj_matched_results(tmp_path / "results", tmp_path / "summary")
    payload = json.loads((output / "summary.json").read_text())
    rows = [row for row in payload["summaries"] if row["level"] == "L100" and row["parameter"] == "T1" and row["stack"] == "2CH"]
    assert {row["endpoint"] for row in rows} == {"central", "psf128"}
    assert {row["proposed_rmse_mean"] for row in rows} == {1.0, 4.0}
    with pytest.raises(FileExistsError, match="overwrite"):
        comparison.aggregate_cyj_matched_results(tmp_path / "results", output)
