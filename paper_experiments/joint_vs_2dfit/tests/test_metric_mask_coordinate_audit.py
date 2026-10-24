"""CPU contracts for the read-only Final5 metric/mask/coordinate audit."""
from __future__ import annotations

import numpy as np
import pytest


def test_mask_count_ratios_use_the_named_denominators_and_empty_masks_are_safe() -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import metric_mask_coordinate_audit as audit

    summary = audit.mask_count_summary(
        source_support=np.array([[True, True], [True, False]]),
        phase4r_l100=np.array([[True, True], [False, False]]),
        base_native_support=np.array([[True, True], [False, False]]),
        primary_native_mask=np.array([[True, False], [False, False]]),
        reference=np.array([[1000.0, np.nan], [900.0, 800.0]]),
        proposed=np.array([[1001.0, 1001.0], [901.0, 801.0]]),
        proposed_valid=np.array([[True, True], [True, True]]),
        fit_first=np.array([[999.0, 999.0], [899.0, 799.0]]),
        fit_first_volume_geometry_valid=np.array([[True, True], [False, True]]),
        fit_first_query_effective_mask=np.array([[True, False], [True, True]]),
    )
    assert summary["primary_native_mask_voxels"] == 1
    assert summary["primary_over_source_support"] == pytest.approx(1 / 3)
    assert summary["primary_over_base_native_support"] == pytest.approx(1 / 2)
    assert summary["paired_common_support_voxels"] == 1
    assert summary["paired_over_primary_native_mask"] == 1.0
    assert summary["method_valid_iou"] == 1.0
    assert summary["method_valid_identical"] is True
    overlap = audit.mask_count_summary(
        source_support=np.ones((1, 3), bool), phase4r_l100=np.ones((1, 3), bool),
        base_native_support=np.ones((1, 3), bool), primary_native_mask=np.ones((1, 3), bool),
        reference=np.ones((1, 3)), proposed=np.ones((1, 3)),
        proposed_valid=np.array([[True, False, True]]), fit_first=np.ones((1, 3)),
        fit_first_volume_geometry_valid=np.ones((1, 3), bool),
        fit_first_query_effective_mask=np.array([[True, True, False]]),
    )
    assert overlap["method_valid_iou"] == pytest.approx(1 / 3)
    assert overlap["method_valid_identical"] is False
    empty = audit.safe_ratio(0, 0)
    assert empty is None


def test_training_coordinates_are_converted_before_physical_displacement() -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import metric_mask_coordinate_audit as audit

    train = np.array([[[1.0, 2.0, 3.0]]])
    fit_physical = np.array([[[11.0, 2.0, 3.0]]])
    physical = audit.train_to_physical_ras(train, lambda points: points + np.array([10.0, 0.0, 0.0]))
    report, displacement = audit.coordinate_displacement_summary(physical, fit_physical, np.ones((1, 1), bool))
    assert displacement.item() == 0.0
    assert report["max_displacement_mm"] == 0.0
    assert report["comparison_space"] == "physical_RAS_mm"


def test_coordinate_paths_distinguish_final_pose_from_original_native_grid() -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import metric_mask_coordinate_audit as audit

    assert audit.FINAL_POSE_COORDINATE_SEMANTICS != audit.SPARSE_NATIVE_GRID_COORDINATE_SEMANTICS


def test_new_qc_output_cannot_overlap_formal_artifact_roots(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import metric_mask_coordinate_audit as audit

    root = tmp_path / "final5"
    with pytest.raises(ValueError, match="formal artifact"):
        audit.require_new_qc_output(root, root / "06_metrics" / "audit")
    audit.require_new_qc_output(root, root / "07_qc" / "audit")


def test_missing_metric_inventory_row_is_reported_without_aborting(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import metric_mask_coordinate_audit as audit
    from paper_experiments.joint_vs_2dfit.phase2_clean.final5_batch_runner import CaseRecord

    record = CaseRecord("DYL_2", "L50_fold0", "L50", 0, 2, tmp_path / "sparse.json", "a" * 64, "b" * 40, tmp_path)
    row = audit.case_inventory_row(tmp_path / "final5", record)
    assert row["metric_status"] == "MISSING"
    assert row["auditable"] is False


def test_audit_writes_complete_qc_inventory_when_all_metric_artifacts_are_missing(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import metric_mask_coordinate_audit as audit
    from paper_experiments.joint_vs_2dfit.phase2_clean.final5_batch_runner import CaseRecord

    registry = tmp_path / "registry.json"; registry.write_text("{}", encoding="utf-8")
    record = CaseRecord("DYL_2", "L50_fold0", "L50", 0, 2, tmp_path / "sparse.json", "a" * 64, "b" * 40, tmp_path)
    monkeypatch.setattr(audit, "load_final5_registry", lambda _path: [record])
    monkeypatch.setattr(audit, "build_batch_plan", lambda *_args, **_kwargs: {"cases": {"DYL_2/L50_fold0": {"stages": {"final_native_grid_metrics": {"state": "BLOCKED_DEPENDENCY", "reason": "missing endpoint"}}}}})
    output = audit.run_metric_mask_coordinate_audit(registry=registry, final5_root=tmp_path / "final5", output=tmp_path / "final5" / "07_qc" / "audit", device="cpu", figures=False)
    assert (output / "case_inventory.csv").is_file()
    assert (output / "per_plane_mask_counts.csv").is_file()
    assert "MISSING" in (output / "case_inventory.csv").read_text(encoding="utf-8")


def test_converter_propagates_explicit_device(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import metric_mask_coordinate_audit as audit

    captured: list[str] = []
    class Space:
        def train_ras_to_physical(self, points): return points
    class Torch:
        @staticmethod
        def from_numpy(value): return value
    monkeypatch.setattr("paper_experiments.joint_vs_2dfit.phase2_clean.direct_native_endpoint_audit._load_model", lambda _checkpoint, _prepared, device: captured.append(device) or (object(), Space(), Torch()))
    manifest = {"checkpoint": {"path": str(tmp_path / "model.pt")}, "prepared_observations": {stack: {"path": str(tmp_path / f"{stack}.npz")} for stack in ("sax", "2ch", "4ch")}}
    audit._proposed_physical_converter(tmp_path, manifest, device="cuda:0")
    assert captured == ["cuda:0"]
