"""Contract tests for the Phase 0 read-only audit utilities."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    audit_dir = str(ROOT / "audit")
    if audit_dir not in sys.path:
        sys.path.insert(0, audit_dir)
    path = ROOT / "audit" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_reference(name: str):
    path = ROOT / "reference" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_observations(path: Path, *, groups: int = 2, tr_ms: float = 2.61, vps: int = 87) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = groups * 10
    images = np.zeros((count, 3, 4), dtype=np.float32)
    np.savez(
        path,
        images=images,
        masks=np.ones_like(images, dtype=bool),
        group_idx=np.repeat(np.arange(groups), 10),
        weight_idx=np.tile(np.arange(10), groups),
        stack_idx=np.zeros(count, dtype=np.int64),
        acquisition_time_ms=np.arange(count, dtype=np.float64),
        timing9_ms=np.tile(np.arange(9, dtype=np.float64), (count, 1)),
        affine_lps_rc=np.tile(np.eye(4), (count, 1, 1)),
        pixel_spacing_rc_mm=np.tile([1.2, 1.3], (count, 1)),
        slice_thickness_mm=np.full(count, 8.0),
        tr_ms=np.full(count, tr_ms),
        vps=np.full(count, vps),
    )


def test_decoder_audit_rejects_missing_checkpoint(tmp_path: Path) -> None:
    module = _load("audit_decoder_checkpoint")
    with pytest.raises(FileNotFoundError):
        module.audit_checkpoint(tmp_path / "missing.pth")


def test_decoder_audit_rejects_non_approved_metadata(tmp_path: Path) -> None:
    module = _load("audit_decoder_checkpoint")
    metadata = {
        "architecture": "12-200-200-200-10",
        "protocol_hhz_v1": {"tr_ms": 2.61, "vps": 87},
        "parameter_ranges": {"t1_ms": [20, 2500]},
        "validation_status": "awaiting_manual_review",
        "approved_by_manual_review": False,
        "timing9_min_ms": [0] * 9,
        "timing9_max_ms": [1] * 9,
        "train_timing9_min_ms": [0] * 9,
        "train_timing9_max_ms": [1] * 9,
    }
    checkpoint = tmp_path / "decoder.json"
    checkpoint.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="approved"):
        module.audit_checkpoint(checkpoint, require_approved=True)


def test_decoder_audit_extracts_protocol_metadata(tmp_path: Path) -> None:
    module = _load("audit_decoder_checkpoint")
    metadata = {
        "architecture": "12-200-200-200-10",
        "protocol_hhz_v1": {"tr_ms": 2.61, "vps": 87, "fa_deg": [45, 45, 45]},
        "parameter_ranges": {"t1_ms": [20, 2500]},
        "validation_status": "approved_by_manual_review",
        "approved_by_manual_review": True,
        "timing9_min_ms": [0] * 9,
        "timing9_max_ms": [1] * 9,
        "train_timing9_min_ms": [0] * 9,
        "train_timing9_max_ms": [1] * 9,
    }
    checkpoint = tmp_path / "decoder.json"
    checkpoint.write_text(json.dumps(metadata), encoding="utf-8")
    report = module.audit_checkpoint(checkpoint)
    assert report["approved_for_paper"] is True
    assert report["protocol_hhz_v1"]["tr_ms"] == 2.61


def test_cohort_audit_counts_complete_ten_weight_groups(tmp_path: Path) -> None:
    module = _load("audit_real_cohort")
    _write_observations(tmp_path / "prepared" / "S1" / "sax" / "observations.npz")
    row = module.inspect_observations(tmp_path / "prepared" / "S1" / "sax" / "observations.npz")
    assert row["group_count"] == 2
    assert row["weights_per_group"] == [10]
    assert row["prepared_input_eligible"] is True


def test_current_audit_output_schema(tmp_path: Path) -> None:
    module = _load("audit_current_method")
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("num_images: 10\n", encoding="utf-8")
    report = module.build_report(repo_root=tmp_path, protocol_path=protocol, reference_paths_path=None)
    assert {"git", "hhz_protocol", "inr_ranges", "expected_observation_schema"} <= set(report)


def test_audits_do_not_modify_existing_runs(tmp_path: Path) -> None:
    module = _load("audit_real_cohort")
    run_file = tmp_path / "runs" / "historical.txt"
    run_file.parent.mkdir(parents=True)
    run_file.write_text("preserve", encoding="utf-8")
    _write_observations(tmp_path / "prepared" / "S1" / "sax" / "observations.npz")
    module.audit_cohort(tmp_path / "prepared", preprocessed_root=None, native_reference_root=None, myocardium_mask_root=None)
    assert run_file.read_text(encoding="utf-8") == "preserve"


def _approved_decoder_manifest(path: Path, *, tr_ms: float = 2.61, vps: int = 87) -> Path:
    path.write_text(json.dumps({"approved_for_paper": True, "protocol_hhz_v1": {"tr_ms": tr_ms, "vps": vps}}), encoding="utf-8")
    return path


def test_vps_mismatch_is_not_decoder_compatible(tmp_path: Path) -> None:
    module = _load("audit_real_cohort")
    _write_observations(tmp_path / "prepared" / "DYL" / "sax" / "observations.npz", vps=85)
    report = module.audit_cohort(tmp_path / "prepared", preprocessed_root=None, native_reference_root=None, myocardium_mask_root=None, decoder_manifest=_approved_decoder_manifest(tmp_path / "decoder.json"), reference_search_roots=[])
    row = report["rows"][0]
    assert row["prepared_input_eligible"] is True
    assert row["decoder_compatible"] is False
    assert row["reconstruction_eligible_current"] is False
    assert "VPS" in row["decoder_incompatibility_reason"]


def test_vps_match_is_decoder_compatible(tmp_path: Path) -> None:
    module = _load("audit_real_cohort")
    _write_observations(tmp_path / "prepared" / "CYJ" / "sax" / "observations.npz", vps=87)
    report = module.audit_cohort(tmp_path / "prepared", preprocessed_root=None, native_reference_root=None, myocardium_mask_root=None, decoder_manifest=_approved_decoder_manifest(tmp_path / "decoder.json"), reference_search_roots=[])
    row = report["rows"][0]
    assert row["decoder_compatible"] is True
    assert row["reconstruction_eligible_current"] is True


def test_evaluation_eligibility_is_not_prepared_eligibility(tmp_path: Path) -> None:
    module = _load("audit_real_cohort")
    _write_observations(tmp_path / "prepared" / "CYJ" / "sax" / "observations.npz")
    report = module.audit_cohort(tmp_path / "prepared", preprocessed_root=None, native_reference_root=None, myocardium_mask_root=None, decoder_manifest=_approved_decoder_manifest(tmp_path / "decoder.json"), reference_search_roots=[])
    row = report["rows"][0]
    assert row["prepared_input_eligible"] is True
    assert row["quantitative_evaluation_eligible"] is False


def test_missing_reference_or_mask_keeps_evaluation_ineligible(tmp_path: Path) -> None:
    module = _load("audit_real_cohort")
    _write_observations(tmp_path / "prepared" / "CYJ" / "sax" / "observations.npz")
    old_root = tmp_path / "MultiMapCode"
    native = old_root / "subjects" / "CYJ" / "results" / "2D_fit_first" / "experiments" / "20260822_v5_full_pipeline_v1" / "fitting_qc" / "NIfTI_Slices"
    native.mkdir(parents=True)
    for parameter in ("T1", "T2"):
        (native / f"sax_{parameter}_stack_r0100.nii.gz").write_bytes(b"native")
    report = module.audit_cohort(tmp_path / "prepared", preprocessed_root=None, native_reference_root=None, myocardium_mask_root=None, decoder_manifest=_approved_decoder_manifest(tmp_path / "decoder.json"), reference_search_roots=[old_root])
    row = report["rows"][0]
    assert row["native_reference_available"] is True
    assert row["myocardium_mask_available"] is False
    assert row["quantitative_evaluation_eligible"] is False


def test_manual_roi_marker_rejects_legacy_mask_reuse(tmp_path: Path) -> None:
    module = _load("audit_real_cohort")
    _write_observations(tmp_path / "prepared" / "DYZ" / "sax" / "observations.npz")
    old_root = tmp_path / "MultiMapCode"
    roi = old_root / "subjects" / "DYZ" / "data" / "roi" / "20260823_v5_newdata_v1"
    roi.mkdir(parents=True)
    (roi / "ROI_MANUAL_STEP_REQUIRED.md").write_text("manual annotation required", encoding="utf-8")
    legacy = old_root / "subjects" / "DYZ" / "results" / "2D_fit_first" / "evaluations" / "20260823_dual_full_v1" / "nesvor_v5_128_w1" / "masks" / "native_sax"
    legacy.mkdir(parents=True)
    for name in ("myocardium_full.nii.gz", "myocardium_core.nii.gz"):
        (legacy / name).write_bytes(b"legacy-mask")
    report = module.audit_cohort(tmp_path / "prepared", preprocessed_root=None, native_reference_root=None, myocardium_mask_root=None, decoder_manifest=_approved_decoder_manifest(tmp_path / "decoder.json"), reference_search_roots=[old_root])
    subject = report["subject_summary"][0]
    assert subject["myocardium_mask_available"] is False
    assert subject["quantitative_evaluation_eligible"] is False
    assert "legacy evaluation masks are not reusable" in subject["reasons"][0]


def test_verified_legacy_roi_marker_becomes_migration_pending_not_incompatible(tmp_path: Path) -> None:
    module = _load("audit_real_cohort")
    _write_observations(tmp_path / "prepared" / "DYZ" / "sax" / "observations.npz")
    old_root = tmp_path / "MultiMapCode"
    roi = old_root / "subjects" / "DYZ" / "data" / "roi" / "20260823_v5_newdata_v1"
    roi.mkdir(parents=True)
    (roi / "ROI_MANUAL_STEP_REQUIRED.md").write_text("manual annotation required", encoding="utf-8")
    compatibility = tmp_path / "legacy_roi_compatibility.json"
    compatibility.write_text(json.dumps({"subjects": {"DYZ": {"compatibility": {"status": "reusable_with_deterministic_transform"}}}}), encoding="utf-8")
    report = module.audit_cohort(tmp_path / "prepared", preprocessed_root=None, native_reference_root=None, myocardium_mask_root=None, decoder_manifest=_approved_decoder_manifest(tmp_path / "decoder.json"), reference_search_roots=[old_root], legacy_roi_compatibility_report=compatibility)
    subject = report["subject_summary"][0]

    assert subject["myocardium_mask_available"] is False
    assert "verified compatible" in subject["reasons"][0]


def test_identical_roi_geometry_is_reusable_exact() -> None:
    module = _load("audit_legacy_roi_compatibility")

    result = module.classify_compatibility(
        uid_match=True,
        raw_pixels_match=True,
        geometry_class="IDENTICAL_GRID",
        deterministic_crop=False,
        marker_text=None,
    )

    assert result["status"] == "reusable_exact"


def test_exact_geometry_plus_crop_offset_is_reusable_with_deterministic_transform() -> None:
    module = _load("audit_legacy_roi_compatibility")

    result = module.classify_compatibility(
        uid_match=True,
        raw_pixels_match=True,
        geometry_class="EXACT_REORIENTABLE_GRID",
        deterministic_crop=True,
        marker_text=None,
    )

    assert result["status"] == "reusable_with_deterministic_transform"


def test_uid_or_geometry_mismatch_is_incompatible() -> None:
    module = _load("audit_legacy_roi_compatibility")

    uid_mismatch = module.classify_compatibility(
        uid_match=False,
        raw_pixels_match=True,
        geometry_class="IDENTICAL_GRID",
        deterministic_crop=False,
        marker_text=None,
    )
    geometry_mismatch = module.classify_compatibility(
        uid_match=True,
        raw_pixels_match=True,
        geometry_class="NON_EQUIVALENT_GRID",
        deterministic_crop=False,
        marker_text=None,
    )

    assert uid_mismatch["status"] == "incompatible"
    assert geometry_mismatch["status"] == "incompatible"


def test_marker_alone_does_not_make_compatible_roi_incompatible() -> None:
    module = _load("audit_legacy_roi_compatibility")

    result = module.classify_compatibility(
        uid_match=True,
        raw_pixels_match=True,
        geometry_class="IDENTICAL_GRID",
        deterministic_crop=False,
        marker_text="manual annotation was pending when this note was written",
    )

    assert result["status"] == "reusable_exact"


def test_migration_refuses_existing_destination_and_unverified_subject(tmp_path: Path) -> None:
    module = _load_reference("migrate_verified_legacy_roi")
    destination = tmp_path / "existing"
    destination.mkdir()
    report = {"subjects": {"S1": {"compatibility": {"status": "unresolved_manual_review"}}}}

    with pytest.raises(ValueError, match="not verified"):
        module.validate_migration_request(report, "S1", tmp_path / "new")
    report["subjects"]["S1"]["compatibility"]["status"] = "reusable_exact"
    with pytest.raises(FileExistsError, match="overwrite"):
        module.validate_migration_request(report, "S1", destination)


def test_pose_audit_requires_independent_t1_t2_artifact_chain(tmp_path: Path) -> None:
    module = _load("audit_2dfit_pose_policy")
    old_root = tmp_path / "MultiMapCode"
    base = old_root / "subjects" / "CYJ" / "results" / "2D_fit_first" / "experiments" / "20260822_v5_full_pipeline_v1" / "output" / "nesvor_v5_128_w1"
    for parameter in ("T1", "T2"):
        result = base / parameter / "result.json"
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(json.dumps({"registration": "stack", "no_transformation_optimization": False}), encoding="utf-8")
        (result.parent / "registered_slices").mkdir()
        (result.parent / "registered_slices" / "0.nii.gz").write_bytes(parameter.encode())
    report = module.audit_pose_policy(old_root)
    assert report["shared_final_pose"] is False
    assert set(report["artifact_chain"]) == {"T1", "T2"}
    assert report["artifact_chain"]["T1"]["final_pose_artifact"].endswith("T1/registered_slices/0.nii.gz")
