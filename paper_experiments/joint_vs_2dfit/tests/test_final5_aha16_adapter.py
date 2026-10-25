from __future__ import annotations

import json

import pytest

def _roi(root, subject: str) -> None:
    folder = root / subject / "roi"
    folder.mkdir(parents=True)
    for name in (
        "Segmentation_1-out-label.nii.gz",
        "Segmentation_2-in-label.nii.gz",
        "Segmentation_3-label_point.nii.gz",
        "sax_composite_reference.nii.gz",
    ):
        (folder / name).touch()


def _final5_roi(root, subject: str) -> None:
    _roi(root, subject)
    roi = root / subject / "roi"
    if subject == "DYL_2":
        (roi / "final_aha_roi_provenance.json").write_text(json.dumps({
            "schema": "dyl_2_final5_aha_manual_roi/v1",
        }))
    else:
        (roi / "roi_derivation_manifest.json").write_text(json.dumps({
            "schema": "final5_cropped_roi_bundle/v1", "subject": subject, "resampling": False,
        }))


def test_dyl_2_is_roi_pending_and_never_borrows_dyl(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.final5_adapter import assess_final5_aha_readiness

    _roi(tmp_path / "roi", "DYL")

    readiness = assess_final5_aha_readiness(tmp_path / "final5", "DYL_2", "L100", tmp_path / "roi")

    assert readiness["status"] == "ROI_PENDING"
    assert readiness["roi_root"] is None


def test_dyl_2_accepts_only_its_own_explicit_roi(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.final5_adapter import assess_final5_aha_readiness

    _final5_roi(tmp_path / "roi", "DYL_2")
    readiness = assess_final5_aha_readiness(tmp_path / "final5", "DYL_2", "L100", tmp_path / "roi")

    assert readiness["status"] == "ENDPOINTS_PENDING"
    assert readiness["roi_root"] == str(tmp_path / "roi" / "DYL_2" / "roi")


def test_missing_final5_endpoints_are_pending_without_path_discovery(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.final5_adapter import assess_final5_aha_readiness

    _final5_roi(tmp_path / "roi", "CYJ")

    readiness = assess_final5_aha_readiness(tmp_path / "final5", "CYJ", "L100", tmp_path / "roi")

    assert readiness["status"] == "ENDPOINTS_PENDING"
    assert readiness["proposed_endpoint"] == str(tmp_path / "final5" / "05_formal_endpoints" / "central" / "proposed" / "CYJ" / "L100")
    assert readiness["fit_first_endpoint"] == str(tmp_path / "final5" / "05_formal_endpoints" / "central" / "fit_first" / "CYJ" / "L100")


def test_final5_root_does_not_fallback_to_legacy_full_fov_roi(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.final5_adapter import resolve_subject_roi

    legacy = tmp_path / "subjects" / "CYJ" / "data" / "roi" / "20260821_corrected_sequence_v1"
    legacy.mkdir(parents=True)
    for name in (
        "Segmentation_1-out-label.nii.gz", "Segmentation_2-in-label.nii.gz",
        "Segmentation_3-label_point.nii.gz", "sax_composite_reference.nii.gz",
    ):
        (legacy / name).touch()

    assert resolve_subject_roi(tmp_path / "subjects", "CYJ") is None


def test_final5_root_resolves_only_versioned_derived_roi(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.final5_adapter import resolve_subject_roi

    root = tmp_path / "subjects"
    derived = root / "CYJ" / "data" / "roi" / "20261025_final5_cropped_roi_v1"
    derived.mkdir(parents=True)
    for name in (
        "Segmentation_1-out-label.nii.gz", "Segmentation_2-in-label.nii.gz",
        "Segmentation_3-label_point.nii.gz", "sax_composite_reference.nii.gz",
    ):
        (derived / name).touch()
    (derived / "roi_derivation_manifest.json").write_text(json.dumps({
        "schema": "final5_cropped_roi_bundle/v1", "subject": "CYJ", "resampling": False,
    }))

    assert resolve_subject_roi(root, "CYJ") == derived


def test_native_roi_mapping_must_match_the_declared_standardization_groups(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.final5_adapter import require_exact_native_roi_mapping
    from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256

    standard = tmp_path / "standardization.json"
    standard.write_text(json.dumps({"subject": "CYJ", "stacks": {"sax": {"standardized_to_original_group_idx": [{"derived_group_idx": 0, "original_group_idx": 4}]}}}))
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({
        "schema": "final5_aha_native_roi_mapping/v1", "subject": "CYJ", "case": "L100",
        "standardization_manifest": {"path": str(standard), "sha256": sha256(standard)},
        "planes": [{"level": "basal", "native_group_idx": 0, "original_group_idx": 3, "roi_slice_index": 1, "anterior_xy": [1, 2], "inferior_xy": [3, 4]}],
    }))

    with pytest.raises(ValueError, match="disagrees"):
        require_exact_native_roi_mapping(mapping, subject="CYJ", case="L100")
