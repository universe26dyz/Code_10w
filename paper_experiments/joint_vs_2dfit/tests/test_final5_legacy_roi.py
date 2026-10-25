from __future__ import annotations

import hashlib
import json

import nibabel as nib
import numpy as np
import pytest


ROI_FILES = (
    "Segmentation_1-out-label.nii.gz",
    "Segmentation_2-in-label.nii.gz",
    "Segmentation_3-label_point.nii.gz",
    "sax_composite_reference.nii.gz",
)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image(path, values: np.ndarray, affine: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(values, affine), str(path))


def _derivation_inputs(tmp_path, *, target_offset=(2.0, 1.0, 0.0), clipped_point=False):
    legacy = tmp_path / "legacy"; legacy.mkdir()
    affine = np.eye(4)
    outer = np.zeros((6, 5, 2), np.uint8); outer[2:5, 1:3, :] = 1
    inner = np.zeros_like(outer); inner[3:4, 1:2, :] = 1
    points = np.zeros_like(outer); points[2, 1, 0] = 5; points[4, 2, 0] = 6
    points[2, 1, 1] = 1; points[4, 2, 1] = 2
    if clipped_point:
        points[0, 0, 0] = 3
    composite = np.arange(60, dtype=np.float32).reshape(6, 5, 2)
    for name, values in zip(ROI_FILES, (outer, inner, points, composite)):
        _image(legacy / name, values, affine)
    target = tmp_path / "final5_cropped_reference.nii.gz"
    target_affine = np.eye(4); target_affine[:3, 3] = target_offset
    _image(target, np.zeros((3, 2, 2), np.float32), target_affine)
    standard = tmp_path / "standardization.json"
    standard.write_text(json.dumps({"subject": "CYJ", "stacks": {"sax": {"standardized_to_original_group_idx": [
        {"derived_group_idx": 0, "original_group_idx": 7}, {"derived_group_idx": 1, "original_group_idx": 3},
    ]}}}))
    config = tmp_path / "derivation.json"
    config.write_text(json.dumps({
        "schema": "final5_legacy_roi_derivation_config/v1", "subject": "CYJ",
        "legacy_roi": {"path": str(legacy)},
        "final5_sax_reference": {"path": str(target), "sha256": _sha256(target)},
        "standardization_manifest": {"path": str(standard), "sha256": _sha256(standard)},
        "planes": [
            {"native_group_idx": 0, "original_group_idx": 7, "legacy_roi_slice_index": 0, "target_reference_slice_index": 0},
            {"native_group_idx": 1, "original_group_idx": 3, "legacy_roi_slice_index": 1, "target_reference_slice_index": 1},
        ],
    }))
    return config, legacy, target


def test_derivation_crops_masks_on_the_exact_final5_grid(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.final5_legacy_roi import derive_final5_cropped_roi

    config, legacy, target = _derivation_inputs(tmp_path)
    output = derive_final5_cropped_roi(config=config, output=tmp_path / "derived")

    assert output.is_dir()
    assert all((output / name).is_file() for name in (*ROI_FILES, "roi_derivation_manifest.json", "roi_qc_montage.png"))
    derived = nib.load(str(output / "Segmentation_1-out-label.nii.gz"))
    original = nib.load(str(legacy / "Segmentation_1-out-label.nii.gz"))
    assert np.array_equal(derived.get_fdata(), original.get_fdata()[2:5, 1:3, :])
    assert np.allclose(derived.affine, nib.load(str(target)).affine, atol=1e-6, rtol=0)
    manifest = json.loads((output / "roi_derivation_manifest.json").read_text())
    assert [p["original_group_idx"] for p in manifest["planes"]] == [7, 3]
    assert manifest["resampling"] is False


def test_derivation_rejects_non_integer_crop(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.final5_legacy_roi import derive_final5_cropped_roi

    config, _, _ = _derivation_inputs(tmp_path, target_offset=(2.5, 1.0, 0.0))
    with pytest.raises(ValueError, match="integer"):
        derive_final5_cropped_roi(config=config, output=tmp_path / "derived")
    assert not (tmp_path / "derived").exists()


def test_derivation_rejects_integer_but_resampled_grid(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.final5_legacy_roi import derive_final5_cropped_roi

    config, _, target = _derivation_inputs(tmp_path)
    image = nib.load(str(target)); affine = image.affine.copy(); affine[0, 0] = 2.0
    _image(target, image.get_fdata(), affine)
    payload = json.loads(config.read_text()); payload["final5_sax_reference"]["sha256"] = _sha256(target)
    config.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="unit-grid"):
        derive_final5_cropped_roi(config=config, output=tmp_path / "derived")
    assert not (tmp_path / "derived").exists()


def test_derivation_rejects_landmark_outside_crop(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.final5_legacy_roi import derive_final5_cropped_roi

    config, _, _ = _derivation_inputs(tmp_path, clipped_point=True)
    with pytest.raises(ValueError, match="landmark"):
        derive_final5_cropped_roi(config=config, output=tmp_path / "derived")
    assert not (tmp_path / "derived").exists()


def test_final5_mapping_binds_the_derived_roi_manifest(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.final5_adapter import require_exact_native_roi_mapping
    from paper_experiments.joint_vs_2dfit.aha.final5_legacy_roi import derive_final5_cropped_roi

    config, _, _ = _derivation_inputs(tmp_path)
    output = derive_final5_cropped_roi(config=config, output=tmp_path / "derived")
    standard = tmp_path / "standardization.json"
    derivation = output / "roi_derivation_manifest.json"
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({
        "schema": "final5_aha_native_roi_mapping/v1", "subject": "CYJ", "case": "L100",
        "standardization_manifest": {"path": str(standard), "sha256": _sha256(standard)},
        "roi_derivation_manifest": {"path": str(derivation), "sha256": _sha256(derivation)},
        "planes": [{"level": "basal", "native_group_idx": 0, "original_group_idx": 7,
                    "roi_slice_index": 0, "roi_transform": "identity", "anterior_xy": [0, 0], "inferior_xy": [1, 1]}],
    }))

    assert require_exact_native_roi_mapping(mapping, subject="CYJ", case="L100", roi_root=output)["subject"] == "CYJ"
    payload = json.loads(mapping.read_text()); payload["roi_derivation_manifest"]["sha256"] = "stale"
    mapping.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="derivation"):
        require_exact_native_roi_mapping(mapping, subject="CYJ", case="L100", roi_root=output)


def test_derivation_dry_run_reports_missing_explicit_final5_input(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.final5_legacy_roi import derivation_readiness

    config, _, target = _derivation_inputs(tmp_path)
    target.unlink()

    readiness = derivation_readiness(config)
    assert readiness["status"] == "INPUT_PENDING"
    assert "final5_sax_reference" in readiness["reason"]
