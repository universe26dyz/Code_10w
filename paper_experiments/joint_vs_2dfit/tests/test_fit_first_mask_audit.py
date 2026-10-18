"""CPU contracts for the read-only sparse fit-first sampling-mask audit."""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.phase2_clean import fit_first_mask_audit as audit


def test_exported_mask_summary_preserves_nifti_shape_and_affine(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint_mask.nii.gz"
    affine = np.diag((2.0, 3.0, 4.0, 1.0)); affine[:3, 3] = (5.0, 6.0, 7.0)
    nib.save(nib.Nifti1Image(np.ones((2, 3, 4), np.uint8), affine), path)
    summary = audit.mask_nifti_summary(path)
    assert summary["shape"] == [2, 3, 4]
    assert summary["affine"] == affine.tolist()
    assert summary["mask_voxel_count"] == 24


def test_refuse_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "already_exists"; output.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        audit.require_new_output(output)


def test_sparse_mask_is_compared_on_l100_world_grid() -> None:
    l100_mask = np.ones((3, 1, 1), bool)
    sparse_mask = np.ones((2, 1, 1), bool)
    l100_affine = np.eye(4)
    sparse_affine = np.eye(4); sparse_affine[0, 3] = 1.0
    result = audit.mask_support_audit(l100_mask, l100_affine, sparse_mask, sparse_affine)
    assert result["l100_mask_voxel_count"] == 3
    assert result["sparse_mask_voxel_count"] == 2
    assert result["intersection_voxel_count"] == 2
    assert result["l100_only_voxel_count"] == 1
    assert result["sparse_to_l100_coverage_ratio"] == pytest.approx(2 / 3)


def test_heldout_newly_recovered_and_both_valid_supports_are_disjoint() -> None:
    reference = np.array([[10.0, 20.0, 30.0]])
    default = np.array([[11.0, np.nan, 31.0]])
    fixed = np.array([[12.0, 22.0, 32.0]])
    source = np.array([[True, True, False]])
    default_valid = np.array([[True, False, True]])
    fixed_valid = np.array([[True, True, True]])
    result = audit.heldout_mask_plane_audit(reference, default, fixed, source, default_valid, fixed_valid)
    assert result["newly_recovered_voxel_count"] == 1
    assert result["both_valid_voxel_count"] == 1
    assert result["both_valid_default_vs_fixed"]["rmse"] == pytest.approx(1.0)
    assert result["newly_recovered_fixed_vs_native"]["reference_mean_ms"] == pytest.approx(20.0)
    assert result["newly_recovered_fixed_vs_native"]["prediction_mean_ms"] == pytest.approx(22.0)


@pytest.mark.parametrize(("parameter", "scale"), (("t1", 2500.0), ("t2", 200.0)))
def test_sampling_scales_and_runtime_contract_are_fixed(parameter: str, scale: float, tmp_path: Path) -> None:
    command = audit.sample_volume_runtime_command(tmp_path / "python", tmp_path / "model.pt", tmp_path / "mask.nii.gz", tmp_path / "output.nii.gz", tmp_path / "runtime.json", device="0")
    assert audit.NORMALIZATION_SCALES[parameter] == scale
    assert command[:2] == [str(tmp_path / "python"), "-c"]
    assert "--sample-mask" not in command
    assert "sample_volume" in command[2]
    assert "reconstruct" not in command[2] and "train(" not in command[2]
    assert audit.INFERENCE_PARAMETERS == {"output_resolution_mm": 1.0, "output_psf_factor": 1, "n_inference_samples": 128, "inference_batch_size": 16000, "seed": 20260823, "output_intensity_mean": 0}


def test_sampling_runtime_resolves_device_and_requires_one_device_for_all_tensors(tmp_path: Path) -> None:
    assert audit.resolve_torch_device("0") == "cuda:0"
    assert audit.resolve_torch_device("-1") == "cpu"
    with pytest.raises(ValueError, match="device"):
        audit.resolve_torch_device("cuda:0")
    command = audit.sample_volume_runtime_command(tmp_path / "python", tmp_path / "model.pt", tmp_path / "mask.nii.gz", tmp_path / "output.nii.gz", tmp_path / "runtime.json", device="0")
    script = command[2]
    assert "load_mask(sample_mask_path, device=device)" in script
    assert "model.to(device)" in script
    assert "sample_mask.xyz_masked.device" in script
    assert "device mismatch" in script


def test_manifest_records_runtime_provenance(tmp_path: Path) -> None:
    runtime = {"nesvor_package_path": "/data/dengyz/code/NeSVoR-master/nesvor/__init__.py", "sample_source_path": "/data/dengyz/code/NeSVoR-master/nesvor/inr/sample.py", "requested_device": "0", "resolved_torch_device": "cuda:0"}
    manifest = audit.audit_manifest(sparse_manifest=tmp_path / "sparse.json", r0_native_maps_root=tmp_path, nesvor_executable="nesvor", checkpoints={}, sampling_runtime=runtime)
    assert manifest["sampling_backend"] == "nesvor_adjacent_python_direct_sample_volume"
    assert manifest["sampling_runtime"] == runtime


def test_manifest_declares_sampling_mask_only(tmp_path: Path) -> None:
    sparse_manifest = tmp_path / "sparse_manifest.json"; sparse_manifest.write_text(json.dumps({"status": "ELIGIBLE"}))
    manifest = audit.audit_manifest(
        sparse_manifest=sparse_manifest,
        r0_native_maps_root=tmp_path,
        nesvor_executable="nesvor",
        checkpoints={"t1": {"sparse": tmp_path / "s.pt", "l100": tmp_path / "l.pt"}, "t2": {"sparse": tmp_path / "s2.pt", "l100": tmp_path / "l2.pt"}},
    )
    assert manifest["no_training"] is True
    assert manifest["no_refitting"] is True
    assert manifest["scientific_change"] == "sampling_mask_only"


def test_audit_does_not_invoke_training_or_native_dictionary_refit() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "fit_native(" not in source
    assert '"reconstruct"' not in source
