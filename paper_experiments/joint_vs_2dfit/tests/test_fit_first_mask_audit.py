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
    assert result["both_valid_default_vs_fixed"]["descriptive_only"] is True
    assert result["both_valid_default_vs_fixed"]["monte_carlo_realizations_not_matched"] is True
    assert result["newly_recovered_fixed_vs_native"]["reference_mean_ms"] == pytest.approx(20.0)
    assert result["newly_recovered_fixed_vs_native"]["prediction_mean_ms"] == pytest.approx(22.0)


@pytest.mark.parametrize(("parameter", "scale"), (("t1", 2500.0), ("t2", 200.0)))
def test_sampling_scales_and_runtime_contract_are_fixed(parameter: str, scale: float, tmp_path: Path) -> None:
    command = audit.sample_volume_runtime_command(tmp_path / "python", tmp_path / "model.pt", tmp_path / "mask.nii.gz", tmp_path / "output.nii.gz", tmp_path / "effective_mask.nii.gz", tmp_path / "runtime.json", device="0")
    assert audit.NORMALIZATION_SCALES[parameter] == scale
    assert command[:2] == [str(tmp_path / "python"), "-c"]
    assert "--sample-mask" not in command
    assert "sample_volume" in command[2]
    assert "sample_mask.save_mask(effective_mask_path)" in command[2]
    assert command[-3:] == ["1.0", "128", "k128"]
    assert "reconstruct" not in command[2] and "train(" not in command[2]
    assert audit.INFERENCE_PARAMETERS == {"output_resolution_mm": 1.0, "output_psf_factor": 1, "n_inference_samples": 128, "inference_batch_size": 16000, "seed": 20260823, "output_intensity_mean": 0}


def test_sampling_runtime_resolves_device_and_requires_one_device_for_all_tensors(tmp_path: Path) -> None:
    assert audit.resolve_torch_device("0") == "cuda:0"
    assert audit.resolve_torch_device("-1") == "cpu"
    with pytest.raises(ValueError, match="device"):
        audit.resolve_torch_device("cuda:0")
    command = audit.sample_volume_runtime_command(tmp_path / "python", tmp_path / "model.pt", tmp_path / "mask.nii.gz", tmp_path / "output.nii.gz", tmp_path / "effective_mask.nii.gz", tmp_path / "runtime.json", device="0")
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


def test_formal_vs_restarted_k128_mismatch_is_descriptive_not_a_gate() -> None:
    result = audit.formal_vs_restarted_k128_audit(np.array([1.0, 2.0]), np.array([1.1, 2.2]), np.array([True, True]))
    assert result["rmse"] > 0
    assert result["numerical_identity_expected"] is False
    assert result["descriptive_only"] is True


def test_formal_self_geometry_mismatch_fails() -> None:
    geometry = audit.geometry_check((2, 2, 2), np.eye(4), (2, 2, 3), np.eye(4))
    assert geometry["status"] == "FAIL"
    with pytest.raises(RuntimeError, match="geometry"):
        audit.require_sanity_controls(geometry, {"status": "PASS"}, {"status": "PASS"})


def test_same_seed_repeatability_mismatch_fails() -> None:
    repeatability = audit.same_seed_repeatability_audit(np.ones((2, 2, 2)), np.eye(4), np.ones((2, 2, 2)) * 1.1, np.eye(4))
    assert repeatability["status"] == "FAIL"
    with pytest.raises(RuntimeError, match="repeatability"):
        audit.require_sanity_controls({"status": "PASS"}, repeatability, {"status": "PASS"})


def test_same_seed_repeatability_passes_on_common_sampling_mask() -> None:
    support = np.array([[[True]], [[False]]])
    result = audit.same_seed_repeatability_audit(
        np.array([[[1.0]], [[999.0]]]), np.eye(4), np.array([[[1.0]], [[-999.0]]]), np.eye(4),
        support, support,
    )
    assert result["status"] == "PASS"
    assert result["finite_common_voxel_count"] == 1


def test_k1_mask_only_common_support_predictions_are_deterministic() -> None:
    k1 = audit.k1_mask_only_audit(
        np.ones((2, 2, 2)), np.eye(4), np.ones((2, 2, 2)), np.eye(4),
        np.ones((2, 2, 2), bool), np.eye(4), np.ones((2, 2, 2), bool),
    )
    assert k1["intersection_voxel_count"] == 8
    assert k1["rmse"] == pytest.approx(0.0)
    assert k1["status"] == "PASS"


def test_k1_mask_only_excludes_l100_only_voxels() -> None:
    k1 = audit.k1_mask_only_audit(
        np.array([[[1.0]], [[999.0]]]), np.eye(4), np.array([[[1.0]], [[2.0]]]), np.eye(4),
        np.array([[[True]], [[False]]]), np.eye(4), np.array([[[True]], [[True]]]),
    )
    assert k1["intersection_voxel_count"] == 1
    assert k1["rmse"] == pytest.approx(0.0)


def test_independently_resampled_mask_bounding_boxes_need_not_share_a_grid() -> None:
    sparse_grid = audit.geometry_check((3, 3, 3), np.eye(4), (5, 3, 3), np.eye(4))
    shifted_l100_affine = np.eye(4); shifted_l100_affine[0, 3] = -2.0
    shifted_grid = audit.geometry_check((3, 3, 3), np.eye(4), (3, 3, 3), shifted_l100_affine)
    assert sparse_grid["status"] == "FAIL"
    assert shifted_grid["status"] == "FAIL"


def test_sparse_mask_maps_to_l100_reference_grid_with_exact_geometry() -> None:
    sparse = np.array([[[True]], [[False]], [[True]]])
    sparse_affine = np.eye(4); sparse_affine[0, 3] = -1.0
    reference_affine = np.eye(4)
    mapped = audit.map_mask_to_reference_grid(sparse, sparse_affine, (3, 1, 1), reference_affine)
    assert mapped.dtype == bool
    assert mapped.tolist() == [[[False]], [[True]], [[False]]]


def test_sparse_mask_mapping_uses_nearest_neighbor() -> None:
    source = np.array([[[True]], [[False]]])
    source_affine = np.eye(4); source_affine[0, 0] = 2.0
    mapped = audit.map_mask_to_reference_grid(source, source_affine, (3, 1, 1), np.eye(4))
    assert mapped[:, 0, 0].tolist() == [True, False, False]
    mapping_source = Path(audit.__file__).read_text(encoding="utf-8").split("def map_mask_to_reference_grid", 1)[1].split("def _resample_mask_to_grid", 1)[0]
    assert "order=0" in mapping_source


def test_k1_runtime_uses_central_sampling_and_fixed_l100_follows_sanity_gate(tmp_path: Path) -> None:
    command = audit.sample_volume_runtime_command(tmp_path / "python", tmp_path / "model.pt", tmp_path / "mask.nii.gz", tmp_path / "output.nii.gz", tmp_path / "effective_mask.nii.gz", tmp_path / "runtime.json", device="0", mode="k1")
    assert command[-3:] == ["0.0", "1", "k1"]
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert source.index("require_sanity_controls(self_geometry, repeatability, k1_control)") < source.index("fixed_normalized, fixed_effective_mask, fixed_runtime")


def test_k1_fixed_grid_runner_does_not_auto_resample_mask(tmp_path: Path) -> None:
    command = audit.sample_volume_runtime_command(tmp_path / "python", tmp_path / "model.pt", tmp_path / "mask.nii.gz", tmp_path / "output.nii.gz", tmp_path / "effective_mask.nii.gz", tmp_path / "runtime.json", device="0", mode="k1_fixed_grid")
    assert command[-3:] == ["0.0", "1", "k1_fixed_grid"]
    assert "if mode != \"k1_fixed_grid\":\n    sample_mask = sample_mask.resample(1.0, None)" in command[2]


def test_k1_reference_grid_is_created_from_l100_before_sparse_common_grid_sample() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")
    l100_reference = 'l100_mask, k1_l100_normalized, l100_effective_mask, k1_l100_runtime, device=device, mode="k1"'
    sparse_common_grid = 'sparse_mask_on_l100_grid, k1_self_normalized, k1_self_effective_mask, k1_self_runtime, device=device, mode="k1_fixed_grid"'
    assert source.index(l100_reference) < source.index("sparse_on_l100 = map_mask_to_reference_grid") < source.index(sparse_common_grid)


def test_k1_common_grid_comparison_has_no_spatial_interpolation() -> None:
    report = audit.k1_mask_only_audit(
        np.array([[[1.0]], [[2.0]]]), np.eye(4), np.array([[[1.0]], [[2.0]]]), np.eye(4),
        np.array([[[True]], [[False]]]), np.eye(4), np.array([[[True]], [[True]]]),
    )
    assert report["status"] == "PASS"
    assert report["spatial_interpolation_between_predictions"] is False
    assert report["reference_grid"] == "l100_effective_1mm_grid"
    function_source = Path(audit.__file__).read_text(encoding="utf-8").split("def k1_mask_only_audit", 1)[1].split("def require_sanity_controls", 1)[0]
    assert "resample_volume_to_grid" not in function_source


def test_manifest_explains_unrecoverable_historical_formal_rng(tmp_path: Path) -> None:
    manifest = audit.audit_manifest(sparse_manifest=tmp_path / "sparse.json", r0_native_maps_root=tmp_path, nesvor_executable="nesvor", checkpoints={})
    rng = manifest["rng_provenance"]
    assert rng["checkpoint_contains_rng_state"] is False
    assert rng["formal_volume_rng_state_recoverable"] is False
    assert rng["audit_same_seed_repeatability"] is True
    assert manifest["primary_scientific_change"] == "sampling_mask_only"


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
