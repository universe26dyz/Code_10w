"""CPU regression coverage for the Fit-first final-pose contract."""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.phase2_clean.fit_first_native_bundles import build_fit_first_native_prediction_bundle
from paper_experiments.joint_vs_2dfit.phase2_clean.fit_first_native_pose_audit import audit_fit_first_native_poses
from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256


def _write_case(root: Path, r0: Path, *, final_translation: float = 2.0, initial_round_trip_error_mm: float = 0.0, negative_det: bool = False) -> None:
    groups = {"sax": [5], "2ch": [2], "4ch": [8]}
    ras = np.eye(4, dtype=float)
    if negative_det:
        ras[0, 0] = -1.0
    lps = np.diag([-1.0, -1.0, 1.0, 1.0]) @ ras
    for stack, ids in groups.items():
        maps_root, r0_stack = root / "native_maps" / stack, r0 / stack
        maps_root.mkdir(parents=True); r0_stack.mkdir(parents=True)
        t1 = np.array([[[1000, 1100, np.nan], [1200, 1300, 1400]]], np.float32)
        t2 = np.array([[[100, 110, np.nan], [120, 130, 140]]], np.float32)
        np.savez_compressed(maps_root / "native_maps.npz", group_idx=np.asarray(ids), t1_ms=t1, t2_ms=t2)
        np.savez_compressed(r0_stack / "native_maps.npz", group_idx=np.asarray(ids), t1_ms=t1, t2_ms=t2)
        np.savez_compressed(r0_stack / "geometry.npz", affine_lps_rc=lps[None])
        inputs = root / "nesvor_inputs" / stack
        inputs.mkdir(parents=True); (inputs / "manifest.json").write_text(json.dumps({"group_idx": ids}), encoding="utf-8")
        for parameter, raw, scale in (("t1", t1, 2500), ("t2", t2, 200)):
            values = np.where(np.isfinite(raw), raw / scale, 0.0)[0]
            (inputs / parameter).mkdir(exist_ok=True)
            initial_affine = ras.copy(); initial_affine[0, 3] += initial_round_trip_error_mm
            nib.save(nib.Nifti1Image(values[:, :, None], initial_affine), inputs / parameter / "stack.nii.gz")
    for parameter in ("t1", "t2"):
        nesvor = root / "nesvor" / parameter; (nesvor / "slices").mkdir(parents=True)
        (nesvor / "result.json").write_text(json.dumps({"registration": "stack", "no_transformation_optimization": False}), encoding="utf-8")
        (nesvor / "model.pt").write_bytes(b"checkpoint")
        volume = np.indices((12, 12, 12), dtype=np.float32)[0] + (100 if parameter == "t1" else 10)
        nib.save(nib.Nifti1Image(volume, np.eye(4)), nesvor / "volume_ms.nii.gz")
        for index, stack in enumerate(("sax", "2ch", "4ch")):
            raw = np.load(r0 / stack / "native_maps.npz")[f"{parameter}_ms"][0]
            values = np.where(np.isfinite(raw), raw / (2500 if parameter == "t1" else 200), 0.0)
            if negative_det:
                # save_nii_volume writes the internal row axis after the
                # affine2transformation negative-determinant flip.
                values = values[::-1]
            final = ras.copy(); final[0, 3] = final_translation + (0.25 if parameter == "t2" else 0.0)
            nib.save(nib.Nifti1Image(values[:, :, None], final), nesvor / "slices" / f"{index}.nii.gz")


@pytest.mark.parametrize("negative_det", [False, True])
def test_audit_separates_representation_flip_from_true_final_motion(tmp_path: Path, negative_det: bool) -> None:
    baseline, r0 = tmp_path / "baseline", tmp_path / "r0"
    _write_case(baseline, r0, negative_det=negative_det)
    output = audit_fit_first_native_poses(baseline_root=baseline, r0_native_maps_root=r0, output=tmp_path / "audit")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["reconstruction_input_geometry_gate"]["status"] == "RECONSTRUCTION_INPUT_GEOMETRY_VALID"
    record = manifest["parameters"]["t1"]["groups"][0]
    assert record["r0_vs_initial"]["status"] == "PASS_REPRESENTATION_EQUIVALENT"
    assert record["reindex_flip"]["flip_original_row"] is negative_det
    assert record["initial_vs_final"]["center_displacement_mm"] == pytest.approx(2.0)
    assert record["t1_final_vs_t2_final"]["center_displacement_mm"] == pytest.approx(.25)
    assert record["raw_affine_diagnostic"]["raw_affine_axis_convention_difference_not_motion"]
    assert manifest["schema"] == "fit_first_native_pose_audit/v3"
    assert manifest["reconstruction_input_geometry_gate"]["geometry_equivalence_tolerance_mm"] == pytest.approx(1e-3)


@pytest.mark.parametrize("round_trip_error_mm", [1.5e-4, 9e-4])
def test_geometry_gate_accepts_sub_micrometre_and_within_tolerance_round_trip_error(tmp_path: Path, round_trip_error_mm: float) -> None:
    baseline, r0 = tmp_path / "baseline", tmp_path / "r0"
    _write_case(baseline, r0, initial_round_trip_error_mm=round_trip_error_mm)
    output = audit_fit_first_native_poses(baseline_root=baseline, r0_native_maps_root=r0, output=tmp_path / "audit")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["reconstruction_input_geometry_gate"]["status"] == "RECONSTRUCTION_INPUT_GEOMETRY_VALID"
    assert manifest["parameters"]["t1"]["groups"][0]["r0_vs_initial"]["max_world_error_mm"] == pytest.approx(round_trip_error_mm, abs=1e-7)


def test_geometry_gate_rejects_real_input_geometry_mismatch(tmp_path: Path) -> None:
    baseline, r0 = tmp_path / "baseline", tmp_path / "r0"
    _write_case(baseline, r0, initial_round_trip_error_mm=.01)
    output = audit_fit_first_native_poses(baseline_root=baseline, r0_native_maps_root=r0, output=tmp_path / "audit")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["reconstruction_input_geometry_gate"]["status"] == "RECONSTRUCTION_INPUT_GEOMETRY_MISMATCH"
    assert manifest["reconstruction_input_geometry_gate"]["failures"]


def test_bundle_uses_volume_not_output_slice_and_refuses_invalid_gate(tmp_path: Path) -> None:
    baseline, r0 = tmp_path / "baseline", tmp_path / "r0"
    _write_case(baseline, r0)
    audit_root = audit_fit_first_native_poses(baseline_root=baseline, r0_native_maps_root=r0, output=tmp_path / "audit")
    endpoint = tmp_path / "endpoint"
    for parameter in ("t1", "t2"):
        target = endpoint / parameter; target.mkdir(parents=True)
        (target / "volume_ms.nii.gz").write_bytes((baseline / "nesvor" / parameter / "volume_ms.nii.gz").read_bytes())
        volume = nib.load(str(target / "volume_ms.nii.gz"))
        nib.save(nib.Nifti1Image(np.ones(volume.shape, np.uint8), volume.affine), target / "query_effective_mask.nii.gz")
    parameters = {parameter: {"effective_query_mask": {"path": str((endpoint / parameter / "query_effective_mask.nii.gz").resolve()), "sha256": sha256(endpoint / parameter / "query_effective_mask.nii.gz")}} for parameter in ("t1", "t2")}
    (endpoint / "manifest.json").write_text(json.dumps({"schema": "phase4r_fixed_l100_fit_first_export_bundle/v2", "endpoint": "central", "parameters": parameters}), encoding="utf-8")
    bundle = build_fit_first_native_prediction_bundle(audit_root=audit_root, baseline_root=baseline, r0_native_maps_root=r0, endpoint="central", endpoint_source_root=endpoint, output=tmp_path / "bundle")
    with np.load(bundle / "planes" / "t1" / "sax" / "group005.npz") as plane:
        assert np.allclose(plane["prediction_ms"], [[102, 102, 102], [103, 103, 103]])
        assert plane["valid"].all()
    audit = json.loads((audit_root / "manifest.json").read_text(encoding="utf-8"))
    audit["schema"] = "fit_first_native_pose_audit/v2"
    (audit_root / "manifest.json").write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="v3"):
        build_fit_first_native_prediction_bundle(audit_root=audit_root, baseline_root=baseline, r0_native_maps_root=r0, endpoint="central", endpoint_source_root=endpoint, output=tmp_path / "old_schema")
    audit["schema"] = "fit_first_native_pose_audit/v3"
    audit["reconstruction_input_geometry_gate"]["status"] = "RECONSTRUCTION_INPUT_GEOMETRY_MISMATCH"
    (audit_root / "manifest.json").write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="RECONSTRUCTION_INPUT_GEOMETRY_MISMATCH"):
        build_fit_first_native_prediction_bundle(audit_root=audit_root, baseline_root=baseline, r0_native_maps_root=r0, endpoint="central", endpoint_source_root=endpoint, output=tmp_path / "rejected")
