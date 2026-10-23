"""CPU integration for corrected Fit-first formal endpoint -> native bundle."""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from paper_experiments.joint_vs_2dfit.phase2_clean.fit_first_native_bundles import build_fit_first_native_prediction_bundle


def _route(root: Path, r0: Path, endpoint: Path) -> None:
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"schema": "phase4r_intensity_preserving_fit_first/v1", "no_dictionary_refit": True, "no_pose_copying": True}))
    (root / "registered_quantitative_inputs_gate.json").write_text(json.dumps({"status": "REGISTERED_QUANTITATIVE_INPUTS_VALID"}))
    (root / "final_pose_audit.json").write_text(json.dumps({"schema": "fit_first_intensity_preserving_final_pose_audit/v1", "status": "POSE_STAGE1_AND_QUANTITATIVE_STAGE2_VALID"}))
    registered = {"parameters": {}}
    for parameter, translation in (("t1", 1.0), ("t2", 2.0)):
        target = root / "stage2_reconstruction" / parameter; target.mkdir(parents=True)
        (target / "model.pt").write_bytes(b"stage2")
        (target / "result.json").write_text(json.dumps({"input_stacks": None, "input_slices": str((root / f"registered_quantitative_inputs/{parameter}/slices").resolve()), "no_transformation_optimization": True, "no_slice_scale": True}))
        (target / "slices").mkdir()
        affine = np.eye(4); affine[0, 3] = translation
        nib.save(nib.Nifti1Image(np.zeros((1, 1, 1)), affine), target / "slices/0.nii.gz")
        # Deliberately incompatible Stage-1 intensity evidence: bundle must
        # sample endpoint volume, never these registration pixels.
        stage1 = root / "stage1_registration" / parameter / "slices"; stage1.mkdir(parents=True)
        nib.save(nib.Nifti1Image(np.full((1, 1, 1), 999.0), affine), stage1 / "0.nii.gz")
        registered["parameters"][parameter] = [{"index": 0, "stack": "sax", "standardized_group_idx": 0, "original_prestandardization_group_idx": 1}]
    target = root / "registered_quantitative_inputs"; target.mkdir(); (target / "manifest.json").write_text(json.dumps(registered))
    r0_stack = r0 / "sax"; r0_stack.mkdir(parents=True)
    np.savez_compressed(r0_stack / "native_maps.npz", group_idx=np.array([0]), t1_ms=np.ones((1, 1, 1)), t2_ms=np.ones((1, 1, 1)))
    volume = np.indices((5, 5, 5), dtype=np.float32)[0]
    for parameter in ("t1", "t2"):
        target = endpoint / parameter; target.mkdir(parents=True)
        nib.save(nib.Nifti1Image(volume, np.eye(4)), target / "volume_ms.nii.gz")
    (endpoint / "manifest.json").write_text(json.dumps({"schema": "phase4r_fixed_l100_fit_first_export_bundle/v1", "endpoint": "central"}))


def test_intensity_preserving_bundle_uses_stage2_pose_and_endpoint_values_not_stage1_pixels(tmp_path: Path) -> None:
    root, r0, endpoint = tmp_path / "fit", tmp_path / "r0", tmp_path / "endpoint"
    _route(root, r0, endpoint)
    bundle = build_fit_first_native_prediction_bundle(baseline_root=root, r0_native_maps_root=r0, endpoint="central", endpoint_source_root=endpoint, output=tmp_path / "bundle")
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["schema"] == "fit_first_native_prediction_bundle/v3"
    assert manifest["route"]["route_kind"] == "intensity_preserving_fit_first"
    assert np.load(bundle / "planes/t1/sax/group000.npz")["prediction_ms"].item() == 1.0
    assert np.load(bundle / "planes/t2/sax/group000.npz")["prediction_ms"].item() == 2.0
