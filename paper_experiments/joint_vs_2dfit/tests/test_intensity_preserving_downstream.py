"""CPU integration for corrected Fit-first formal endpoint -> native bundle."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import nibabel as nib
import numpy as np

from paper_experiments.joint_vs_2dfit.phase2_clean.fit_first_native_bundles import build_fit_first_native_prediction_bundle
from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256
from paper_experiments.joint_vs_2dfit.phase2_clean import final5_batch_runner as runner
from paper_experiments.joint_vs_2dfit.phase2_clean import matched_output


def _route(root: Path, r0: Path, endpoint: Path) -> None:
    root.mkdir(parents=True)
    (root / "registered_quantitative_inputs_gate.json").write_text(json.dumps({"schema": "phase4r_fit_first_registered_quantitative_inputs/v2", "status": "REGISTERED_QUANTITATIVE_INPUTS_VALID"}))
    registered = {"parameters": {}}
    stack_masks = {}
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
        stack = root / f"nesvor_inputs/sax/{parameter}/stack.nii.gz"; stack.parent.mkdir(parents=True); stack.write_bytes(b"stack")
        mask = root / f"nesvor_inputs/sax/{parameter}/stack_mask.nii.gz"; mask.write_bytes(b"mask")
        (stage1.parent / "result.json").write_text(json.dumps({"registration": "stack", "input_stacks": [str(stack.resolve())], "stack_masks": [str(mask.resolve())]}))
        stack_masks[parameter] = {"sax": {"path": str(mask.resolve()), "sha256": sha256(mask)}}
        registered["parameters"][parameter] = [{"index": 0, "stack": "sax", "standardized_group_idx": 0, "original_prestandardization_group_idx": 1}]
    stage1_manifest = root / "stage1_registration_manifest.json"
    stage1_manifest.write_text(json.dumps({"schema": "phase4r_fit_first_registration_stage/v2", "active_stacks": ["sax"], "stack_masks": stack_masks}))
    (root / "final_pose_audit.json").write_text(json.dumps({"schema": "fit_first_intensity_preserving_final_pose_audit/v2", "status": "POSE_STAGE1_AND_QUANTITATIVE_STAGE2_VALID", "parameters": {parameter: {"stage1_stack_masks": [stack_masks[parameter]["sax"]]} for parameter in ("t1", "t2")}}))
    (root / "manifest.json").write_text(json.dumps({"schema": "phase4r_intensity_preserving_fit_first/v2", "no_dictionary_refit": True, "no_pose_copying": True, "active_stacks": ["sax"], "stage1": {"path": str(stage1_manifest.resolve()), "sha256": sha256(stage1_manifest)}}))
    target = root / "registered_quantitative_inputs"; target.mkdir(); (target / "manifest.json").write_text(json.dumps(registered))
    r0_stack = r0 / "sax"; r0_stack.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(r0_stack / "native_maps.npz", group_idx=np.array([0]), t1_ms=np.ones((1, 1, 1)), t2_ms=np.ones((1, 1, 1)))
    volume = np.indices((5, 5, 5), dtype=np.float32)[0]
    for parameter in ("t1", "t2"):
        target = endpoint / parameter; target.mkdir(parents=True)
        nib.save(nib.Nifti1Image(volume, np.eye(4)), target / "volume_ms.nii.gz")
        nib.save(nib.Nifti1Image(np.zeros(volume.shape, np.uint8) if parameter == "t1" else np.ones(volume.shape, np.uint8), np.eye(4)), target / "query_effective_mask.nii.gz")
    parameters = {parameter: {"effective_query_mask": {"path": str((endpoint / parameter / "query_effective_mask.nii.gz").resolve()), "sha256": sha256(endpoint / parameter / "query_effective_mask.nii.gz")}} for parameter in ("t1", "t2")}
    (endpoint / "manifest.json").write_text(json.dumps({"schema": "phase4r_fixed_l100_fit_first_export_bundle/v2", "endpoint": "central", "parameters": parameters}))


def test_intensity_preserving_bundle_uses_stage2_pose_and_endpoint_values_not_stage1_pixels(tmp_path: Path) -> None:
    root, r0, endpoint = tmp_path / "fit", tmp_path / "r0", tmp_path / "endpoint"
    _route(root, r0, endpoint)
    bundle = build_fit_first_native_prediction_bundle(baseline_root=root, r0_native_maps_root=r0, endpoint="central", endpoint_source_root=endpoint, output=tmp_path / "bundle")
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["schema"] == "fit_first_native_prediction_bundle/v4"
    assert manifest["route"]["route_kind"] == "intensity_preserving_fit_first"
    assert np.load(bundle / "planes/t1/sax/group000.npz")["prediction_ms"].item() == 1.0
    assert np.load(bundle / "planes/t2/sax/group000.npz")["prediction_ms"].item() == 2.0
    assert not np.load(bundle / "planes/t1/sax/group000.npz")["query_effective_mask"].item()
    assert not np.load(bundle / "planes/t1/sax/group000.npz")["valid"].item()


def test_fixed_l100_builder_routes_are_admitted_by_final5_validator(tmp_path: Path, monkeypatch) -> None:
    """Exercise the real endpoint builder while mocking only NeSVoR runtime calls."""
    root = tmp_path / "final"
    sparse_manifest = tmp_path / "sparse_manifest.json"
    sparse_manifest.write_text(json.dumps({"subject": "CYJ", "n_samples": 1}))
    l100 = runner.CaseRecord("CYJ", "L100", "L100", None, 1, sparse_manifest, sha256(sparse_manifest), "a" * 40, tmp_path)
    sparse = runner.CaseRecord("CYJ", "L50_fold0", "L50", 0, 2, sparse_manifest, sha256(sparse_manifest), "a" * 40, tmp_path)
    l100_root, sparse_root = runner._paths(root, l100)["fit_first"], runner._paths(root, sparse)["fit_first"]
    _route(l100_root, tmp_path / "r0", tmp_path / "unused_l100_endpoint")
    _route(sparse_root, tmp_path / "r0", tmp_path / "unused_sparse_endpoint")
    nesvor = tmp_path / "nesvor"; nesvor.write_text("")
    nesvor.with_name("python").write_text("")

    monkeypatch.setattr(matched_output, "checkpoint_mask_export_command", lambda *_args: ("checkpoint-mask", str(_args[-1])))
    monkeypatch.setattr(matched_output, "sample_volume_runtime_command", lambda *_args, **_kwargs: ("sample-volume", *(str(value) for value in _args[3:6])))

    def fake_run(command):
        if command[0] == "checkpoint-mask":
            nib.save(nib.Nifti1Image(np.ones((2, 2, 1), np.uint8), np.eye(4)), command[1])
            return
        assert command[0] == "sample-volume"
        normalized, effective_mask, runtime = map(Path, command[1:])
        nib.save(nib.Nifti1Image(np.ones((2, 2, 1), np.float32), np.eye(4)), normalized)
        nib.save(nib.Nifti1Image(np.ones((2, 2, 1), np.uint8), np.eye(4)), effective_mask)
        runtime.write_text(json.dumps({"runtime": "mock"}))

    monkeypatch.setattr(matched_output, "_run", fake_run)
    output = runner._paths(root, sparse)["fit_endpoint"]
    matched_output.export_fixed_l100_fit_first(
        subject="CYJ", condition="L50_fold0", l100_fit_first_root=l100_root,
        sparse_fit_first_root=sparse_root, sparse_manifest=sparse_manifest,
        r0_native_maps_root=tmp_path / "r0", nesvor_executable=nesvor,
        endpoint="central", output=output, device="-1",
    )

    records = {("CYJ", "L100"): l100, ("CYJ", "L50_fold0"): sparse}
    validation = runner.validate_fit_first_endpoint(output, sparse, records, root)
    assert validation.state == "SKIP_VALID", validation
    bundle = json.loads((output / "manifest.json").read_text())
    routes = bundle["fit_first_routes"]
    for parameter in ("t1", "t2"):
        assert bundle["parameters"][parameter]["fit_first_routes"] == routes
        assert json.loads((output / parameter / "manifest.json").read_text())["fit_first_routes"] == routes

    def write_routes(value: dict | None) -> None:
        changed = deepcopy(bundle)
        if value is None:
            changed.pop("fit_first_routes", None)
        else:
            changed["fit_first_routes"] = value
            for parameter in ("t1", "t2"):
                changed["parameters"][parameter]["fit_first_routes"] = value
                (output / parameter / "manifest.json").write_text(json.dumps(changed["parameters"][parameter]))
        (output / "manifest.json").write_text(json.dumps(changed))

    write_routes(None)
    assert not runner.validate_fit_first_endpoint(output, sparse, records, root).valid
    for provenance_key in ("formal_reconstruction_model", "parameter_results", "final_pose_audit"):
        changed_routes = deepcopy(routes)
        if provenance_key == "final_pose_audit":
            changed_routes["l100"][provenance_key]["sha256"] = "0" * 64
        else:
            changed_routes["l100"][provenance_key]["t1"]["sha256"] = "0" * 64
        write_routes(changed_routes)
        assert not runner.validate_fit_first_endpoint(output, sparse, records, root).valid
