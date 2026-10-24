"""CPU integration for corrected Fit-first formal endpoint -> native bundle."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.phase2_clean.fit_first_native_bundles import (
    _scmr_one_stage_group_records,
    _scmr_output_slices,
    build_fit_first_native_prediction_bundle,
)
from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256
from paper_experiments.joint_vs_2dfit.phase2_clean import final5_batch_runner as runner
from paper_experiments.joint_vs_2dfit.phase2_clean import matched_output
from paper_experiments.joint_vs_2dfit.phase2_clean.fit_first_route_contract import SCMR_ONE_STAGE_FIXED_ARGS


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


def _scmr_route(root: Path, r0: Path, sparse_manifest: Path, case_id: str) -> None:
    root.mkdir(parents=True)
    selected = root / "selected_standardized_native_maps" / "native_maps" / "sax"; selected.mkdir(parents=True)
    np.savez_compressed(selected / "native_maps.npz", group_idx=np.array([0]), t1_ms=np.ones((1, 1, 1)), t2_ms=np.ones((1, 1, 1)), b1=np.ones((1, 1, 1)))
    np.savez_compressed(selected / "geometry.npz", affine_lps_rc=np.eye(4)[None], pixel_spacing_rc_mm=np.ones((1, 2)), slice_thickness_mm=np.ones(1))
    selection = selected.parent / "manifest.json"
    selection.write_text(json.dumps({"schema": "phase4r_fit_first_selected_standardized_native_maps/v1", "stacks": {"sax": {"training_stack_present": True, "standardized_group_ids": [0], "group_identity": [{"standardized_group_idx": 0, "original_prestandardization_group_idx": 0}]}}}))
    r0_stack = r0 / "sax"; r0_stack.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(r0_stack / "native_maps.npz", group_idx=np.array([0]), t1_ms=np.ones((1, 1, 1)), t2_ms=np.ones((1, 1, 1)))
    commands = root / "commands.sh"; commands.write_text("nesvor reconstruct --input-stacks sax --registration stack --no-slice-scale\n")
    parameters = {}
    for parameter in ("t1", "t2"):
        input_stack = root / "nesvor_inputs/sax" / parameter / "stack.nii.gz"; input_stack.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.zeros((1, 1, 1)), np.eye(4)), input_stack)
        (input_stack.parent.parent / "manifest.json").write_text(json.dumps({"group_idx": [0]}))
        target = root / "nesvor" / parameter; target.mkdir(parents=True)
        model, result = target / "model.pt", target / "result.json"
        model.write_bytes(parameter.encode()); result.write_text(json.dumps({"input_stacks": [str((root / "nesvor_inputs/sax" / parameter / "stack.nii.gz").resolve())], "thicknesses": [8], "registration": "stack", "no_slice_scale": True, "no_transformation_optimization": False, **SCMR_ONE_STAGE_FIXED_ARGS}))
        slices = target / "slices"; slices.mkdir()
        nib.save(nib.Nifti1Image(np.zeros((1, 1, 1)), np.eye(4)), slices / "0.nii.gz")
        parameters[parameter] = {"model": {"path": str(model.resolve()), "sha256": sha256(model)}, "result": {"path": str(result.resolve()), "sha256": sha256(result)}, "final_slices": str(slices.resolve())}
    (root / "manifest.json").write_text(json.dumps({"schema": "phase4r_scmr_one_stage_fit_first/v1", "subject": "CYJ", "case_id": case_id, "profile_id": case_id.split("_fold")[0], "fold_id": None, "active_stacks": ["sax"], "no_dictionary_refit": True, "independent_t1_t2": True, "transformation_optimization_enabled": True, "no_slice_scale": True, "nesvor_version": "nesvor_v5_128_w1", "commands": {"path": str(commands.resolve()), "sha256": sha256(commands)}, "sparse_manifest": {"path": str(sparse_manifest.resolve()), "sha256": sha256(sparse_manifest), "generator_git_sha": "a" * 40}, "selected_standardized_native_maps": {"path": str(selection.resolve()), "sha256": sha256(selection)}, "selected_group_ids": {"sax": [0], "2ch": [], "4ch": []}, "parameters": parameters}))


def _scmr_three_stack_route(root: Path, r0: Path, sparse_manifest: Path) -> None:
    """Formal fixture with input representation flips independent of final pose."""
    root.mkdir(parents=True)
    stacks = ("sax", "2ch", "4ch")
    selected_root = root / "selected_standardized_native_maps" / "native_maps"
    selected_stacks = {}
    for stack in stacks:
        source = selected_root / stack; source.mkdir(parents=True)
        np.savez_compressed(source / "native_maps.npz", group_idx=np.array([0]), t1_ms=np.ones((1, 2, 3)), t2_ms=np.ones((1, 2, 3)), b1=np.ones((1, 2, 3)))
        np.savez_compressed(source / "geometry.npz", affine_lps_rc=np.eye(4)[None], pixel_spacing_rc_mm=np.ones((1, 2)), slice_thickness_mm=np.ones(1))
        selected_stacks[stack] = {"training_stack_present": True, "standardized_group_ids": [0], "group_identity": [{"standardized_group_idx": 0, "original_prestandardization_group_idx": 0}]}
        r0_stack = r0 / stack; r0_stack.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(r0_stack / "native_maps.npz", group_idx=np.array([0]), t1_ms=np.ones((1, 2, 3)), t2_ms=np.ones((1, 2, 3)))
    selection = selected_root / "manifest.json"; selection.write_text(json.dumps({"schema": "phase4r_fit_first_selected_standardized_native_maps/v1", "stacks": selected_stacks}))
    commands = root / "commands.sh"; commands.write_text("nesvor reconstruct --registration stack --no-slice-scale\n")
    parameters = {}
    for parameter in ("t1", "t2"):
        input_paths = []
        for stack, determinant_sign in (("sax", -1), ("2ch", 1), ("4ch", -1)):
            input_stack = root / "nesvor_inputs" / stack / parameter / "stack.nii.gz"; input_stack.parent.mkdir(parents=True, exist_ok=True)
            affine = np.eye(4); affine[0, 0] = determinant_sign
            nib.save(nib.Nifti1Image(np.zeros((2, 3, 1)), affine), input_stack)
            (input_stack.parent.parent / "manifest.json").write_text(json.dumps({"group_idx": [0]}))
            input_paths.append(str(input_stack.resolve()))
        target = root / "nesvor" / parameter; target.mkdir(parents=True)
        model, result, slices = target / "model.pt", target / "result.json", target / "slices"
        model.write_bytes(parameter.encode()); result.write_text(json.dumps({"input_stacks": input_paths, "thicknesses": [8, 8, 8], "registration": "stack", "no_slice_scale": True, "no_transformation_optimization": False, **SCMR_ONE_STAGE_FIXED_ARGS}))
        slices.mkdir()
        for index in range(3):
            nib.save(nib.Nifti1Image(np.zeros((2, 3, 1)), np.eye(4)), slices / f"{index}.nii.gz")
        parameters[parameter] = {"model": {"path": str(model.resolve()), "sha256": sha256(model)}, "result": {"path": str(result.resolve()), "sha256": sha256(result)}, "final_slices": str(slices.resolve())}
    (root / "manifest.json").write_text(json.dumps({"schema": "phase4r_scmr_one_stage_fit_first/v1", "subject": "CYJ", "case_id": "L100", "profile_id": "L100", "fold_id": None, "fold_count": 1, "active_stacks": list(stacks), "no_dictionary_refit": True, "independent_t1_t2": True, "transformation_optimization_enabled": True, "no_slice_scale": True, "nesvor_version": "nesvor_v5_128_w1", "commands": {"path": str(commands.resolve()), "sha256": sha256(commands)}, "sparse_manifest": {"path": str(sparse_manifest.resolve()), "sha256": sha256(sparse_manifest), "generator_git_sha": "a" * 40}, "selected_standardized_native_maps": {"path": str(selection.resolve()), "sha256": sha256(selection)}, "selected_group_ids": {stack: [0] for stack in stacks}, "parameters": parameters}))


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


def test_scmr_one_stage_bundle_uses_parameter_final_slice_geometry_and_endpoint_values(tmp_path: Path) -> None:
    sparse = tmp_path / "sparse.json"; sparse.write_text(json.dumps({"subject": "CYJ"}))
    root, r0, endpoint = tmp_path / "scmr", tmp_path / "r0", tmp_path / "endpoint"
    _scmr_route(root, r0, sparse, "L100")
    parameters = {}
    for parameter, value in (("t1", 1000.0), ("t2", 100.0)):
        target = endpoint / parameter; target.mkdir(parents=True)
        volume, mask = target / "volume_ms.nii.gz", target / "query_effective_mask.nii.gz"
        nib.save(nib.Nifti1Image(np.full((2, 2, 2), value), np.eye(4)), volume)
        nib.save(nib.Nifti1Image(np.ones((2, 2, 2), np.uint8), np.eye(4)), mask)
        parameters[parameter] = {"effective_query_mask": {"path": str(mask.resolve()), "sha256": sha256(mask)}}
    (endpoint / "manifest.json").write_text(json.dumps({"schema": "phase4r_fixed_l100_fit_first_export_bundle/v2", "endpoint": "central", "parameters": parameters}))
    bundle = build_fit_first_native_prediction_bundle(baseline_root=root, r0_native_maps_root=r0, endpoint="central", endpoint_source_root=endpoint, output=tmp_path / "bundle")
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["schema"] == "fit_first_native_prediction_bundle/v5"
    assert manifest["route"]["route_kind"] == "scmr_one_stage_fit_first"
    assert np.load(bundle / "planes/t1/sax/group000.npz")["prediction_ms"].item() == 1000.0


def test_scmr_one_stage_native_bundle_uses_input_affine_for_reindex_and_final_slice_for_pose(tmp_path: Path) -> None:
    """Historical CYJ geometry: input representation flip and output pose differ."""
    sparse = tmp_path / "sparse.json"; sparse.write_text(json.dumps({"subject": "CYJ"}))
    root, r0, endpoint = tmp_path / "scmr", tmp_path / "r0", tmp_path / "endpoint"
    _scmr_three_stack_route(root, r0, sparse)
    parameters = {}
    for parameter in ("t1", "t2"):
        target = endpoint / parameter; target.mkdir(parents=True)
        volume, mask = target / "volume_ms.nii.gz", target / "query_effective_mask.nii.gz"
        # Positive-determinant final output slices all use identity pose.  The
        # asymmetric x ramp makes a wrong descriptor visibly mirror rows.
        values = np.indices((4, 4, 4), dtype=np.float32)[0]
        nib.save(nib.Nifti1Image(values, np.eye(4)), volume)
        nib.save(nib.Nifti1Image(np.ones((4, 4, 4), np.uint8), np.eye(4)), mask)
        parameters[parameter] = {"effective_query_mask": {"path": str(mask.resolve()), "sha256": sha256(mask)}}
    (endpoint / "manifest.json").write_text(json.dumps({"schema": "phase4r_fixed_l100_fit_first_export_bundle/v2", "endpoint": "central", "parameters": parameters}))
    bundle = build_fit_first_native_prediction_bundle(baseline_root=root, r0_native_maps_root=r0, endpoint="central", endpoint_source_root=endpoint, output=tmp_path / "bundle")
    manifest = json.loads((bundle / "manifest.json").read_text())
    flips = {plane["stack"]: plane["reindex_flip"]["flip_original_row"] for plane in manifest["planes"] if plane["parameter"] == "t1"}
    assert flips == {"sax": True, "2ch": False, "4ch": True}
    # The sampled arrays are returned in original [row, col] order.  Negative
    # input stack determinants map native row 0 to final output x=H-1.
    assert np.array_equal(np.load(bundle / "planes/t1/sax/group000.npz")["prediction_ms"], [[1, 1, 1], [0, 0, 0]])
    assert np.array_equal(np.load(bundle / "planes/t1/2ch/group000.npz")["prediction_ms"], [[0, 0, 0], [1, 1, 1]])
    assert np.array_equal(np.load(bundle / "planes/t1/4ch/group000.npz")["prediction_ms"], [[1, 1, 1], [0, 0, 0]])


def test_scmr_output_slice_lattice_requires_exact_37_output_ids(tmp_path: Path) -> None:
    root = tmp_path / "route"
    slices = root / "nesvor/t1/slices"; slices.mkdir(parents=True)
    for index in range(37):
        (slices / f"{index}.nii.gz").write_bytes(b"slice")
    assert [path.name for path in _scmr_output_slices(root, "t1", 37)] == [f"{index}.nii.gz" for index in range(37)]
    (slices / "18.nii.gz").unlink()
    with pytest.raises(ValueError, match="exactly 0..36"):
        _scmr_output_slices(root, "t1", 37)
    (slices / "18.nii.gz").write_bytes(b"slice")
    (slices / "00.nii.gz").write_bytes(b"duplicate zero")
    with pytest.raises(ValueError, match="exactly 0..36"):
        _scmr_output_slices(root, "t1", 37)


def test_scmr_group_order_excludes_inactive_lax_and_requires_matching_output_count(tmp_path: Path) -> None:
    sparse = tmp_path / "sparse.json"; sparse.write_text(json.dumps({"subject": "CYJ"}))
    root, r0 = tmp_path / "scmr", tmp_path / "r0"
    _scmr_three_stack_route(root, r0, sparse)
    route = json.loads((root / "manifest.json").read_text())
    route["active_stacks"] = ["sax", "4ch"]
    (root / "manifest.json").write_text(json.dumps(route))
    selected_path = root / "selected_standardized_native_maps/native_maps/manifest.json"
    selected = json.loads(selected_path.read_text())
    selected["stacks"]["2ch"]["training_stack_present"] = False
    selected_path.write_text(json.dumps(selected))
    for parameter in ("t1", "t2"):
        (root / f"nesvor/{parameter}/slices/2.nii.gz").unlink()
    groups = _scmr_one_stage_group_records(root, "t1")
    assert [(item["stack"], item["output_slice_index"]) for item in groups] == [("sax", 0), ("4ch", 1)]
    # A selected/exported additional group makes the required count three,
    # while only output IDs 0..1 remain: fail before any filename guessing.
    selected["stacks"]["sax"]["standardized_group_ids"] = [0, 1]
    selected["stacks"]["sax"]["group_identity"].append({"standardized_group_idx": 1, "original_prestandardization_group_idx": 1})
    selected_path.write_text(json.dumps(selected))
    (root / "nesvor_inputs/sax/manifest.json").write_text(json.dumps({"group_idx": [0, 1]}))
    with pytest.raises(ValueError, match="output-slice IDs"):
        _scmr_one_stage_group_records(root, "t1")


@pytest.mark.parametrize("mutation", ["extra_output_id", "input_group_order"])
def test_scmr_bundle_rejects_non_authoritative_output_slice_or_input_group_mapping(tmp_path: Path, mutation: str) -> None:
    """Output IDs and exported stack group order are formal one-stage evidence."""
    sparse = tmp_path / "sparse.json"; sparse.write_text(json.dumps({"subject": "CYJ"}))
    root, r0, endpoint = tmp_path / "scmr", tmp_path / "r0", tmp_path / "endpoint"
    _scmr_three_stack_route(root, r0, sparse)
    if mutation == "extra_output_id":
        for parameter in ("t1", "t2"):
            nib.save(nib.Nifti1Image(np.zeros((2, 3, 1)), np.eye(4)), root / "nesvor" / parameter / "slices/3.nii.gz")
    else:
        (root / "nesvor_inputs/2ch/manifest.json").write_text(json.dumps({"group_idx": [99]}))
    parameters = {}
    for parameter in ("t1", "t2"):
        target = endpoint / parameter; target.mkdir(parents=True)
        volume, mask = target / "volume_ms.nii.gz", target / "query_effective_mask.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((4, 4, 4), np.float32), np.eye(4)), volume)
        nib.save(nib.Nifti1Image(np.ones((4, 4, 4), np.uint8), np.eye(4)), mask)
        parameters[parameter] = {"effective_query_mask": {"path": str(mask.resolve()), "sha256": sha256(mask)}}
    (endpoint / "manifest.json").write_text(json.dumps({"schema": "phase4r_fixed_l100_fit_first_export_bundle/v2", "endpoint": "central", "parameters": parameters}))
    with pytest.raises(ValueError, match="output-slice IDs|group order"):
        build_fit_first_native_prediction_bundle(baseline_root=root, r0_native_maps_root=r0, endpoint="central", endpoint_source_root=endpoint, output=tmp_path / "bundle")


def test_fixed_l100_builder_routes_are_admitted_by_final5_validator(tmp_path: Path, monkeypatch) -> None:
    """Exercise the real endpoint builder while mocking only NeSVoR runtime calls."""
    root = tmp_path / "final"
    sparse_manifest = tmp_path / "sparse_manifest.json"
    sparse_manifest.write_text(json.dumps({"subject": "CYJ", "n_samples": 1}))
    l100 = runner.CaseRecord("CYJ", "L100", "L100", None, 1, sparse_manifest, sha256(sparse_manifest), "a" * 40, tmp_path)
    sparse = runner.CaseRecord("CYJ", "L50_fold0", "L50", 0, 2, sparse_manifest, sha256(sparse_manifest), "a" * 40, tmp_path)
    l100_root, sparse_root = runner._paths(root, l100)["fit_first"], runner._paths(root, sparse)["fit_first"]
    _scmr_route(l100_root, tmp_path / "r0", sparse_manifest, "L100")
    _scmr_route(sparse_root, tmp_path / "r0", sparse_manifest, "L50_fold0")
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
    for provenance_key in ("formal_reconstruction_model", "parameter_results"):
        changed_routes = deepcopy(routes)
        changed_routes["l100"][provenance_key]["t1"]["sha256"] = "0" * 64
        write_routes(changed_routes)
        assert not runner.validate_fit_first_endpoint(output, sparse, records, root).valid
