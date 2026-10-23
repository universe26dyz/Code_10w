"""CPU contracts for the two-stage quantitative Fit-first route."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from paper_experiments.joint_vs_2dfit.phase2_clean.fit_first import build_independent_nesvor_commands
from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256
from paper_experiments.joint_vs_2dfit.phase2_clean.fit_first_intensity_preserving import (
    INPUT_SCHEMA,
    SCALES,
    VALUE_TOLERANCE,
    _reconstruct_args,
    _selected_standardized_maps,
    _validate_prepared_root,
    audit_intensity_preserving_final_pose,
    build_registration_commands,
    q99_normalisation_multiplier,
    rebuild_registered_quantitative_inputs,
    validate_registered_quantitative_inputs,
)


def _case(root: Path) -> tuple[Path, Path]:
    native, stage1 = root / "native", root / "stage1"
    for stack, group, tx in (("sax", 7, 2.0), ("2ch", 3, 5.0)):
        folder = native / stack; folder.mkdir(parents=True)
        t1 = np.array([[[1000.0, np.nan], [1250.0, 1500.0]]], np.float32)
        t2 = np.array([[[100.0, np.nan], [125.0, 150.0]]], np.float32)
        np.savez_compressed(folder / "native_maps.npz", group_idx=np.array([group]), t1_ms=t1, t2_ms=t2)
        for parameter in ("t1", "t2"):
            target = stage1 / parameter / "slices"; target.mkdir(parents=True, exist_ok=True)
            index = 0 if stack == "sax" else 1
            affine = np.eye(4); affine[0, 0] = -1; affine[0, 3] = tx + (0.5 if parameter == "t2" else 0.0)
            nib.save(nib.Nifti1Image(np.zeros((2, 2, 1), np.float32), affine), target / f"{index}.nii.gz")
    (native / "manifest.json").write_text(json.dumps({"schema": "phase4r_fit_first_selected_standardized_native_maps/v1", "stacks": {stack: {"training_stack_present": True, "group_identity": [{"standardized_group_idx": group, "original_prestandardization_group_idx": group + 100}], "output_maps": {"sha256": sha256(native / stack / "native_maps.npz")}} for stack, group in (("sax", 7), ("2ch", 3))}}))
    return native, stage1


def test_q99_bug_multiplier_uses_foreground_not_zero_background() -> None:
    values = np.r_[np.zeros(10_000), np.full(100, .85)]
    foreground = values > 0
    assert q99_normalisation_multiplier(values, foreground) == pytest.approx(1 / .85)


def test_registered_quantitative_inputs_restore_values_and_keep_independent_poses(tmp_path: Path) -> None:
    native, stage1 = _case(tmp_path)
    output = rebuild_registered_quantitative_inputs(native_maps_root=native, registration_root=stage1, output=tmp_path / "registered", active_stacks=["sax", "2ch"])
    gate = validate_registered_quantitative_inputs(native_maps_root=native, registration_root=stage1, inputs_root=output, active_stacks=["sax", "2ch"])
    assert json.loads((output / "manifest.json").read_text())["schema"] == INPUT_SCHEMA
    assert gate["status"] == "REGISTERED_QUANTITATIVE_INPUTS_VALID"
    assert gate["parameters"]["t1"]["scale"] == SCALES["t1"]
    assert gate["parameters"]["t2"]["scale"] == SCALES["t2"]
    t1 = nib.load(output / "t1/slices/0.nii.gz")
    t2 = nib.load(output / "t2/slices/0.nii.gz")
    assert not np.allclose(t1.affine, t2.affine)  # independently registered poses
    assert np.allclose(t1.get_fdata()[::-1, :, 0], [[1000 / 2500, 0], [1250 / 2500, 1500 / 2500]])


@pytest.mark.parametrize("mutation", ["pixel", "affine"])
def test_gate_rejects_mutated_quantitative_slice(tmp_path: Path, mutation: str) -> None:
    native, stage1 = _case(tmp_path)
    output = rebuild_registered_quantitative_inputs(native_maps_root=native, registration_root=stage1, output=tmp_path / "registered", active_stacks=["sax", "2ch"])
    path = output / "t1/slices/0.nii.gz"; image = nib.load(path)
    values, affine = image.get_fdata(), image.affine.copy()
    if mutation == "pixel": values[0, 0, 0] += VALUE_TOLERANCE * 2
    else: affine[0, 3] += VALUE_TOLERANCE * 2
    nib.save(nib.Nifti1Image(values, affine), path)
    with pytest.raises(ValueError):
        validate_registered_quantitative_inputs(native_maps_root=native, registration_root=stage1, inputs_root=output, active_stacks=["sax", "2ch"])


def test_stage_commands_keep_registration_and_reconstruction_separate(tmp_path: Path) -> None:
    stage1 = build_registration_commands(tmp_path / "inputs", ["sax", "4ch"], tmp_path / "reg", nesvor_executable="nesvor", device="-1")
    assert all(" register " in f" {command} " and "--registration stack" in command for command in stage1)
    legacy = build_independent_nesvor_commands("/inputs", "/output", nesvor_executable="nesvor")[0]
    assert "--input-stacks" in legacy and "--registration stack" in legacy
    sample_mask = tmp_path / "sax.nii.gz"
    stage2 = _reconstruct_args(tmp_path / "registered/t1/slices", tmp_path / "out", "t1", nesvor_executable="nesvor", device="-1", sample_mask=sample_mask)
    assert "--input-slices" in stage2 and "--registration" not in stage2
    assert "--no-transformation-optimization" in stage2 and "--no-slice-scale" in stage2
    assert stage2[stage2.index("--sample-mask") + 1] == str(sample_mask)
    for flag in ("--weight-transformation", "--weight-deform", "--weight-image", "--weight-pixel-var-smooth", "--output-intensity-mean", "--output-resolution", "--output-psf-factor", "--single-precision", "--log2-hashmap-size", "--batch-size", "--n-iter", "--n-samples", "--n-inference-samples", "--inference-batch-size", "--pixel-variance-weight", "--coarsest-resolution", "--finest-resolution", "--level-scale", "--n-features-per-level", "--depth", "--width", "--n-features-z", "--n-features-slice", "--weight-bias", "--image-regularization", "--delta", "--learning-rate", "--gamma", "--milestones", "--seed"):
        assert flag in stage2


def test_rebuild_refuses_existing_output_and_inactive_stack_is_not_requested(tmp_path: Path) -> None:
    native, stage1 = _case(tmp_path)
    output = rebuild_registered_quantitative_inputs(native_maps_root=native, registration_root=stage1, output=tmp_path / "registered", active_stacks=["sax", "2ch"])
    with pytest.raises(FileExistsError):
        rebuild_registered_quantitative_inputs(native_maps_root=native, registration_root=stage1, output=output, active_stacks=["sax", "2ch"])
    command = build_registration_commands(tmp_path / "inputs", ["sax"], tmp_path / "reg", nesvor_executable="nesvor", device="-1")[0]
    assert "/4ch/" not in command


def _selection_fixture(root: Path, *, sparse_sha: str = "standardized-sha", inactive: bool = False) -> tuple[Path, dict, dict]:
    standardized, output = root / "standardized", root / "selection"
    standardized.mkdir(parents=True)
    standardization = {"stacks": {}}
    reuse = {"_manifest_path": str(root / "reuse_manifest.json"), "standardization_manifest": {"path": str(root / "standardization.json")}, "stacks": {}}
    (root / "reuse_manifest.json").write_text("{}")
    for stack in ("sax", "2ch", "4ch"):
        source, standardized_observation = root / f"{stack}_source.npz", root / f"{stack}_standardized.npz"
        source.write_bytes(b"pre"); standardized_observation.write_bytes(b"post")
        standardization["stacks"][stack] = {"source_observation": {"path": str(source), "sha256": "pre-sha"}, "output_observation": {"path": str(standardized_observation), "sha256": "standardized-sha"}}
        mapping = [{"derived_group_idx": group, "original_group_idx": group + 1} for group in range(12)]
        reuse["stacks"][stack] = {"standardized_to_original_group_idx": mapping}
        folder = standardized / stack; folder.mkdir()
        values = np.arange(12, dtype=np.float32)[:, None, None]
        np.savez_compressed(folder / "native_maps.npz", group_idx=np.arange(12), t1_ms=values, t2_ms=values, b1=values)
        np.savez_compressed(folder / "geometry.npz", affine_lps_rc=np.repeat(np.eye(4)[None], 12, axis=0), pixel_spacing_rc_mm=np.ones((12, 2)), slice_thickness_mm=np.ones(12))
    (root / "standardization.json").write_text(json.dumps(standardization))
    sparse = {"_manifest_path": str(root / "sparse.json"), "stacks": {}}
    (root / "sparse.json").write_text("{}")
    for stack in ("sax", "2ch", "4ch"):
        retained = [] if inactive and stack == "2ch" else [0, 2, 4, 6, 8, 10]
        sparse["stacks"][stack] = {"source_observation": {"path": str(root / f"{stack}_standardized.npz"), "sha256": sparse_sha}, "retained_original_group_ids": retained, "training_stack_present": bool(retained)}
    return standardized, reuse, sparse


def test_cyj_style_14_to_12_selection_uses_standardized_ids_and_records_prestandardized_provenance(tmp_path: Path) -> None:
    standardized, reuse, sparse = _selection_fixture(tmp_path)
    native, active, records = _selected_standardized_maps(standardized_root=standardized, reuse=reuse, sparse=sparse, output=tmp_path / "selection")
    assert active == ["sax", "2ch", "4ch"]
    assert np.load(native / "sax/native_maps.npz")["group_idx"].tolist() == [0, 2, 4, 6, 8, 10]
    assert records["sax"]["group_identity"] == [{"standardized_group_idx": 0, "original_prestandardization_group_idx": 1}, {"standardized_group_idx": 2, "original_prestandardization_group_idx": 3}, {"standardized_group_idx": 4, "original_prestandardization_group_idx": 5}, {"standardized_group_idx": 6, "original_prestandardization_group_idx": 7}, {"standardized_group_idx": 8, "original_prestandardization_group_idx": 9}, {"standardized_group_idx": 10, "original_prestandardization_group_idx": 11}]


def test_selection_rejects_prestandardized_source_sha_and_skips_inactive_stack(tmp_path: Path) -> None:
    standardized, reuse, sparse = _selection_fixture(tmp_path / "bad", sparse_sha="pre-sha")
    with pytest.raises(ValueError, match="standardized output"):
        _selected_standardized_maps(standardized_root=standardized, reuse=reuse, sparse=sparse, output=tmp_path / "bad/selection")
    standardized, reuse, sparse = _selection_fixture(tmp_path / "inactive", inactive=True)
    _native, active, records = _selected_standardized_maps(standardized_root=standardized, reuse=reuse, sparse=sparse, output=tmp_path / "inactive/selection")
    assert "2ch" not in active and records["2ch"]["training_stack_present"] is False


def test_prepared_root_must_equal_sparse_prepared_root(tmp_path: Path) -> None:
    expected = tmp_path / "expected"; expected.mkdir()
    assert _validate_prepared_root(expected, {"prepared_root": str(expected)}) == expected.resolve()
    with pytest.raises(ValueError, match="prepared_root"):
        _validate_prepared_root(tmp_path / "wrong", {"prepared_root": str(expected)})


@pytest.mark.parametrize("field", ["no_transformation_optimization", "no_slice_scale"])
def test_final_audit_rejects_missing_stage2_freeze_evidence(tmp_path: Path, field: str) -> None:
    root = tmp_path / "route"; root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"schema": "phase4r_intensity_preserving_fit_first/v1"}))
    (root / "registered_quantitative_inputs_gate.json").write_text(json.dumps({"status": "REGISTERED_QUANTITATIVE_INPUTS_VALID"}))
    for parameter in ("t1", "t2"):
        stage1, stage2 = root / f"stage1_registration/{parameter}", root / f"stage2_reconstruction/{parameter}"
        stage1.mkdir(parents=True); stage2.mkdir(parents=True)
        (stage1 / "result.json").write_text(json.dumps({"registration": "stack"}))
        values = {"input_slices": str((root / f"registered_quantitative_inputs/{parameter}/slices").resolve()), "no_transformation_optimization": True, "no_slice_scale": True, "input_stacks": None}
        values.pop(field)
        (stage2 / "result.json").write_text(json.dumps(values))
    with pytest.raises(ValueError, match=field):
        audit_intensity_preserving_final_pose(root=root)


def test_final_audit_requires_stage2_final_affines_to_equal_frozen_inputs(tmp_path: Path) -> None:
    root = tmp_path / "route"; root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"schema": "phase4r_intensity_preserving_fit_first/v1"}))
    (root / "registered_quantitative_inputs_gate.json").write_text(json.dumps({"status": "REGISTERED_QUANTITATIVE_INPUTS_VALID"}))
    for parameter in ("t1", "t2"):
        input_folder = root / f"registered_quantitative_inputs/{parameter}/slices"; output_folder = root / f"stage2_reconstruction/{parameter}/slices"
        input_folder.mkdir(parents=True); output_folder.mkdir(parents=True)
        affine = np.eye(4); affine[0, 3] = 3 if parameter == "t1" else 7
        nib.save(nib.Nifti1Image(np.zeros((2, 2, 1)), affine), input_folder / "0.nii.gz")
        altered = affine.copy(); altered[0, 3] += .01
        nib.save(nib.Nifti1Image(np.zeros((2, 2, 1)), altered), output_folder / "0.nii.gz")
        stage1 = root / f"stage1_registration/{parameter}"; stage1.mkdir(parents=True)
        (stage1 / "result.json").write_text(json.dumps({"registration": "stack"}))
        result = {"input_slices": str(input_folder.resolve()), "input_stacks": None, "output_slices": str(output_folder.resolve()), "no_transformation_optimization": True, "no_slice_scale": True}
        (root / f"stage2_reconstruction/{parameter}/result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="final affine"):
        audit_intensity_preserving_final_pose(root=root)
