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
    (native / "manifest.json").write_text(json.dumps({"schema": "phase4r_fit_first_selected_standardized_native_maps/v1", "stacks": {stack: {"training_stack_present": True, "output_maps": {"sha256": sha256(native / stack / "native_maps.npz")}} for stack in ("sax", "2ch")}}))
    return native, stage1


def test_q99_bug_multiplier_is_explicit() -> None:
    assert q99_normalisation_multiplier(np.array([0.85] * 100)) == pytest.approx(1 / 0.85)


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
    stage2 = _reconstruct_args(tmp_path / "registered/t1/slices", tmp_path / "out", "t1", nesvor_executable="nesvor", device="-1")
    assert "--input-slices" in stage2 and "--registration" not in stage2


def test_rebuild_refuses_existing_output_and_inactive_stack_is_not_requested(tmp_path: Path) -> None:
    native, stage1 = _case(tmp_path)
    output = rebuild_registered_quantitative_inputs(native_maps_root=native, registration_root=stage1, output=tmp_path / "registered", active_stacks=["sax", "2ch"])
    with pytest.raises(FileExistsError):
        rebuild_registered_quantitative_inputs(native_maps_root=native, registration_root=stage1, output=output, active_stacks=["sax", "2ch"])
    command = build_registration_commands(tmp_path / "inputs", ["sax"], tmp_path / "reg", nesvor_executable="nesvor", device="-1")[0]
    assert "/4ch/" not in command
