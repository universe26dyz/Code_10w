"""Final5 CYJ Experiment 1.1 frozen-map, one-stage Fit-first reconstruction.

This formal route consumes existing standardized native dictionary maps.  It
never invokes dictionary fitting and uses the sparse manifest only to select
authoritative retained group identities.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import sha256, write_immutable_json
from .fit_first import build_independent_nesvor_commands, export_nesvor_native_inputs
from .sparse_profile_manifest import validate_sparse_profile_manifest


STACKS = ("sax", "2ch", "4ch")
SCHEMA = "phase4r_exp1_1_fit_first/v1"
SELECTION_SCHEMA = "phase4r_exp1_1_selected_native_maps/v1"
NORMALIZATION = {"t1": 2500.0, "t2": 200.0}


def _record(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _sparse_provenance(profile: dict[str, Any]) -> dict[str, str]:
    """Build one fail-closed sparse identity used by input and final manifests."""
    path = Path(str(profile.get("_manifest_path", "")))
    expected_sha, generator_git_sha = profile.get("_manifest_sha256"), profile.get("git_sha")
    if not path.is_file() or not isinstance(expected_sha, str) or len(expected_sha) != 64 or any(char not in "0123456789abcdef" for char in expected_sha):
        raise ValueError("formal Fit-first profile lacks a valid sparse-manifest SHA256")
    if not isinstance(generator_git_sha, str) or len(generator_git_sha) != 40 or any(char not in "0123456789abcdef" for char in generator_git_sha):
        raise ValueError("formal Fit-first profile lacks a valid sparse generator git SHA")
    record = _record(path)
    if record["sha256"] != expected_sha:
        raise ValueError("formal Fit-first sparse-manifest SHA256 disagrees with validated profile")
    return {**record, "generator_git_sha": generator_git_sha}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: JSON object required")
    return payload


def _select_stack(native_root: Path, stack: str, sparse_stack: dict[str, Any], target: Path) -> tuple[bool, list[int], dict[str, Any]]:
    retained = sparse_stack.get("retained_original_group_ids")
    active = sparse_stack.get("training_stack_present")
    if not isinstance(retained, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in retained) or len(retained) != len(set(retained)):
        raise ValueError(f"{stack}: sparse retained group IDs must be unique integers")
    if active is not bool(retained):
        raise ValueError(f"{stack}: sparse active-stack flag disagrees with retained group IDs")
    if not active:
        return False, [], {"training_stack_present": False, "selected_group_ids": []}
    source = sparse_stack.get("source_observation")
    if not isinstance(source, dict):
        raise ValueError(f"{stack}: sparse source-observation evidence is absent")
    stack_root = native_root / stack
    maps_path, geometry_path, manifest_path = stack_root / "native_maps.npz", stack_root / "geometry.npz", stack_root / "manifest.json"
    native_manifest = _read_json(manifest_path)
    if native_manifest.get("native_maps_sha256") != sha256(maps_path) or native_manifest.get("geometry_sha256") != sha256(geometry_path):
        raise ValueError(f"{stack}: native dictionary map or geometry SHA256 mismatch")
    native_source = native_manifest.get("source_observations")
    if not isinstance(native_source, dict) or native_source.get("sha256") != source.get("sha256") or Path(str(native_source.get("path", ""))).resolve() != Path(str(source.get("path", ""))).resolve():
        raise ValueError(f"{stack}: native dictionary and sparse manifest source observations disagree")
    source_path = Path(str(source.get("path", "")))
    if not source_path.is_file() or sha256(source_path) != source.get("sha256"):
        raise ValueError(f"{stack}: sparse source observation is unavailable or has changed")
    with np.load(maps_path, allow_pickle=False) as maps, np.load(geometry_path, allow_pickle=False) as geometry:
        needed_maps = ("t1_ms", "t2_ms", "b1", "group_idx")
        needed_geometry = ("affine_lps_rc", "pixel_spacing_rc_mm", "slice_thickness_mm")
        if any(name not in maps.files for name in needed_maps) or any(name not in geometry.files for name in needed_geometry):
            raise ValueError(f"{stack}: native dictionary maps or geometry are incomplete")
        groups = [int(value) for value in maps["group_idx"]]
        if len(groups) != len(set(groups)):
            raise ValueError(f"{stack}: native dictionary group IDs must be unique")
        positions = []
        for group in retained:
            if groups.count(group) != 1:
                raise ValueError(f"{stack}: retained sparse group {group} is not represented exactly once in frozen maps")
            positions.append(groups.index(group))
        if any(np.asarray(maps[name]).shape[0] != len(groups) for name in needed_maps[:-1]) or any(np.asarray(geometry[name]).shape[0] != len(groups) for name in needed_geometry):
            raise ValueError(f"{stack}: frozen map/geometry arrays disagree with group IDs")
        target.mkdir(parents=True)
        np.savez_compressed(target / "native_maps.npz", **{name: np.asarray(maps[name])[positions].copy() for name in needed_maps})
        np.savez_compressed(target / "geometry.npz", **{name: np.asarray(geometry[name])[positions].copy() for name in needed_geometry})
    return True, [int(value) for value in retained], {
        "training_stack_present": True,
        "selected_group_ids": [int(value) for value in retained],
        "source_native_maps": _record(maps_path),
        "source_geometry": _record(geometry_path),
        "source_native_manifest": _record(manifest_path),
        "source_observation": {"path": str(source_path.resolve()), "sha256": str(source["sha256"])},
        "selected_native_maps": _record(target / "native_maps.npz"),
        "selected_geometry": _record(target / "geometry.npz"),
    }


def select_frozen_native_maps(native_root: str | Path, sparse_manifest: str | Path, output: str | Path, *, subject: str,
                              expected_generator_git_sha: str | None = None, expected_manifest_sha256: str | None = None) -> dict[str, list[int]]:
    """Write a fresh deterministic subset of frozen native maps; never refit."""
    native_root, manifest_path, output = Path(native_root), Path(sparse_manifest), Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    strict = expected_generator_git_sha is not None or expected_manifest_sha256 is not None
    if strict and not (expected_generator_git_sha and expected_manifest_sha256):
        raise ValueError("formal sparse route requires both generator and manifest SHA256 pins")
    profile = validate_sparse_profile_manifest(manifest_path, subject=subject, strict=strict, expected_generator_git_sha=expected_generator_git_sha, expected_manifest_sha256=expected_manifest_sha256)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        records: dict[str, Any] = {}
        selected: dict[str, list[int]] = {}
        for stack in STACKS:
            active, groups, record = _select_stack(native_root, stack, profile["stacks"][stack], staging / stack)
            selected[stack] = groups
            records[stack] = record
        write_immutable_json(staging / "manifest.json", {
            "schema": SELECTION_SCHEMA, "subject": profile["subject"], "case_id": profile["case_id"],
            "profile_id": profile["profile_id"], "fold_id": profile.get("fold_id"), "fold_count": profile.get("fold_count"),
            "sparse_manifest": _sparse_provenance(profile), "dictionary_refit": False,
            "selection_semantics": "sparse_manifest retained_original_group_ids matched directly to frozen native_maps group_idx",
            "stacks": records,
        })
        output.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return selected


def build_exp1_1_nesvor_commands(nesvor_inputs: list[str | Path], output: str | Path, *, device: str | int,
                                 nesvor_executable: str | Path, sample_masks: dict[str, str | Path] | None = None) -> list[str]:
    """Return independent one-stage T1/T2 commands with optional output sample masks."""
    return build_independent_nesvor_commands(nesvor_inputs, output, device=device, nesvor_executable=nesvor_executable, sample_masks=sample_masks)


def _finalize(output: Path, profile: dict[str, Any], selected_root: Path, commands: list[str], output_sampling: dict[str, Any]) -> Path:
    parameters = {}
    for parameter in ("t1", "t2"):
        target = output / "nesvor" / parameter
        model, result, slices = target / "model.pt", target / "result.json", target / "slices"
        if not model.is_file() or not result.is_file() or not slices.is_dir() or not any(slices.glob("[0-9]*.nii.gz")):
            raise FileNotFoundError(f"{parameter}: NeSVoR model/result/final slices are incomplete")
        parameters[parameter] = {"model": _record(model), "result": _record(result), "final_slices": str(slices.resolve()), "volume_ms": _record(target / "volume_ms.nii.gz")}
    selected_manifest = selected_root / "manifest.json"
    return write_immutable_json(output / "manifest.json", {
        "schema": SCHEMA, "scientific_route": "CYJ Experiment 1.1 mapping-to-one-stage-NeSVoR",
        "subject": profile["subject"], "case_id": profile["case_id"], "profile_id": profile["profile_id"],
        "fold_id": profile.get("fold_id"), "fold_count": profile.get("fold_count"),
        "sparse_manifest": _sparse_provenance(profile), "dictionary_refit": False,
        "no_dictionary_refit": True, "independent_t1_t2": True, "transformation_optimization_enabled": True,
        "no_slice_scale": True, "active_stacks": [stack for stack in STACKS if (selected_root / stack).is_dir()],
        "output_sampling": output_sampling,
        "selected_native_maps": _record(selected_manifest), "normalization": {"t1_divide_multiply_ms": 2500, "t2_divide_multiply_ms": 200},
        "commands": _record(output / "commands.sh"), "exact_nesvor_commands": commands,
        "nesvor_version": "nesvor_v5_128_w1", "parameters": parameters,
    })


def run_exp1_1_fit_first(*, subject: str, sparse_manifest: str | Path, standardized_native_dictionary_root: str | Path, output: str | Path,
                          nesvor_executable: str | Path, device: str | int, expected_generator_git_sha: str, expected_manifest_sha256: str, execute: bool) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    profile = validate_sparse_profile_manifest(sparse_manifest, subject=subject, strict=True, expected_generator_git_sha=expected_generator_git_sha, expected_manifest_sha256=expected_manifest_sha256)
    output.mkdir(parents=True)
    try:
        select_frozen_native_maps(standardized_native_dictionary_root, sparse_manifest, output / "selected_native_maps", subject=subject, expected_generator_git_sha=expected_generator_git_sha, expected_manifest_sha256=expected_manifest_sha256)
        active = [stack for stack in STACKS if (output / "selected_native_maps" / stack).is_dir()]
        if not active:
            raise ValueError("formal Fit-first route has no active sparse stacks")
        exports = []
        for stack in active:
            exports.append(export_nesvor_native_inputs(output / "selected_native_maps" / stack, output / "nesvor_inputs" / stack, NORMALIZATION))
        sample_masks = None
        output_sampling: dict[str, Any] = {
            "policy": "nesvor_default_estimated_mask",
            "scope": "NeSVoR output INR sampling only; not a reconstruction/training input mask",
        }
        if len(active) == 1:
            single_stack = active[0]
            sample_masks = {
                parameter: output / "nesvor_inputs" / single_stack / parameter / "stack_mask.nii.gz"
                for parameter in ("t1", "t2")
            }
            output_sampling = {
                "policy": "explicit_single_stack_native_finite_support",
                "scope": "NeSVoR output INR sampling only; not a reconstruction/training input mask",
                "sample_masks": {parameter: _record(path) for parameter, path in sample_masks.items()},
            }
        commands = build_exp1_1_nesvor_commands(exports, output / "nesvor", device=device, nesvor_executable=nesvor_executable, sample_masks=sample_masks)
        (output / "commands.sh").write_text("\n".join(commands) + "\n", encoding="utf-8")
        write_immutable_json(output / "input_manifest.json", {"schema": SCHEMA, "subject": subject, "case_id": profile["case_id"], "dictionary_refit": False, "sparse_manifest": _sparse_provenance(profile), "selected_native_maps": _record(output / "selected_native_maps" / "manifest.json"), "commands": _record(output / "commands.sh")})
        if execute:
            for parameter in ("t1", "t2"):
                (output / "nesvor" / parameter).mkdir(parents=True, exist_ok=False)
            for command in commands:
                subprocess.run(command, shell=True, check=True)
            _finalize(output, profile, output / "selected_native_maps", commands, output_sampling)
        return output
    except BaseException:
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True); parser.add_argument("--sparse-manifest", required=True)
    parser.add_argument("--standardized-native-dictionary-root", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--nesvor", required=True); parser.add_argument("--device", required=True)
    parser.add_argument("--expected-generator-git-sha", required=True); parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(run_exp1_1_fit_first(subject=args.subject, sparse_manifest=args.sparse_manifest, standardized_native_dictionary_root=args.standardized_native_dictionary_root, output=args.output, nesvor_executable=args.nesvor, device=args.device, expected_generator_git_sha=args.expected_generator_git_sha, expected_manifest_sha256=args.expected_manifest_sha256, execute=args.execute))


if __name__ == "__main__":
    main()
