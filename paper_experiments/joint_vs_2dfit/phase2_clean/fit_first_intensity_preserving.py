"""Two-stage, intensity-preserving Fit-first NeSVoR route.

Stage 1 deliberately uses NeSVoR stack registration only to estimate poses.
Stage 2 consumes newly written individual slices whose affines are those final
poses but whose pixels are the authoritative dictionary values.  It therefore
does not invoke NeSVoR registration a second time.
"""
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import HISTORICAL_NESVOR_EXECUTABLE_DEFAULT, sha256, write_immutable_json
from .fit_first import STACKS, export_nesvor_native_inputs
from .fit_first_native_pose_audit import _GEOMETRY_EQUIVALENCE_TOL_MM, _VALUE_CORRESPONDENCE_TOL
from .reuse_standardized_native_dictionary_maps import validate_standardized_native_dictionary_reuse
from .sparse_profile_manifest import validate_sparse_profile_manifest

PARAMETERS = ("t1", "t2")
SCALES = {"t1": 2500.0, "t2": 200.0}
VALUE_TOLERANCE = _VALUE_CORRESPONDENCE_TOL
REGISTRATION_SCHEMA = "phase4r_fit_first_registration_stage/v1"
INPUT_SCHEMA = "phase4r_fit_first_registered_quantitative_inputs/v1"
ROUTE_SCHEMA = "phase4r_intensity_preserving_fit_first/v1"


def q99_normalisation_multiplier(values: np.ndarray, foreground: np.ndarray | None = None) -> float:
    """Expose the Stage-1 q99 effect for regression evidence, not correction."""
    value_array = np.asarray(values, float)
    support = np.isfinite(value_array) if foreground is None else (np.isfinite(value_array) & np.asarray(foreground, bool))
    finite = value_array[support]
    if not finite.size:
        raise ValueError("q99 normalisation requires finite foreground")
    q99 = float(np.quantile(finite, 0.99))
    if q99 <= 0:
        raise ValueError("q99 normalisation quantile must be positive")
    return 1.0 / q99


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _slice_paths(folder: Path) -> list[Path]:
    paths = sorted(folder.glob("[0-9]*.nii.gz"), key=lambda p: int(p.name.split(".", 1)[0]))
    ids = [int(p.name.split(".", 1)[0]) for p in paths]
    if not paths or ids != list(range(len(paths))):
        raise ValueError(f"{folder}: stage-1 slices must have contiguous zero-based NIfTI IDs")
    return paths


def _flip_for_affine(affine: np.ndarray) -> bool:
    return bool(np.linalg.det(np.asarray(affine, float)[:3, :3]) < 0)


def _physical_slice_geometry_equivalence(initial_path: Path, final_path: Path) -> dict[str, Any]:
    """Compare NIfTI slice geometry in physical RAS space, not raw floats."""
    import nibabel as nib

    initial, final = nib.load(str(initial_path)), nib.load(str(final_path))
    if initial.shape != final.shape or len(initial.shape) != 3 or initial.shape[2] != 1:
        raise ValueError(f"{final_path}: Stage-2 final slice shape differs from frozen input")
    shape = initial.shape
    points = np.asarray(((0., 0., 0., 1.), (shape[0] - 1., 0., 0., 1.), (0., shape[1] - 1., 0., 1.), (shape[0] - 1., shape[1] - 1., 0., 1.), ((shape[0] - 1.) / 2., (shape[1] - 1.) / 2., 0., 1.)))
    initial_world, final_world = (np.asarray(initial.affine, float) @ points.T).T[:, :3], (np.asarray(final.affine, float) @ points.T).T[:, :3]
    errors = np.linalg.norm(initial_world - final_world, axis=1)
    spacing_error = float(max(abs(np.linalg.norm(initial.affine[:3, axis]) - np.linalg.norm(final.affine[:3, axis])) for axis in (0, 1)))
    raw = float(np.max(np.abs(np.asarray(initial.affine, float) - np.asarray(final.affine, float))))
    return {"max_world_error_mm": float(errors.max()), "corner_max_error_mm": float(errors[:4].max()), "center_error_mm": float(errors[4]), "spacing_error_mm": spacing_error, "raw_affine_max_abs_error": raw, "geometry_equivalence_tolerance_mm": _GEOMETRY_EQUIVALENCE_TOL_MM, "status": "PASS_GEOMETRY_EQUIVALENT" if max(float(errors.max()), spacing_error) <= _GEOMETRY_EQUIVALENCE_TOL_MM else "FAIL_GEOMETRY_MISMATCH"}


def _validate_prepared_root(prepared_root: str | Path | None, sparse: dict[str, Any] | None) -> Path | None:
    """Admit the wrapper's prepared root only when it is manifest evidence."""
    if prepared_root is None:
        return None
    if sparse is None or not isinstance(sparse.get("prepared_root"), str):
        raise ValueError("prepared_root requires a validated sparse-profile prepared_root")
    supplied, expected = Path(prepared_root).resolve(), Path(sparse["prepared_root"]).resolve()
    if supplied != expected:
        raise ValueError(f"prepared_root {supplied} does not match validated sparse manifest {expected}")
    return supplied


def _selected_standardized_maps(*, standardized_root: Path, reuse: dict[str, Any], sparse: dict[str, Any] | None, output: Path) -> tuple[Path, list[str], dict[str, Any]]:
    """Materialise active groups in *standardized* group-ID space.

    Pre-standardization IDs are audit provenance only; sparse selection is
    entirely over the IDs present in standardized observations/native maps.
    """
    native = output / "native_maps"
    native.mkdir(parents=True)
    records: dict[str, Any] = {}
    active: list[str] = []
    standardization = json.loads(Path(reuse["standardization_manifest"]["path"]).read_text(encoding="utf-8"))
    for stack in STACKS:
        mapping = reuse["stacks"][stack]["standardized_to_original_group_idx"]
        standardized_ids = [int(item["derived_group_idx"]) for item in mapping]
        prestandardized_by_standardized = {int(item["derived_group_idx"]): int(item["original_group_idx"]) for item in mapping}
        if sparse is None:
            retained = standardized_ids
            present = True
        else:
            item = sparse["stacks"][stack]
            output_observation = standardization["stacks"][stack]["output_observation"]
            sparse_source = item["source_observation"]
            if sparse_source["sha256"] != output_observation["sha256"] or Path(sparse_source["path"]).resolve() != Path(output_observation["path"]).resolve():
                raise ValueError(f"{stack}: sparse source identity must match standardized output observation")
            retained = [int(x) for x in item["retained_original_group_ids"]]
            present = bool(item["training_stack_present"])
            if present != bool(retained):
                raise ValueError(f"{stack}: sparse training_stack_present disagrees with retained groups")
        if any(group not in standardized_ids for group in retained) or retained != [group for group in standardized_ids if group in retained]:
            raise ValueError(f"{stack}: retained groups are not the standardized physical source order")
        if not present:
            records[stack] = {"training_stack_present": False, "original_group_ids": []}
            continue
        source = standardized_root / stack
        with np.load(source / "native_maps.npz", allow_pickle=False) as maps, np.load(source / "geometry.npz", allow_pickle=False) as geometry:
            source_groups = [int(value) for value in maps["group_idx"]]
            if source_groups != standardized_ids:
                raise ValueError(f"{stack}: standardized native-map group IDs disagree with standardization mapping")
            positions = [source_groups.index(group) for group in retained]
            target = native / stack; target.mkdir()
            np.savez_compressed(target / "native_maps.npz", t1_ms=np.asarray(maps["t1_ms"])[positions], t2_ms=np.asarray(maps["t2_ms"])[positions], b1=np.asarray(maps["b1"])[positions], group_idx=np.asarray(retained, dtype=np.int64))
            np.savez_compressed(target / "geometry.npz", affine_lps_rc=np.asarray(geometry["affine_lps_rc"])[positions], pixel_spacing_rc_mm=np.asarray(geometry["pixel_spacing_rc_mm"])[positions], slice_thickness_mm=np.asarray(geometry["slice_thickness_mm"])[positions])
        records[stack] = {"training_stack_present": True, "standardized_group_ids": retained, "group_identity": [{"standardized_group_idx": group, "original_prestandardization_group_idx": prestandardized_by_standardized[group]} for group in retained], "source_maps": _record(source / "native_maps.npz"), "output_maps": _record(target / "native_maps.npz"), "output_geometry": _record(target / "geometry.npz")}
        active.append(stack)
    if not active:
        raise ValueError("Fit-first profile has no active training stack")
    write_immutable_json(native / "manifest.json", {"schema": "phase4r_fit_first_selected_standardized_native_maps/v1", "reconstruction_identity_space": "standardized_group_idx (standardized observation/native-map group_idx)", "prestandardization_identity_role": "per-group provenance only; never sparse selection", "stacks": records, "standardized_reuse_manifest": _record(Path(reuse["_manifest_path"])), "sparse_manifest": None if sparse is None else _record(Path(sparse["_manifest_path"])), "immutable": True})
    return native, active, records


def build_registration_commands(nesvor_inputs: Path, active_stacks: list[str], registration_root: Path, *, nesvor_executable: str | Path, device: str) -> list[str]:
    commands = []
    for parameter in PARAMETERS:
        target = registration_root / parameter
        args = [str(nesvor_executable), "register", "--input-stacks", *[str(nesvor_inputs / stack / parameter / "stack.nii.gz") for stack in active_stacks], "--thicknesses", *(["8"] * len(active_stacks)), "--registration", "stack", "--scanner-space", "--output-slices", str(target / "slices"), "--output-json", str(target / "result.json"), "--output-log", str(target / "nesvor.log"), "--device", str(device)]
        commands.append(" ".join(shlex.quote(value) for value in args))
    return commands


def _reconstruct_args(input_slices: Path, target: Path, parameter: str, *, nesvor_executable: str | Path, device: str, sample_mask: Path | None = None) -> list[str]:
    sample_mask_args = [] if sample_mask is None else ["--sample-mask", str(sample_mask)]
    return [str(nesvor_executable), "reconstruct", "--input-slices", str(input_slices), *sample_mask_args, "--weight-transformation", "1", "--weight-deform", "0.1", "--weight-image", "1", "--weight-pixel-var-smooth", "0", "--output-volume", str(target / "volume_normalized.nii.gz"), "--output-model", str(target / "model.pt"), "--output-json", str(target / "result.json"), "--output-log", str(target / "nesvor.log"), "--output-slices", str(target / "slices"), "--output-intensity-mean", "0", "--output-resolution", "1.0", "--output-psf-factor", "1", "--no-slice-scale", "--no-transformation-optimization", "--single-precision", "--log2-hashmap-size", "19", "--batch-size", "2000", "--n-iter", "10000", "--n-samples", "256", "--n-inference-samples", "128", "--inference-batch-size", "16000", "--pixel-variance-weight", "1", "--coarsest-resolution", "16", "--finest-resolution", "0.5", "--level-scale", "1.3819", "--n-features-per-level", "2", "--depth", "1", "--width", "64", "--n-features-z", "15", "--n-features-slice", "16", "--weight-bias", "100", "--image-regularization", "edge", "--delta", "0.2", "--learning-rate", "0.005", "--gamma", "0.33", "--milestones", "0.5", "0.75", "0.9", "--seed", "20260823", "--device", str(device)]


def rebuild_registered_quantitative_inputs(*, native_maps_root: str | Path, registration_root: str | Path, output: str | Path, active_stacks: list[str]) -> Path:
    """Write Stage-2 slices with Stage-1 affines and dictionary pixels only."""
    import nibabel as nib
    native, registration, destination = Path(native_maps_root), Path(registration_root), Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        records: dict[str, list[dict[str, Any]]] = {}
        native_manifest = json.loads((native / "manifest.json").read_text(encoding="utf-8"))
        for parameter in PARAMETERS:
            final = _slice_paths(registration / parameter / "slices")
            ordered: list[tuple[str, int, np.ndarray]] = []
            for stack in active_stacks:
                identities = {int(record["standardized_group_idx"]): int(record["original_prestandardization_group_idx"]) for record in native_manifest["stacks"][stack]["group_identity"]}
                with np.load(native / stack / "native_maps.npz", allow_pickle=False) as maps:
                    for index, group in enumerate(np.asarray(maps["group_idx"], int)):
                        if int(group) not in identities:
                            raise ValueError(f"{stack}: standardized group lacks pre-standardization provenance")
                        ordered.append((stack, int(group), identities[int(group)], np.asarray(maps[f"{parameter}_ms"][index], np.float32)))
            if len(final) != len(ordered):
                raise ValueError(f"{parameter}: stage-1 output-slice count disagrees with active native groups")
            folder = staging / parameter / "slices"; folder.mkdir(parents=True)
            records[parameter] = []
            for index, ((stack, group, prestandardized_group, values_ms), final_slice) in enumerate(zip(ordered, final)):
                image = nib.load(str(final_slice)); affine = np.asarray(image.affine, float)
                if image.shape[:2] != values_ms.shape or image.shape[2:] != (1,):
                    raise ValueError(f"{parameter}/{stack}/{group}: Stage-1 slice shape is not source geometry")
                expected = np.where(np.isfinite(values_ms), values_ms / SCALES[parameter], 0.0).astype(np.float32)
                stored = expected[::-1] if _flip_for_affine(affine) else expected
                target = folder / f"{index}.nii.gz"; nib.save(nib.Nifti1Image(stored[:, :, None], affine), str(target))
                records[parameter].append({"index": index, "stack": stack, "standardized_group_idx": group, "original_prestandardization_group_idx": prestandardized_group, "stage1_final_slice": _record(final_slice), "registered_input_slice": _record(target), "affine_ras": affine.tolist(), "flip_original_row": _flip_for_affine(affine), "source_support_voxels": int(np.isfinite(values_ms).sum())})
        write_immutable_json(staging / "manifest.json", {"schema": INPUT_SCHEMA, "native_maps_root": str(native.resolve()), "registration_stage": str(registration.resolve()), "parameters": records, "value_policy": "finite dictionary ms / parameter scale; nonfinite dictionary background is exactly zero", "geometry_policy": "each Stage-2 affine is copied from its corresponding Stage-1 final registered slice", "immutable": True})
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True); raise
    return destination


def validate_registered_quantitative_inputs(*, native_maps_root: str | Path, registration_root: str | Path, inputs_root: str | Path, active_stacks: list[str]) -> dict[str, Any]:
    """Fail-closed gate immediately before Stage-2 reconstruction."""
    import nibabel as nib
    native, registration, inputs = Path(native_maps_root), Path(registration_root), Path(inputs_root)
    manifest_path = inputs / "manifest.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != INPUT_SCHEMA:
        raise ValueError("registered quantitative input schema mismatch")
    if Path(manifest.get("native_maps_root", "")).resolve() != native.resolve() or Path(manifest.get("registration_stage", "")).resolve() != registration.resolve():
        raise ValueError("registered quantitative input provenance-root mismatch")
    native_manifest = json.loads((native / "manifest.json").read_text(encoding="utf-8"))
    if native_manifest.get("schema") != "phase4r_fit_first_selected_standardized_native_maps/v1":
        raise ValueError("selected native-map provenance schema mismatch")
    result: dict[str, Any] = {"schema": INPUT_SCHEMA, "status": "REGISTERED_QUANTITATIVE_INPUTS_VALID", "parameters": {}}
    for parameter in PARAMETERS:
        expected: list[tuple[str, int, int, np.ndarray]] = []
        for stack in active_stacks:
            stack_record = native_manifest.get("stacks", {}).get(stack, {})
            if stack_record.get("training_stack_present") is not True or stack_record.get("output_maps", {}).get("sha256") != sha256(native / stack / "native_maps.npz"):
                raise ValueError(f"{stack}: selected native-map provenance mismatch")
            identities = {int(record["standardized_group_idx"]): int(record["original_prestandardization_group_idx"]) for record in stack_record.get("group_identity", [])}
            with np.load(native / stack / "native_maps.npz", allow_pickle=False) as maps:
                expected.extend((stack, int(group), identities[int(group)], np.asarray(maps[f"{parameter}_ms"][i], np.float32)) for i, group in enumerate(np.asarray(maps["group_idx"], int)))
        records = manifest.get("parameters", {}).get(parameter)
        slices = _slice_paths(inputs / parameter / "slices")
        stage1 = _slice_paths(registration / parameter / "slices")
        if not isinstance(records, list) or len(records) != len(expected) or len(slices) != len(expected) or len(stage1) != len(expected):
            raise ValueError(f"{parameter}: identity/order/count gate failed")
        max_error = 0.0
        for index, ((stack, group, prestandardized_group, values_ms), record, path, registered) in enumerate(zip(expected, records, slices, stage1)):
            if (record.get("index"), record.get("stack"), record.get("standardized_group_idx"), record.get("original_prestandardization_group_idx")) != (index, stack, group, prestandardized_group):
                raise ValueError(f"{parameter}: stack/group order gate failed")
            if record.get("registered_input_slice", {}).get("sha256") != sha256(path) or record.get("stage1_final_slice", {}).get("sha256") != sha256(registered):
                raise ValueError(f"{parameter}/{stack}/{group}: immutable slice provenance mismatch")
            got = np.asarray(nib.load(str(path)).get_fdata(), float)
            final = np.asarray(nib.load(str(registered)).affine, float)
            if got.shape != (*values_ms.shape, 1) or not np.allclose(np.asarray(nib.load(str(path)).affine, float), final, atol=VALUE_TOLERANCE, rtol=0):
                raise ValueError(f"{parameter}/{stack}/{group}: registered affine gate failed")
            expected_values = np.where(np.isfinite(values_ms), values_ms / SCALES[parameter], 0.0)
            original_order = got[::-1, :, 0] if _flip_for_affine(final) else got[:, :, 0]
            error = float(np.max(np.abs(original_order - expected_values)))
            if error > VALUE_TOLERANCE or not np.isfinite(got).all():
                raise ValueError(f"{parameter}/{stack}/{group}: quantitative pixel gate failed")
            max_error = max(max_error, error)
        result["parameters"][parameter] = {"slice_count": len(expected), "max_abs_value_error": max_error, "scale": SCALES[parameter]}
    return result


def audit_intensity_preserving_final_pose(*, root: str | Path) -> Path:
    """Record the new final-pose meaning without mislabelling Stage 2 as registration.

    Stage-1 ``output-slices`` are the only authoritative final-pose artifact.
    Stage-2 consumes those frozen affines with restored quantitative pixels;
    its output slices are reconstruction outputs, not a second optimisation.
    """
    base = Path(root)
    route = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    gate_path = base / "registered_quantitative_inputs_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if route.get("schema") != ROUTE_SCHEMA or gate.get("status") != "REGISTERED_QUANTITATIVE_INPUTS_VALID":
        raise ValueError("intensity-preserving final-pose audit requires a valid route and quantitative-input gate")
    records: dict[str, Any] = {}
    for parameter in PARAMETERS:
        stage1_result = base / "stage1_registration" / parameter / "result.json"
        stage2_result = base / "stage2_reconstruction" / parameter / "result.json"
        if not stage1_result.is_file() or not stage2_result.is_file():
            raise FileNotFoundError(f"{parameter}: missing stage result provenance")
        registration = json.loads(stage1_result.read_text(encoding="utf-8")); reconstruction = json.loads(stage2_result.read_text(encoding="utf-8"))
        stage1_args, stage2_args = registration.get("args", registration), reconstruction.get("args", reconstruction)
        if stage1_args.get("registration") != "stack":
            raise ValueError(f"{parameter}: Stage 1 does not prove stack registration")
        expected = str((base / "registered_quantitative_inputs" / parameter / "slices").resolve())
        if str(stage2_args.get("input_slices", "")) != expected:
            raise ValueError(f"{parameter}: Stage 2 was not run from the gated registered quantitative slices")
        if stage2_args.get("input_stacks") is not None:
            raise ValueError(f"{parameter}: Stage 2 must not consume input_stacks")
        for field in ("no_transformation_optimization", "no_slice_scale"):
            if stage2_args.get(field) is not True:
                raise ValueError(f"{parameter}: Stage 2 result does not prove {field}=true")
        expected_output = str((base / "stage2_reconstruction" / parameter / "slices").resolve())
        if str(stage2_args.get("output_slices", "")) != expected_output:
            raise ValueError(f"{parameter}: Stage 2 output_slices provenance mismatch")
        # NeSVoR's float32 load/save can perturb raw NIfTI affine elements.
        # Formal admission therefore compares physical sampled points and
        # spacing, matching the historical native-pose geometry contract.
        inputs, outputs = _slice_paths(Path(expected)), _slice_paths(Path(expected_output))
        if len(inputs) != len(outputs):
            raise ValueError(f"{parameter}: Stage-2 input/output slice count mismatch")
        geometry_records = []
        for initial, final in zip(inputs, outputs):
            geometry = _physical_slice_geometry_equivalence(initial, final)
            if geometry["status"] != "PASS_GEOMETRY_EQUIVALENT":
                raise ValueError(f"{parameter}: Stage-2 final geometry differs from frozen registered input")
            geometry_records.append(geometry)
        records[parameter] = {"stage1_registration": _record(stage1_result), "stage2_reconstruction": _record(stage2_result), "stage1_pose_semantics": "authoritative final rigid pose; optimisation enabled", "stage2_pose_semantics": "no registration and no transformation optimisation; frozen Stage-1 affine is consumed through --input-slices", "stage2_input_slices": expected, "stage2_input_vs_final": {"geometry_equivalence_tolerance_mm": _GEOMETRY_EQUIVALENCE_TOL_MM, "max_world_error_mm": max(item["max_world_error_mm"] for item in geometry_records), "corner_max_error_mm": max(item["corner_max_error_mm"] for item in geometry_records), "center_error_mm": max(item["center_error_mm"] for item in geometry_records), "spacing_error_mm": max(item["spacing_error_mm"] for item in geometry_records), "raw_affine_max_abs_error": max(item["raw_affine_max_abs_error"] for item in geometry_records), "status": "PASS_GEOMETRY_EQUIVALENT"}, "no_transformation_optimization": True, "no_slice_scale": True}
    output = base / "final_pose_audit.json"
    write_immutable_json(output, {"schema": "fit_first_intensity_preserving_final_pose_audit/v1", "status": "POSE_STAGE1_AND_QUANTITATIVE_STAGE2_VALID", "route_manifest": _record(base / "manifest.json"), "quantitative_input_gate": _record(gate_path), "pose_authority": "Stage-1 final registered slices", "quantitative_value_authority": "standardized native dictionary maps", "parameters": records, "stage2_registration_assertion": "Stage-2 receives --input-slices, bypasses Reconstruct._register, and result.json proves no_transformation_optimization=true", "immutable": True})
    return output


def prepare_intensity_preserving_fit_first(*, subject_id: str, standardized_native_maps_root: str | Path, standardized_reuse_manifest: str | Path, sparse_manifest: str | Path | None, output: str | Path, prepared_root: str | Path | None = None, expected_generator_git_sha: str | None = None, expected_manifest_sha256: str | None = None, nesvor_executable: str | Path = HISTORICAL_NESVOR_EXECUTABLE_DEFAULT, device: str = "-1") -> tuple[Path, list[str]]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing overwrite: {destination}")
    if bool(expected_generator_git_sha) != bool(expected_manifest_sha256):
        raise ValueError("sparse pins must be supplied together")
    if sparse_manifest is not None and not expected_generator_git_sha:
        raise ValueError("intensity-preserving sparse Fit-first requires exact generator and manifest SHA256 pins")
    sparse = validate_sparse_profile_manifest(sparse_manifest, subject=subject_id, strict=True, expected_generator_git_sha=expected_generator_git_sha, expected_manifest_sha256=expected_manifest_sha256) if sparse_manifest else None
    validated_prepared_root = _validate_prepared_root(prepared_root, sparse)
    reuse = validate_standardized_native_dictionary_reuse(standardized_reuse_manifest, subject=subject_id); reuse["_manifest_path"] = str(Path(standardized_reuse_manifest).resolve())
    if Path(standardized_native_maps_root).resolve() != Path(standardized_reuse_manifest).resolve().parent:
        raise ValueError("standardized native maps root must be the validated reuse-manifest parent")
    destination.mkdir(parents=True)
    try:
        native, active, selected = _selected_standardized_maps(standardized_root=Path(standardized_native_maps_root), reuse=reuse, sparse=sparse, output=destination)
        for stack in active:
            export_nesvor_native_inputs(native / stack, destination / "nesvor_inputs" / stack, SCALES)
        commands = build_registration_commands(destination / "nesvor_inputs", active, destination / "stage1_registration", nesvor_executable=nesvor_executable, device=device)
        (destination / "stage1_registration_commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(commands) + "\n", encoding="utf-8")
        write_immutable_json(destination / "stage1_registration_manifest.json", {"schema": REGISTRATION_SCHEMA, "subject_id": subject_id, "active_stacks": active, "commands": _record(destination / "stage1_registration_commands.sh"), "registration": "stack", "t1_t2_independent": True, "value_role": "pose estimation only; q99-normalised registration pixels are never Stage-2 values", "immutable": True})
        sample_masks = {parameter: destination / "nesvor_inputs" / "sax" / parameter / "stack.nii.gz" for parameter in PARAMETERS}
        if "sax" not in active or any(not path.is_file() for path in sample_masks.values()):
            raise ValueError("intensity-preserving sparse route requires active SAX parameter-matched sample masks")
        write_immutable_json(destination / "manifest.json", {"schema": ROUTE_SCHEMA, "subject_id": subject_id, "prepared_root": None if validated_prepared_root is None else str(validated_prepared_root), "sparse_manifest": None if sparse is None else _record(Path(sparse["_manifest_path"])), "standardized_reuse_manifest": _record(Path(standardized_reuse_manifest)), "active_stacks": active, "selected_native_maps": selected, "stage1": _record(destination / "stage1_registration_manifest.json"), "stage2": {"schema": INPUT_SCHEMA, "purpose": "frozen Stage-1 affines plus authoritative dictionary values", "registration": "none; reconstruct --input-slices; --no-transformation-optimization", "raw_output_sampling_mask": {"policy": "explicit_parameter_matched_sax_native_support", "scope": "post_training_raw_volume_sampling_only", "training_affected": False, "registration_affected": False, "pose_optimization_affected": False, "formal_endpoint_affected": False, "sample_masks": {parameter: _record(path) for parameter, path in sample_masks.items()}}}, "no_dictionary_refit": True, "no_pose_copying": True, "code_git_sha": _git_sha(), "immutable": True})
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True); raise
    return destination, commands


def execute_intensity_preserving_fit_first(**kwargs: Any) -> Path:
    root, commands = prepare_intensity_preserving_fit_first(**kwargs)
    for parameter in PARAMETERS:
        (root / "stage1_registration" / parameter).mkdir(parents=True, exist_ok=False)
    for command in commands:
        subprocess.run(command, shell=True, check=True)
    route = json.loads((root / "manifest.json").read_text(encoding="utf-8")); active = route["active_stacks"]
    registered = rebuild_registered_quantitative_inputs(native_maps_root=root / "native_maps", registration_root=root / "stage1_registration", output=root / "registered_quantitative_inputs", active_stacks=active)
    gate = validate_registered_quantitative_inputs(native_maps_root=root / "native_maps", registration_root=root / "stage1_registration", inputs_root=registered, active_stacks=active)
    write_immutable_json(root / "registered_quantitative_inputs_gate.json", gate)
    commands = []
    for parameter in PARAMETERS:
        target = root / "stage2_reconstruction" / parameter; target.mkdir(parents=True)
        sample_mask = root / "nesvor_inputs" / "sax" / parameter / "stack.nii.gz"
        args = _reconstruct_args(registered / parameter / "slices", target, parameter, nesvor_executable=kwargs.get("nesvor_executable", HISTORICAL_NESVOR_EXECUTABLE_DEFAULT), device=kwargs.get("device", "-1"), sample_mask=sample_mask)
        commands.append(" ".join(shlex.quote(value) for value in args) + f" && {shlex.quote(sys.executable)} -m paper_experiments.joint_vs_2dfit.phase2_clean.fit_first_denormalize --input {shlex.quote(str(target / 'volume_normalized.nii.gz'))} --scale {SCALES[parameter]} --output {shlex.quote(str(target / 'volume_ms.nii.gz'))}")
    (root / "stage2_reconstruction_commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(commands) + "\n", encoding="utf-8")
    for command in commands:
        subprocess.run(command, shell=True, check=True)
    audit_intensity_preserving_final_pose(root=root)
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True); parser.add_argument("--prepared-root")  # identity is held by the pinned sparse/standardized inputs
    parser.add_argument("--standardized-native-maps-root", required=True); parser.add_argument("--standardized-reuse-manifest", required=True); parser.add_argument("--sparse-manifest")
    parser.add_argument("--expected-generator-git-sha"); parser.add_argument("--expected-manifest-sha256"); parser.add_argument("--output", required=True); parser.add_argument("--nesvor-executable", default=HISTORICAL_NESVOR_EXECUTABLE_DEFAULT); parser.add_argument("--device", default="-1"); parser.add_argument("--execute", action="store_true")
    args = vars(parser.parse_args())
    if not args.pop("execute"):
        root, commands = prepare_intensity_preserving_fit_first(**args)
        print(root); print("\n".join(commands)); return
    print(execute_intensity_preserving_fit_first(**args))


if __name__ == "__main__":
    main()
