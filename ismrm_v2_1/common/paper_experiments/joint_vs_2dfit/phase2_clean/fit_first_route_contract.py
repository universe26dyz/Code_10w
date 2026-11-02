"""Explicit formal-downstream contracts for legacy and corrected Fit-first roots."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import sha256

PARAMETERS = ("t1", "t2")
INTENSITY_SCHEMA = "phase4r_intensity_preserving_fit_first/v2"
INTENSITY_INPUT_SCHEMA = "phase4r_fit_first_registered_quantitative_inputs/v2"
INTENSITY_AUDIT_SCHEMA = "fit_first_intensity_preserving_final_pose_audit/v2"
SCMR_ONE_STAGE_SCHEMA = "phase4r_scmr_one_stage_fit_first/v1"
EXP1_1_SCHEMA = "phase4r_exp1_1_fit_first/v1"
SCMR_ONE_STAGE_FIXED_ARGS = {
    "n_iter": 10000,
    "n_samples": 256,
    "n_inference_samples": 128,
    "inference_batch_size": 16000,
    "pixel_variance_weight": 1,
    "weight_transformation": 1,
    "weight_deform": 0.1,
    "weight_image": 1,
    "weight_pixel_var_smooth": 0,
    "output_resolution": 1,
    "output_psf_factor": 1,
    "coarsest_resolution": 16,
    "finest_resolution": 0.5,
    "level_scale": 1.3819,
    "n_features_per_level": 2,
    "depth": 1,
    "width": 64,
    "n_features_z": 15,
    "n_features_slice": 16,
    "weight_bias": 100,
    "image_regularization": "edge",
    "delta": 0.2,
    "learning_rate": 0.005,
    "gamma": 0.33,
    "milestones": [0.5, 0.75, 0.9],
    "seed": 20260823,
}


def _record(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _args(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("args", payload)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: result arguments are invalid")
    return value


def _matches_record(value: Any, path: Path) -> bool:
    return isinstance(value, dict) and path.is_file() and Path(str(value.get("path", ""))).resolve() == path.resolve() and value.get("sha256") == sha256(path)


def _same_argument(value: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return isinstance(value, list) and len(value) == len(expected) and all(_same_argument(actual, item) for actual, item in zip(value, expected))
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) == float(expected)
    return value == expected


def _validate_scmr_one_stage_result(root: Path, parameter: str, active_stacks: list[str], args: dict[str, Any]) -> None:
    expected_inputs = [root / "nesvor_inputs" / stack / parameter / "stack.nii.gz" for stack in active_stacks]
    received_inputs = args.get("input_stacks")
    if not isinstance(received_inputs, list) or [Path(str(value)).resolve() for value in received_inputs] != [path.resolve() for path in expected_inputs]:
        raise ValueError(f"SCMR one-stage {parameter}: result input_stacks do not match active stack order")
    if args.get("registration") != "stack" or args.get("no_slice_scale") is not True or args.get("no_transformation_optimization") is True:
        raise ValueError(f"SCMR one-stage {parameter}: result contradicts one-stage registration/pose contract")
    if not _same_argument(args.get("thicknesses"), [8] * len(active_stacks)):
        raise ValueError(f"SCMR one-stage {parameter}: result thicknesses must be 8 mm for every active stack")
    for name, expected in SCMR_ONE_STAGE_FIXED_ARGS.items():
        if not _same_argument(args.get(name), expected):
            raise ValueError(f"SCMR one-stage {parameter}: result {name} does not match the frozen historical contract")


def _validate_stage1_stack_masks(root: Path, manifest: dict[str, Any], audit: dict[str, Any]) -> None:
    """Require the v2 route/audit to prove the masks passed to NeSVoR Stage 1."""
    stage1_path = root / "stage1_registration_manifest.json"
    if not _matches_record(manifest.get("stage1"), stage1_path):
        raise ValueError("Stage-1 registration manifest provenance mismatch")
    stage1 = json.loads(stage1_path.read_text(encoding="utf-8"))
    active = manifest.get("active_stacks")
    if stage1.get("schema") != "phase4r_fit_first_registration_stage/v2" or not isinstance(active, list) or stage1.get("active_stacks") != active:
        raise ValueError("Stage-1 active stack provenance mismatch")
    masks = stage1.get("stack_masks")
    audit_parameters = audit.get("parameters")
    if not isinstance(masks, dict) or not isinstance(audit_parameters, dict):
        raise ValueError("Stage-1 stack mask provenance is absent")
    for parameter in PARAMETERS:
        expected_stacks = [root / "nesvor_inputs" / stack / parameter / "stack.nii.gz" for stack in active]
        expected_masks = [root / "nesvor_inputs" / stack / parameter / "stack_mask.nii.gz" for stack in active]
        args = _args(root / "stage1_registration" / parameter / "result.json")
        got_stacks, got_masks = args.get("input_stacks"), args.get("stack_masks")
        if not isinstance(got_stacks, list) or not isinstance(got_masks, list) or len(got_stacks) != len(expected_stacks) or len(got_masks) != len(expected_masks):
            raise ValueError(f"{parameter}: Stage-1 stack mask count does not match input stacks")
        if [Path(str(value)).resolve() for value in got_stacks] != [path.resolve() for path in expected_stacks] or [Path(str(value)).resolve() for value in got_masks] != [path.resolve() for path in expected_masks]:
            raise ValueError(f"{parameter}: Stage-1 stack mask paths do not match native inputs")
        manifest_masks = masks.get(parameter)
        audit_masks = audit_parameters.get(parameter, {}).get("stage1_stack_masks") if isinstance(audit_parameters.get(parameter), dict) else None
        if not isinstance(manifest_masks, dict) or not isinstance(audit_masks, list) or len(audit_masks) != len(expected_masks):
            raise ValueError(f"{parameter}: Stage-1 stack mask SHA provenance is absent")
        for stack, mask, audit_mask in zip(active, expected_masks, audit_masks):
            if not _matches_record(manifest_masks.get(stack), mask) or not _matches_record(audit_mask, mask):
                raise ValueError(f"{parameter}: Stage-1 stack mask SHA provenance mismatch")


@dataclass(frozen=True)
class FitFirstFormalRoute:
    route_kind: str
    route_schema: str
    root: Path
    models: dict[str, Path]
    results: dict[str, Path]
    final_slices: dict[str, Path]
    pose_audit_path: Path | None
    pose_audit: dict[str, Any] | None
    pose_authority: str
    quantitative_value_authority: str

    def provenance(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "route_kind": self.route_kind,
            "route_schema": self.route_schema,
            "source_fit_first_root": str(self.root.resolve()),
            "formal_reconstruction_model": {parameter: _record(path) for parameter, path in self.models.items()},
            "parameter_results": {parameter: _record(path) for parameter, path in self.results.items()},
            "pose_authority": self.pose_authority,
            "quantitative_value_authority": self.quantitative_value_authority,
        }
        if self.pose_audit_path is not None:
            result["final_pose_audit"] = _record(self.pose_audit_path)
        return result


def _legacy(root: Path, legacy_audit_root: Path | None) -> FitFirstFormalRoute:
    manifest = root / "input_manifest.json"
    audit_path, audit = None, None
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("method") != "fit_first":
            raise ValueError("legacy Fit-first input manifest is invalid")
    elif legacy_audit_root is None:
        raise FileNotFoundError(f"legacy Fit-first root requires {manifest}")
    if legacy_audit_root is not None:
        audit_path = legacy_audit_root / "manifest.json"
        if not audit_path.is_file():
            raise FileNotFoundError(audit_path)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        status = audit.get("reconstruction_input_geometry_gate", {}).get("status")
        if audit.get("schema") != "fit_first_native_pose_audit/v3" or status != "RECONSTRUCTION_INPUT_GEOMETRY_VALID":
            raise ValueError(f"legacy final-pose audit must be fit_first_native_pose_audit/v3 and valid; status={status}")
    models, results, slices = {}, {}, {}
    for parameter in PARAMETERS:
        target = root / "nesvor" / parameter
        model, result, final = target / "model.pt", target / "result.json", target / "slices"
        if not model.is_file() or not result.is_file() or not final.is_dir():
            raise FileNotFoundError(f"legacy {parameter}: model.pt/result.json/slices required")
        models[parameter], results[parameter], slices[parameter] = model, result, final
    return FitFirstFormalRoute("legacy_fit_first", "legacy_fit_first/input_manifest", root, models, results, slices, audit_path, audit, "legacy fit_first_native_pose_audit/v3 required by native-bundle admission", "legacy native dictionary maps")


def _intensity_preserving(root: Path, manifest_path: Path, manifest: dict[str, Any]) -> FitFirstFormalRoute:
    if manifest.get("no_dictionary_refit") is not True or manifest.get("no_pose_copying") is not True:
        raise ValueError("intensity-preserving route lacks no-refit/no-pose-copying evidence")
    gate_path, audit_path = root / "registered_quantitative_inputs_gate.json", root / "final_pose_audit.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("schema") != INTENSITY_INPUT_SCHEMA or gate.get("status") != "REGISTERED_QUANTITATIVE_INPUTS_VALID":
        raise ValueError("registered quantitative input gate is not valid")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema") != INTENSITY_AUDIT_SCHEMA or audit.get("status") != "POSE_STAGE1_AND_QUANTITATIVE_STAGE2_VALID":
        raise ValueError("final pose audit is not valid for intensity-preserving Fit-first")
    _validate_stage1_stack_masks(root, manifest, audit)
    models, results, slices = {}, {}, {}
    for parameter in PARAMETERS:
        target = root / "stage2_reconstruction" / parameter
        model, result, final = target / "model.pt", target / "result.json", target / "slices"
        if not model.is_file() or not result.is_file() or not final.is_dir():
            raise FileNotFoundError(f"intensity-preserving {parameter}: Stage-2 model.pt/result.json/slices required")
        args = _args(result)
        expected_input = str((root / "registered_quantitative_inputs" / parameter / "slices").resolve())
        if str(args.get("input_slices", "")) != expected_input:
            raise ValueError(f"{parameter}: Stage-2 result does not prove gated registered quantitative input_slices")
        if args.get("input_stacks") is not None or args.get("no_transformation_optimization") is not True:
            raise ValueError(f"{parameter}: Stage-2 result does not prove no_transformation_optimization=true with input_slices")
        if args.get("no_slice_scale") is not True:
            raise ValueError(f"{parameter}: Stage-2 result does not prove no_slice_scale=true")
        models[parameter], results[parameter], slices[parameter] = model, result, final
    return FitFirstFormalRoute("intensity_preserving_fit_first", INTENSITY_SCHEMA, root, models, results, slices, audit_path, audit, "validated Stage-1 -> Stage-2 frozen-pose chain", "standardized native dictionary maps; Stage-2 intensity-preserving checkpoint")


def _scmr_one_stage(root: Path, manifest: dict[str, Any]) -> FitFirstFormalRoute:
    if manifest.get("no_dictionary_refit") is not True or manifest.get("independent_t1_t2") is not True or manifest.get("transformation_optimization_enabled") is not True or manifest.get("no_slice_scale") is not True:
        raise ValueError("SCMR one-stage route lacks independent transformation-optimization provenance")
    active_stacks = manifest.get("active_stacks")
    if manifest.get("nesvor_version") != "nesvor_v5_128_w1" or not isinstance(active_stacks, list) or not active_stacks or any(stack not in ("sax", "2ch", "4ch") for stack in active_stacks) or len(set(active_stacks)) != len(active_stacks):
        raise ValueError("SCMR one-stage route lacks historical NeSVoR provenance")
    command = manifest.get("commands")
    selected = manifest.get("selected_standardized_native_maps")
    selected_path = Path(str(selected.get("path", ""))) if isinstance(selected, dict) else None
    if not _matches_record(command, root / "commands.sh") or selected_path is None or not _matches_record(selected, selected_path):
        raise ValueError("SCMR one-stage command/selected-map provenance mismatch")
    command_text = (root / "commands.sh").read_text(encoding="utf-8")
    if "--registration stack" not in command_text or "--no-transformation-optimization" in command_text:
        raise ValueError("SCMR one-stage command does not preserve transformation optimization")
    models, results, slices = {}, {}, {}
    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("SCMR one-stage parameter provenance is absent")
    for parameter in PARAMETERS:
        target = root / "nesvor" / parameter
        model, result, final = target / "model.pt", target / "result.json", target / "slices"
        record = parameters.get(parameter)
        if not model.is_file() or not result.is_file() or not final.is_dir() or not any(final.glob("[0-9]*.nii.gz")):
            raise FileNotFoundError(f"SCMR one-stage {parameter}: model.pt/result.json/final slices required")
        if not isinstance(record, dict) or not _matches_record(record.get("model"), model) or not _matches_record(record.get("result"), result) or Path(str(record.get("final_slices", ""))).resolve() != final.resolve():
            raise ValueError(f"SCMR one-stage {parameter}: parameter provenance mismatch")
        args = _args(result)
        _validate_scmr_one_stage_result(root, parameter, active_stacks, args)
        models[parameter], results[parameter], slices[parameter] = model, result, final
    return FitFirstFormalRoute("scmr_one_stage_fit_first", SCMR_ONE_STAGE_SCHEMA, root, models, results, slices, None, None, "independent parameter-specific one-stage NeSVoR stack registration and transformation optimization", "standardized native dictionary maps; one-stage NeSVoR checkpoint")


def _exp1_1(root: Path, manifest: dict[str, Any]) -> FitFirstFormalRoute:
    """Validate the Final5 frozen-map CYJ Experiment 1.1 route."""
    if manifest.get("scientific_route") != "CYJ Experiment 1.1 mapping-to-one-stage-NeSVoR" or manifest.get("dictionary_refit") is not False:
        raise ValueError("Experiment 1.1 route lacks frozen-map/no-refit provenance")
    if manifest.get("no_dictionary_refit") is not True or manifest.get("independent_t1_t2") is not True or manifest.get("transformation_optimization_enabled") is not True or manifest.get("no_slice_scale") is not True:
        raise ValueError("Experiment 1.1 route lacks independent one-stage transformation provenance")
    active_stacks = manifest.get("active_stacks")
    if manifest.get("nesvor_version") != "nesvor_v5_128_w1" or not isinstance(active_stacks, list) or not active_stacks or any(stack not in ("sax", "2ch", "4ch") for stack in active_stacks) or len(set(active_stacks)) != len(active_stacks):
        raise ValueError("Experiment 1.1 route lacks historical NeSVoR provenance")
    command, selected = manifest.get("commands"), manifest.get("selected_native_maps")
    selected_path = Path(str(selected.get("path", ""))) if isinstance(selected, dict) else None
    if not _matches_record(command, root / "commands.sh") or selected_path is None or not _matches_record(selected, selected_path):
        raise ValueError("Experiment 1.1 command/selected-map provenance mismatch")
    command_text = (root / "commands.sh").read_text(encoding="utf-8")
    if "--registration stack" not in command_text or "--no-transformation-optimization" in command_text:
        raise ValueError("Experiment 1.1 command contradicts historical one-stage semantics")
    output_sampling = manifest.get("output_sampling")
    expected_scope = "NeSVoR output INR sampling only; not a reconstruction/training input mask"
    if len(active_stacks) == 1:
        if not isinstance(output_sampling, dict) or output_sampling.get("scope") != expected_scope:
            raise ValueError("Experiment 1.1 single-stack output-sampling provenance is absent")
        masks = output_sampling.get("sample_masks")
        if output_sampling.get("policy") != "explicit_single_stack_native_finite_support" or not isinstance(masks, dict):
            raise ValueError("Experiment 1.1 single-stack output-sampling provenance is invalid")
        for parameter in PARAMETERS:
            mask = root / "nesvor_inputs" / active_stacks[0] / parameter / "stack_mask.nii.gz"
            if not _matches_record(masks.get(parameter), mask) or f"--sample-mask {mask}" not in command_text:
                raise ValueError(f"Experiment 1.1 {parameter} single-stack output-sampling mask mismatch")
    elif "--sample-mask" in command_text or (output_sampling is not None and (not isinstance(output_sampling, dict) or output_sampling.get("policy") != "nesvor_default_estimated_mask" or output_sampling.get("scope") != expected_scope)):
        raise ValueError("Experiment 1.1 multi-stack output-sampling provenance is invalid")
    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Experiment 1.1 parameter provenance is absent")
    models, results, slices = {}, {}, {}
    for parameter in PARAMETERS:
        target = root / "nesvor" / parameter
        model, result, final = target / "model.pt", target / "result.json", target / "slices"
        record = parameters.get(parameter)
        if not model.is_file() or not result.is_file() or not final.is_dir() or not any(final.glob("[0-9]*.nii.gz")):
            raise FileNotFoundError(f"Experiment 1.1 {parameter}: model.pt/result.json/final slices required")
        if not isinstance(record, dict) or not _matches_record(record.get("model"), model) or not _matches_record(record.get("result"), result) or Path(str(record.get("final_slices", ""))).resolve() != final.resolve():
            raise ValueError(f"Experiment 1.1 {parameter}: parameter provenance mismatch")
        _validate_scmr_one_stage_result(root, parameter, active_stacks, _args(result))
        models[parameter], results[parameter], slices[parameter] = model, result, final
    return FitFirstFormalRoute("exp1_1_fit_first", EXP1_1_SCHEMA, root, models, results, slices, None, None, "independent parameter-specific CYJ Experiment 1.1 one-stage NeSVoR registration and transformation optimization", "frozen standardized native T1/T2 maps; no dictionary refit")


def resolve_fit_first_route(root: str | Path, *, legacy_audit_root: str | Path | None = None) -> FitFirstFormalRoute:
    """Resolve exactly one declared route; never infer from optional paths."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        schema = manifest.get("schema")
        if schema == INTENSITY_SCHEMA:
            return _intensity_preserving(root, manifest_path, manifest)
        if schema == SCMR_ONE_STAGE_SCHEMA:
            return _scmr_one_stage(root, manifest)
        if schema == EXP1_1_SCHEMA:
            return _exp1_1(root, manifest)
        raise ValueError(f"unknown Fit-first route manifest schema: {schema!r}")
    return _legacy(root, None if legacy_audit_root is None else Path(legacy_audit_root))
