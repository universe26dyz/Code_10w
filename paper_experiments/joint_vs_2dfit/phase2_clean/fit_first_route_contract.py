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


def resolve_fit_first_route(root: str | Path, *, legacy_audit_root: str | Path | None = None) -> FitFirstFormalRoute:
    """Resolve exactly one declared route; never infer from optional paths."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        schema = manifest.get("schema")
        if schema == INTENSITY_SCHEMA:
            return _intensity_preserving(root, manifest_path, manifest)
        raise ValueError(f"unknown Fit-first route manifest schema: {schema!r}")
    return _legacy(root, None if legacy_audit_root is None else Path(legacy_audit_root))
