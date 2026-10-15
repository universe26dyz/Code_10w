"""Prepare Code_10w cardiac-ROI XCAT handoff files; never execute XCAT locally."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
PHANTOM_SPEC = ROOT / "cardiac_roi_phantom_spec.json"
GEOMETRY_TEMPLATE = ROOT / "cardiac_roi_geometry.template.json"
_KINDS = {
    "t1_phase1_2.par": "t1_ms",
    "t2_phase1_2.par": "t2_ms",
    "pd_phase1_2.par": "pd",
    "lv_mask_phase1_2.par": "lv_mask",
}
_GEOMETRY_FIELDS = (
    "mode", "motion_option", "resp_period", "hrt_period", "hrt_motion_x",
    "hrt_motion_y", "hrt_motion_z", "organ_file", "heart_base",
    "heart_curve_file", "vessel_flag", "coronary_art_flag",
    "coronary_vein_flag", "phan_rotx", "phan_roty", "phan_rotz",
    "pixel_width", "slice_width", "array_size", "startslice", "endslice",
    "activity_unit", "subvoxel_index",
)
_GEOMETRY_TOP_LEVEL = (
    "minimum_lv_boundary_margin_mm", "minimum_lv_voxel_count",
    "pca_lambda1_lambda2_min", "xcat_length_unit", "xcat_mm_per_length_unit",
    "raw_layout_policy", "origin_policy", "canonical_coordinate_policy",
    "xcat_version_build_compatibility",
    "formal_target_voxel_size_mm",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _format(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _placeholder(value: object) -> bool:
    return not isinstance(value, (int, float)) and (not isinstance(value, str) or not value.strip() or any(token in value.upper() for token in ("REQUIRED", "TODO", "UNKNOWN", "SET_")))


def validate_phantom_spec(phantom: dict[str, Any]) -> None:
    """Validate the frozen tissue table before it can generate XCAT assignments."""
    if phantom.get("frozen_lv") != {"t1_ms": 1410.0, "t2_ms": 50.0, "b1": 1.0}:
        raise ValueError("cardiac ROI phantom spec must retain frozen LV T1/T2/B1 values")
    if phantom.get("m0_policy") != "controlled_binary_signal_support_m0_1" or phantom.get("support_policy") != "native_xcat_pd_positive":
        raise ValueError("cardiac ROI support/M0 policy is not formal Code_10w policy")
    structures = phantom.get("structures")
    if not isinstance(structures, dict) or not structures:
        raise ValueError("cardiac ROI phantom spec has no structures")
    names, lv_positive = set(), []
    for name, structure in structures.items():
        for key in ("xcat_parameter", "t1_ms", "t2_ms", "pd", "signal_support", "lv_mask", "provenance"):
            if key not in structure or structure[key] in (None, ""):
                raise ValueError(f"cardiac ROI structure {name} is missing {key}")
        parameter = structure["xcat_parameter"]
        if parameter in names:
            raise ValueError("XCAT parameter names must be unique")
        names.add(parameter)
        if structure["lv_mask"] not in (0, 1, False, True):
            raise ValueError("lv_mask must be binary")
        if structure["lv_mask"]:
            lv_positive.append((name, structure))
        for key in ("t1_ms", "t2_ms", "pd"):
            if not np.isfinite(float(structure[key])) or float(structure[key]) < 0:
                raise ValueError(f"{name} {key} must be finite and non-negative")
        support = bool(structure["signal_support"])
        if support != (float(structure["pd"]) > 0):
            raise ValueError("signal_support must equal (PD > 0) under native_xcat_pd_positive")
        if support and (float(structure["t1_ms"]) <= 0 or float(structure["t2_ms"]) <= 0):
            raise ValueError("support tissue requires positive T1 and T2")
        if not support and float(structure["pd"]) != 0:
            raise ValueError("non-support tissue must have PD=0")
    if len(lv_positive) != 1 or lv_positive[0][0] != "lv_myocardium" or "LV myocardium" not in str(lv_positive[0][1].get("meaning", "")):
        raise ValueError("exactly one LV myocardium structure must have lv_mask=1")
    lv = lv_positive[0][1]
    if float(lv["t1_ms"]) != 1410.0 or float(lv["t2_ms"]) != 50.0:
        raise ValueError("LV myocardium must retain T1=1410 and T2=50")


def validate_geometry_spec(geometry: dict[str, Any], *, formal: bool) -> None:
    missing = [key for key in _GEOMETRY_TOP_LEVEL if key not in geometry]
    block = geometry.get("xcat_geometry_block")
    if missing or not isinstance(block, dict):
        raise ValueError("cardiac ROI geometry spec is incomplete")
    missing = [key for key in _GEOMETRY_FIELDS if key not in block]
    if missing:
        raise ValueError(f"cardiac ROI geometry is missing assignments: {missing}")
    if geometry["xcat_length_unit"] != "cm" or float(geometry["xcat_mm_per_length_unit"]) != 10.0:
        raise ValueError("Hanson-compatible XCAT geometry requires explicit cm-to-mm conversion (x10)")
    if geometry["raw_layout_policy"] != "hanson_4DPhantomMRI_97558db_zyx_slices_rows_cols":
        raise ValueError("unsupported raw XCAT layout policy requires a version-specific implementation")
    if geometry["origin_policy"] != "centered_synthetic_lps" or geometry["canonical_coordinate_policy"] != "xyz_synthetic_lps_mm":
        raise ValueError("formal XCAT route requires centered synthetic LPS xyz/mm coordinates")
    if float(geometry["minimum_lv_boundary_margin_mm"]) <= 0 or int(geometry["minimum_lv_voxel_count"]) <= 0 or float(geometry["pca_lambda1_lambda2_min"]) <= 1:
        raise ValueError("geometry acceptance thresholds are invalid")
    if tuple(float(value) for value in geometry["formal_target_voxel_size_mm"]) != (1.0, 1.0, 1.0):
        raise ValueError("formal Phase-1.2 target voxel size must be exactly 1x1x1 mm")
    subvoxel = block["subvoxel_index"]
    if isinstance(subvoxel, bool) or not isinstance(subvoxel, int) or subvoxel <= 0:
        raise ValueError("subvoxel_index must be a positive integer")
    if formal and (_placeholder(geometry["xcat_version_build_compatibility"]) or any(_placeholder(block[key]) for key in _GEOMETRY_FIELDS)):
        raise ValueError("formal cardiac geometry spec contains placeholders")
    if formal:
        contract = derive_raw_contract(geometry)
        if contract["canonical_spacing_mm_xyz"] != (1.0, 1.0, 1.0):
            raise ValueError("formal Phase-1.2 requires direct 1-mm XCAT spacing; resampling is not permitted")


def derive_raw_contract(geometry: dict[str, Any]) -> dict[str, Any]:
    """Derive Hanson raw binary shape/spacing; no user-entered raw geometry."""
    block = geometry["xcat_geometry_block"]
    start, end, size = int(block["startslice"]), int(block["endslice"]), int(block["array_size"])
    if start > end or size <= 1:
        raise ValueError("invalid frozen XCAT startslice/endslice/array_size")
    mm = float(geometry["xcat_mm_per_length_unit"])
    raw_shape = (end - start + 1, size, size)
    raw_spacing = (float(block["slice_width"]) * mm, float(block["pixel_width"]) * mm, float(block["pixel_width"]) * mm)
    raw_axis_order = "zyx"
    reorder = tuple(raw_axis_order.index(axis) for axis in "xyz")
    canonical_shape = tuple(raw_shape[index] for index in reorder)
    canonical_spacing = tuple(raw_spacing[index] for index in reorder)
    origin = tuple(-0.5 * (dimension - 1) * spacing for dimension, spacing in zip(canonical_shape, canonical_spacing))
    return {"raw_shape": raw_shape, "raw_spacing_mm": raw_spacing, "raw_axis_order": raw_axis_order, "canonical_shape_xyz": canonical_shape, "canonical_spacing_mm_xyz": canonical_spacing, "world_convention": "LPS", "origin_policy": geometry["origin_policy"], "origin_mm_xyz": origin}


def _load_specs(phantom_spec: Path = PHANTOM_SPEC, geometry_spec: Path = GEOMETRY_TEMPLATE) -> tuple[dict[str, Any], dict[str, Any]]:
    phantom, geometry = json.loads(phantom_spec.read_text()), json.loads(geometry_spec.read_text())
    validate_phantom_spec(phantom)
    validate_geometry_spec(geometry, formal=geometry_spec != GEOMETRY_TEMPLATE)
    return phantom, geometry


def _parameter(phantom: dict[str, Any], geometry: dict[str, Any], kind: str) -> str:
    lines = [f"{key} = {_format(value)}" for key, value in geometry["xcat_geometry_block"].items()]
    for structure in phantom["structures"].values():
        value = structure["lv_mask"] if kind == "lv_mask" else structure[kind]
        lines.append(f"{structure['xcat_parameter']} = {_format(value)}")
    lines.extend((
        "# REFERENCE MECHANISM: Hanson XCAT activity-field generation (not Hanson phantom content).",
        "# CODE_10W CONTENT: cardiac ROI tissue specification and shared geometry block.",
    ))
    return "\n".join(lines) + "\n"


def parse_parameter_assignments(path: str | Path) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key in assignments:
            raise ValueError(f"duplicate XCAT parameter assignment: {key}")
        if not value or any(token in value.upper() for token in ("REQUIRED", "TODO", "UNKNOWN", "SET_")):
            raise ValueError(f"placeholder XCAT parameter assignment: {key}")
        assignments[key] = value
    return assignments


def validate_parameter_contract(parameter_dir: str | Path, phantom: dict[str, Any], geometry: dict[str, Any]) -> dict:
    expected_geometry = {key: _format(value) for key, value in geometry["xcat_geometry_block"].items()}
    result = {"parameter_contract_gate": True, "files": {}}
    for filename, kind in _KINDS.items():
        actual = parse_parameter_assignments(Path(parameter_dir) / filename)
        expected = dict(expected_geometry)
        expected.update({item["xcat_parameter"]: _format(item["lv_mask"] if kind == "lv_mask" else item[kind]) for item in phantom["structures"].values()})
        if actual != expected:
            raise ValueError(f"XCAT parameter semantic contract mismatch: {filename}")
        result["files"][filename] = {"sha256": _sha256(Path(parameter_dir) / filename), "assignment_count": len(actual)}
    return result


def prepare_xcat_parameter_hand_off(output_root: str | Path, *, geometry_spec: str | Path | None = None, phantom_spec: str | Path | None = None) -> dict:
    """Write four same-geometry parameter files from the centralized cardiac specs."""
    root = Path(output_root) / "phase1_2_xcat_handoff"
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    geometry_path = Path(geometry_spec) if geometry_spec else GEOMETRY_TEMPLATE
    phantom_path = Path(phantom_spec) if phantom_spec else PHANTOM_SPEC
    phantom, geometry = _load_specs(phantom_path, geometry_path)
    formal = geometry_path != GEOMETRY_TEMPLATE
    params, reference = root / "parameter_templates", root / "reference_design"
    params.mkdir(parents=True)
    reference.mkdir()
    (root / PHANTOM_SPEC.name).write_bytes(phantom_path.read_bytes())
    (root / GEOMETRY_TEMPLATE.name).write_bytes(geometry_path.read_bytes())
    for filename, kind in _KINDS.items():
        (params / filename).write_text(_parameter(phantom, geometry, kind))
    if formal:
        contract_qc = validate_parameter_contract(params, phantom, geometry)
        (root / "parameter_contract_qc.json").write_text(json.dumps(contract_qc, indent=2, sort_keys=True) + "\n")
    mechanism = {
            "repository": "https://github.com/hmhanson/4DPhantomMRI",
            "commit": "97558db05f4e344865f472de974b9f4d042a1a15",
            "files": {"readme": "README.md", "t1": "XCAT/t1.par", "t2": "XCAT/t2.par", "pd": "XCAT/proton_density.par", "ap": "XCAT/XCAT_input_ap.dat", "dia": "XCAT/XCAT_input_dia.dat", "loader_example": "2D Sample.ipynb"},
            "reused": "multiple full-3D XCAT activity-field runs with common anatomy, phase, rasterizer and geometry",
    }
    provenance = {
        "reference": mechanism,
        "reference_generation_mechanism": mechanism,
        "code_10w_cardiac_phantom_content": "cardiac ROI tissue table, 1410/50/1 LV GT, controlled binary M0, LV mask, 1-mm canonical GT, LV-PCA views, HHZ and 6x6x9 rendering; not a Hanson phantom reproduction",
        "cardiac_roi_phantom_spec_sha256": _sha256(phantom_path),
        "cardiac_roi_geometry_spec_sha256": _sha256(geometry_path),
    }
    (reference / "reference_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    (root / "README_XCAT_EXECUTION.md").write_text(
        "Hanson supplies the quantitative-XCAT generation mechanism; Code_10w supplies the cardiac-ROI content. "
        + ("This is an editable template only; it is not executable until geometry is frozen. " if not formal else "This is a formal frozen handoff; run the four generated files with identical geometry. ")
        + "Record raw, parameter-file, phantom-spec and geometry-spec SHA256 values in xcat3d_inputs.json.\n"
    )
    (root / "xcat3d_inputs.template.json").write_text(json.dumps({
        "phase": "1.2", "source_family": "single_source_xcat_3d_quantitative",
        "xcat": {"version_build": "REQUIRED", "source": "REQUIRED", "cardiac_phase": "REQUIRED", "respiratory_phase": "REQUIRED", "phantom_spec": {"path": PHANTOM_SPEC.name, "sha256": _sha256(phantom_path)}, "geometry_spec": {"path": GEOMETRY_TEMPLATE.name if formal else "FREEZE_PLACEHOLDER_FREE_GEOMETRY.json", "sha256": _sha256(geometry_path) if formal else "REQUIRED"}, "parameter_files": {name: {"path": f"parameter_templates/{name}", "sha256": _sha256(params / name)} for name in _KINDS}, "static_inputs": "RECORD_PATH_SHA256_ROLE_FOR_EACH_REFERENCED_XCAT_STATIC_INPUT"},
        "support_policy": "native_xcat_pd_positive", "m0_policy": "controlled_binary_signal_support_m0_1",
        "volumes": "RECORD_T1_T2_PD_LV_MASK_RAW_FILES_AND_SHA256",
    }, indent=2, sort_keys=True) + "\n")
    return {"handoff_root": str(root), "reference_generation_mechanism": "PASS", "cardiac_phantom_specification": "PASS", "external_xcat_runtime_gate": "UNVERIFIED"}


def freeze_xcat_parameter_hand_off(output_root: str | Path, *, geometry_spec: str | Path, phantom_spec: str | Path | None = None) -> dict:
    """Public formal API: selected placeholder-free geometry is mandatory."""
    return prepare_xcat_parameter_hand_off(output_root, geometry_spec=geometry_spec, phantom_spec=phantom_spec)
