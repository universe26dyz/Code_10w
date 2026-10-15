"""Prepare Code_10w cardiac-ROI XCAT handoff files; never execute XCAT locally."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PHANTOM_SPEC = ROOT / "cardiac_roi_phantom_spec.json"
GEOMETRY_TEMPLATE = ROOT / "cardiac_roi_geometry.template.json"
_KINDS = {
    "t1_phase1_2.par": "t1_ms",
    "t2_phase1_2.par": "t2_ms",
    "pd_phase1_2.par": "pd",
    "lv_mask_phase1_2.par": "lv_mask",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _format(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _load_specs(phantom_spec: Path = PHANTOM_SPEC, geometry_spec: Path = GEOMETRY_TEMPLATE) -> tuple[dict[str, Any], dict[str, Any]]:
    phantom, geometry = json.loads(phantom_spec.read_text()), json.loads(geometry_spec.read_text())
    if phantom["frozen_lv"] != {"t1_ms": 1410.0, "t2_ms": 50.0, "b1": 1.0}:
        raise ValueError("cardiac ROI phantom spec must retain frozen LV T1/T2/B1 values")
    if phantom["m0_policy"] != "controlled_binary_signal_support_m0_1" or phantom["support_policy"] != "native_xcat_pd_positive":
        raise ValueError("cardiac ROI support/M0 policy is not formal Code_10w policy")
    if not phantom.get("structures") or not geometry.get("xcat_geometry_block"):
        raise ValueError("cardiac ROI specs are incomplete")
    for structure in phantom["structures"].values():
        for key in ("xcat_parameter", "t1_ms", "t2_ms", "pd", "signal_support", "lv_mask", "provenance"):
            if key not in structure or structure[key] in (None, ""):
                raise ValueError(f"cardiac ROI structure is missing {key}")
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


def prepare_xcat_parameter_hand_off(output_root: str | Path, *, geometry_spec: str | Path | None = None, phantom_spec: str | Path | None = None) -> dict:
    """Write four same-geometry parameter files from the centralized cardiac specs."""
    root = Path(output_root) / "phase1_2_xcat_handoff"
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    geometry_path = Path(geometry_spec) if geometry_spec else GEOMETRY_TEMPLATE
    phantom_path = Path(phantom_spec) if phantom_spec else PHANTOM_SPEC
    phantom, geometry = _load_specs(phantom_path, geometry_path)
    formal = geometry_path != GEOMETRY_TEMPLATE
    if formal and any("REQUIRED" in str(value).upper() or "TODO" in str(value).upper() for value in geometry["xcat_geometry_block"].values()):
        raise ValueError("formal cardiac geometry spec contains placeholders")
    params, reference = root / "parameter_templates", root / "reference_design"
    params.mkdir(parents=True)
    reference.mkdir()
    (root / PHANTOM_SPEC.name).write_bytes(phantom_path.read_bytes())
    (root / GEOMETRY_TEMPLATE.name).write_bytes(geometry_path.read_bytes())
    for filename, kind in _KINDS.items():
        (params / filename).write_text(_parameter(phantom, geometry, kind))
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
        "cardiac_roi_geometry_template_sha256": _sha256(geometry_path),
    }
    (reference / "reference_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    (root / "README_XCAT_EXECUTION.md").write_text(
        "Hanson supplies the quantitative-XCAT generation mechanism; Code_10w supplies the cardiac-ROI content. "
        "Freeze a placeholder-free geometry copy in the licensed XCAT environment, then run the four generated files with identical geometry. "
        "Record raw, parameter-file, phantom-spec and geometry-spec SHA256 values in xcat3d_inputs.json.\n"
    )
    (root / "xcat3d_inputs.template.json").write_text(json.dumps({
        "phase": "1.2", "source_family": "single_source_xcat_3d_quantitative",
        "xcat": {"version_build": "REQUIRED", "source": "REQUIRED", "cardiac_phase": "REQUIRED", "respiratory_phase": "REQUIRED", "phantom_spec": {"path": PHANTOM_SPEC.name, "sha256": _sha256(phantom_path)}, "geometry_spec": {"path": GEOMETRY_TEMPLATE.name if formal else "FREEZE_PLACEHOLDER_FREE_GEOMETRY.json", "sha256": _sha256(geometry_path) if formal else "REQUIRED"}, "parameter_files": {name: {"path": f"parameter_templates/{name}", "sha256": _sha256(params / name)} for name in _KINDS}},
        "support_policy": "native_xcat_pd_positive", "m0_policy": "controlled_binary_signal_support_m0_1",
        "volumes": "RECORD_T1_T2_PD_LV_MASK_RAW_FILES_AND_SHA256",
    }, indent=2, sort_keys=True) + "\n")
    return {"handoff_root": str(root), "reference_generation_mechanism": "PASS", "cardiac_phantom_specification": "PASS", "external_xcat_runtime_gate": "UNVERIFIED"}
