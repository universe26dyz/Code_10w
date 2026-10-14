"""Prepare a non-executable XCAT hand-off; this module never runs XCAT itself."""
from __future__ import annotations

import json
from pathlib import Path


def prepare_xcat_parameter_hand_off(output_root: str | Path) -> dict:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    template = root / "xcat3d_inputs.template.json"
    if template.exists():
        raise FileExistsError(f"refusing to overwrite {template}")
    payload = {
        "phase": "1.2",
        "source_family": "single_source_xcat_3d_quantitative",
        "xcat": {
            "source": "SET_TO_LICENSED_XCAT_RUNTIME_AND_VERSION",
            "parameter_files": "SET_TO_SHARED_PARAMETER_FILE_DIRECTORY",
            "configuration_hashes": "SET_TO_HASHES_OF_EACH_SHARED_XCAT_PARAMETER_FILE",
            "cardiac_phase": "SET_TO_ONE_SHARED_CARDIAC_PHASE",
            "respiratory_phase": "SET_TO_ONE_SHARED_RESPIRATORY_PHASE",
        },
        "m0_policy": "controlled_binary_signal_support_m0_1",
        "volumes": {
            "t1_ms": {"path": "t1_ms.bin", "dtype": "float32", "shape": ["SET_X", "SET_Y", "SET_Z"], "spacing_mm": ["SET", "SET", "SET"], "origin_mm": ["SET", "SET", "SET"], "axis_order": "xyz", "world_convention": "LPS"},
            "t2_ms": {"path": "t2_ms.bin", "dtype": "float32", "shape": ["SET_X", "SET_Y", "SET_Z"], "spacing_mm": ["SET", "SET", "SET"], "origin_mm": ["SET", "SET", "SET"], "axis_order": "xyz", "world_convention": "LPS"},
            "signal_support_mask": {"path": "signal_support_mask.bin", "dtype": "bool", "shape": ["SET_X", "SET_Y", "SET_Z"], "spacing_mm": ["SET", "SET", "SET"], "origin_mm": ["SET", "SET", "SET"], "axis_order": "xyz", "world_convention": "LPS"},
            "myocardium_mask": {"path": "myocardium_mask.bin", "dtype": "bool", "shape": ["SET_X", "SET_Y", "SET_Z"], "spacing_mm": ["SET", "SET", "SET"], "origin_mm": ["SET", "SET", "SET"], "axis_order": "xyz", "world_convention": "LPS"},
        },
    }
    template.write_text(json.dumps(payload, indent=2) + "\n")
    return {"template": str(template), "external_xcat_input_gate": "UNVERIFIED"}
