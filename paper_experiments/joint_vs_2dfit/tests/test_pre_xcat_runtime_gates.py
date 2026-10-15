from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.simulation.render_observations import coverage_preflight
from paper_experiments.joint_vs_2dfit.xcat3d.audits import quantitative_tuple_audit
from paper_experiments.joint_vs_2dfit.xcat3d.prepare_xcat_parameters import (
    derive_raw_contract,
    prepare_xcat_parameter_hand_off,
    validate_parameter_contract,
    validate_phantom_spec,
)


def _frozen_geometry(tmp_path: Path) -> Path:
    source = Path(__file__).parents[1] / "xcat3d/cardiac_roi_geometry.template.json"
    geometry = json.loads(source.read_text())
    geometry["xcat_version_build_compatibility"] = "fixture-xcat"
    geometry["xcat_geometry_block"].update({"array_size": 8, "startslice": 1, "endslice": 8})
    path = tmp_path / "geometry.json"; path.write_text(json.dumps(geometry))
    return path


def test_parameter_contract_rejects_tampered_assignment(tmp_path: Path) -> None:
    handoff = Path(prepare_xcat_parameter_hand_off(tmp_path, geometry_spec=_frozen_geometry(tmp_path))["handoff_root"])
    phantom = json.loads((handoff / "cardiac_roi_phantom_spec.json").read_text())
    geometry = json.loads((handoff / "cardiac_roi_geometry.template.json").read_text())
    path = handoff / "parameter_templates/t1_phase1_2.par"
    path.write_text(path.read_text().replace("myoLV_act = 1410", "myoLV_act = 1409"))
    with pytest.raises(ValueError, match="semantic contract"):
        validate_parameter_contract(path.parent, phantom, geometry)


def test_quantitative_tuple_audit_rejects_undeclared_and_wrong_lv_tuple(tmp_path: Path) -> None:
    spec = Path(__file__).parents[1] / "xcat3d/cardiac_roi_phantom_spec.json"
    payload = json.loads(spec.read_text())
    t1 = np.array([1410.0, 999.0], np.float32); t2 = np.array([50.0, 1.0], np.float32); pd = np.array([1346.0, 1.0], np.float32)
    with pytest.raises(ValueError, match="tuple audit"):
        quantitative_tuple_audit(t1, t2, pd, np.array([True, False]), spec)
    with pytest.raises(ValueError, match="tuple audit"):
        quantitative_tuple_audit(np.array([999.0]), np.array([1.0]), np.array([1.0]), np.array([True]), spec)
    assert payload["structures"]["lv_myocardium"]["t1_ms"] == 1410.0


def test_coverage_preflight_rejects_through_plane_and_inplane_escape() -> None:
    good = {"group_idx": 0, "image_shape_rows_cols": [1, 1], "affine_lps_rc": np.eye(4).tolist()}
    assert coverage_preflight((3, 3, 3, 10), -np.ones(3), np.ones(3), [good])["gate"] is True
    bad = dict(good); bad["affine_lps_rc"] = np.eye(4).tolist(); bad["affine_lps_rc"][2][3] = 3.0
    with pytest.raises(ValueError, match="out-of-volume"):
        coverage_preflight((3, 3, 3, 10), -np.ones(3), np.ones(3), [bad])


def test_phantom_spec_and_frozen_geometry_contract_are_machine_checked(tmp_path: Path) -> None:
    spec_path = Path(__file__).parents[1] / "xcat3d/cardiac_roi_phantom_spec.json"
    spec = json.loads(spec_path.read_text())
    validate_phantom_spec(spec)
    duplicate = json.loads(json.dumps(spec))
    duplicate["structures"]["rv_myocardium"]["xcat_parameter"] = "myoLV_act"
    with pytest.raises(ValueError, match="unique"):
        validate_phantom_spec(duplicate)
    geometry = json.loads(_frozen_geometry(tmp_path).read_text())
    assert derive_raw_contract(geometry) == {
        "shape": (8, 8, 8), "spacing_mm": (1.0, 1.0, 1.0), "axis_order": "zyx",
        "origin_policy": "centered_synthetic_lps",
    }
