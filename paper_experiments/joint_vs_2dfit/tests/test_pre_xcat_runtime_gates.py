from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.simulation.render_observations import coverage_preflight
from paper_experiments.joint_vs_2dfit.simulation.render_observations import _footprint_bounds_extrema
from paper_experiments.joint_vs_2dfit.xcat3d.audits import quantitative_tuple_audit
from paper_experiments.joint_vs_2dfit.xcat3d.prepare_xcat_parameters import (
    derive_raw_contract,
    prepare_xcat_parameter_hand_off,
    validate_parameter_contract,
    validate_phantom_spec,
)
from paper_experiments.joint_vs_2dfit.xcat3d.validate_quantitative_gt import assert_formal_code_provenance


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
    for filename in ("t1_phase1_2.par", "t2_phase1_2.par", "pd_phase1_2.par", "lv_mask_phase1_2.par"):
        assert "subvoxel_index = 2" in (handoff / "parameter_templates" / filename).read_text()


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
    contract = derive_raw_contract(geometry)
    assert contract["raw_shape"] == (8, 8, 8)
    assert contract["canonical_shape_xyz"] == (8, 8, 8)
    assert contract["origin_mm_xyz"] == (-3.5, -3.5, -3.5)


def test_asymmetric_centered_origin_is_world_xyz_not_raw_axis_order(tmp_path: Path) -> None:
    geometry = json.loads(_frozen_geometry(tmp_path).read_text())
    geometry["xcat_geometry_block"].update({"array_size": 256, "startslice": 500, "endslice": 780, "pixel_width": 0.1, "slice_width": 0.1})
    contract = derive_raw_contract(geometry)
    assert contract["raw_shape"] == (281, 256, 256)
    assert contract["canonical_shape_xyz"] == (256, 256, 281)
    assert contract["canonical_spacing_mm_xyz"] == (1.0, 1.0, 1.0)
    assert contract["origin_mm_xyz"] == (-127.5, -127.5, -140.0)
    centre = np.asarray(contract["origin_mm_xyz"]) + 0.5 * (np.asarray(contract["canonical_shape_xyz"]) - 1) * np.asarray(contract["canonical_spacing_mm_xyz"])
    np.testing.assert_allclose(centre, 0.0)
    geometry["xcat_geometry_block"]["slice_width"] = 0.2
    anisotropic = derive_raw_contract(geometry)
    assert anisotropic["canonical_spacing_mm_xyz"] == (1.0, 1.0, 2.0)
    assert anisotropic["origin_mm_xyz"] == (-127.5, -127.5, -280.0)


def test_extrema_coverage_matches_bruteforce_small_fixture() -> None:
    group = {"image_shape_rows_cols": [3, 4], "affine_lps_rc": [[1.0, 0.3, 0.1, 1.2], [0.1, 1.0, -0.2, 1.1], [0.0, 0.2, 1.0, 0.9], [0, 0, 0, 1]]}
    minimum, maximum, margin = _footprint_bounds_extrema((8, 9, 7, 10), np.zeros(3), np.ones(3), group, 6, 9)
    nodes = []
    for row in range(3):
        for col in range(4):
            for ro in ((np.arange(6) + .5) / 6 - .5):
                for co in ((np.arange(6) + .5) / 6 - .5):
                    for so in ((np.arange(9) + .5) / 9 - .5):
                        nodes.append(np.array(group["affine_lps_rc"], float)[:3, 3] + (row + ro) * np.array(group["affine_lps_rc"], float)[:3, 0] + (col + co) * np.array(group["affine_lps_rc"], float)[:3, 1] + so * np.array(group["affine_lps_rc"], float)[:3, 2])
    nodes = np.asarray(nodes)
    np.testing.assert_allclose(minimum, nodes.min(axis=0))
    np.testing.assert_allclose(maximum, nodes.max(axis=0))
    assert margin == pytest.approx(np.minimum(nodes, (np.array([8, 9, 7]) - 1) - nodes).min())


def test_formal_code_provenance_ignores_untracked_but_rejects_tracked_drift(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "tracked.txt").write_text("base")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=tmp_path, check=True, capture_output=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    source = {"code_provenance": {"git_commit": commit}}
    assert_formal_code_provenance(source, tmp_path)
    (tmp_path / "untracked.txt").write_text("allowed")
    assert_formal_code_provenance(source, tmp_path)
    (tmp_path / "tracked.txt").write_text("modified")
    with pytest.raises(ValueError, match="clean tracked"):
        assert_formal_code_provenance(source, tmp_path)
