from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.xcat3d.canonicalize import canonicalize_volume
from paper_experiments.joint_vs_2dfit.xcat3d.finalize import REQUIRED_ARTIFACTS, REQUIRED_RAW_XCAT_VOLUMES, _sha256, assert_final_manifest_ready, finalize_phase1_2
from paper_experiments.joint_vs_2dfit.xcat3d.prepare_xcat_parameters import prepare_xcat_parameter_hand_off


def test_hanson_handoff_records_reference_and_frozen_lv_values(tmp_path: Path) -> None:
    handoff = Path(prepare_xcat_parameter_hand_off(tmp_path)["handoff_root"])
    provenance = json.loads((handoff / "reference_design/reference_provenance.json").read_text())
    assert provenance["reference"]["commit"] == "97558db05f4e344865f472de974b9f4d042a1a15"
    assert set(provenance["reference"]["files"]) >= {"t1", "t2", "pd", "loader_example"}
    assert "myoLV_act = 1410" in (handoff / "parameter_templates/t1_phase1_2.par").read_text()
    assert "myoLV_act = 50" in (handoff / "parameter_templates/t2_phase1_2.par").read_text()
    assert "myoLV_act = 1" in (handoff / "parameter_templates/lv_mask_phase1_2.par").read_text()


def test_canonicalization_reorders_zyx_ras_to_xyz_lps_at_one_mm() -> None:
    raw = np.zeros((2, 3, 4), np.float32)
    raw[1, 2, 3] = 7.0
    array, geometry = canonicalize_volume(raw, shape=(2, 3, 4), spacing_mm=(1.0, 1.0, 1.0), origin_mm=(10.0, 20.0, 30.0), axis_order="zyx", world_convention="RAS", interpolation="nearest")
    assert geometry.axis_order == "xyz" and geometry.world_convention == "LPS"
    assert geometry.spacing_mm == (1.0, 1.0, 1.0)
    assert array.shape == (4, 3, 2)
    assert array[0, 0, 1] == 7.0


@pytest.mark.parametrize("axis_order,world_convention", (("xyz", "LPS"), ("zyx", "LPS"), ("xyz", "RAS"), ("zyx", "RAS")))
def test_canonicalization_preserves_marked_voxel_physical_coordinates(axis_order: str, world_convention: str) -> None:
    """Axis labels and RAS/LPS conversion must preserve a marked physical point."""
    spatial_shape, spatial_spacing, spatial_index = (5, 6, 7), (2.0, 3.0, 4.0), (1, 2, 3)
    raw_shape = tuple(spatial_shape["xyz".index(axis)] for axis in axis_order)
    raw_spacing = tuple(spatial_spacing["xyz".index(axis)] for axis in axis_order)
    raw_index = tuple(spatial_index["xyz".index(axis)] for axis in axis_order)
    raw = np.zeros(raw_shape, np.float32); raw[raw_index] = 99.0
    canonical, geometry = canonicalize_volume(raw, shape=raw_shape, spacing_mm=raw_spacing, origin_mm=(10.0, 20.0, 30.0), axis_order=axis_order, world_convention=world_convention, interpolation="nearest")
    source_world = np.array((12.0, 26.0, 42.0))
    lps_world = source_world if world_convention == "LPS" else np.array((-12.0, -26.0, 42.0))
    canonical_index = tuple(np.rint(lps_world - np.asarray(geometry.origin_mm)).astype(int))
    assert canonical[canonical_index] == 99.0
    assert np.array_equal(np.asarray(geometry.origin_mm) + np.asarray(canonical_index), lps_world)


def test_finalizer_refuses_missing_artifacts(tmp_path: Path) -> None:
    (tmp_path / "synthetic_source_manifest.json").write_text(json.dumps({"phase": "1.2", "source_family": "single_source_xcat_3d_quantitative", "scientific_gt_geometry_gate": True}))
    with pytest.raises(FileNotFoundError, match="required Phase-1.2 artifact"):
        finalize_phase1_2(tmp_path)


def test_cardiac_roi_spec_drives_all_parameter_files_and_lv_mask_is_explicit(tmp_path: Path) -> None:
    """A missing structure assignment must not fall through to an XCAT default."""
    handoff = Path(prepare_xcat_parameter_hand_off(tmp_path)["handoff_root"])
    spec = json.loads((handoff / "cardiac_roi_phantom_spec.json").read_text())
    geometry = json.loads((handoff / "cardiac_roi_geometry.template.json").read_text())
    assert spec["frozen_lv"] == {"t1_ms": 1410.0, "t2_ms": 50.0, "b1": 1.0}
    assert spec["m0_policy"] == "controlled_binary_signal_support_m0_1"
    assert spec["support_policy"] == "native_xcat_pd_positive"
    assert geometry["minimum_lv_boundary_margin_mm"] == 5.0
    assert len(spec["structures"]) >= 12
    for filename in ("t1_phase1_2.par", "t2_phase1_2.par", "pd_phase1_2.par", "lv_mask_phase1_2.par"):
        assignments = {
            line.split(" = ")[0]: line.split(" = ")[1]
            for line in (handoff / "parameter_templates" / filename).read_text().splitlines()
            if "_act" in line and " = " in line
        }
        assert assignments["myoLV_act"] == ({"t1_phase1_2.par": "1410", "t2_phase1_2.par": "50", "pd_phase1_2.par": "1346", "lv_mask_phase1_2.par": "1"}[filename])
        assert set(assignments) >= {record["xcat_parameter"] for record in spec["structures"].values()}
        if filename == "lv_mask_phase1_2.par":
            assert all(value == "0" for parameter, value in assignments.items() if parameter != "myoLV_act")


def test_final_manifest_rejects_empty_required_hash_set(tmp_path: Path) -> None:
    """A true flag alone must never authorize Phase-2."""
    (tmp_path / "phase1_2_final_manifest.json").write_text(json.dumps({"phase": "1.2", "formal_synthetic_ready": True, "artifact_sha256": {}, "raw_xcat_volume_sha256": {}}))
    with pytest.raises(ValueError, match="artifact_sha256"):
        assert_final_manifest_ready(tmp_path)


def _complete_final_manifest(root: Path) -> None:
    for relative in REQUIRED_ARTIFACTS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    (root / "phase1_2_final_manifest.json").write_text(json.dumps({
        "phase": "1.2", "formal_synthetic_ready": True,
        "artifact_sha256": {relative: _sha256(root / relative) for relative in REQUIRED_ARTIFACTS},
        "raw_xcat_volume_sha256": {name: "a" * 64 for name in REQUIRED_RAW_XCAT_VOLUMES},
        "cardiac_roi_phantom_spec_sha256": "b" * 64,
        "cardiac_roi_geometry_spec_sha256": "c" * 64,
    }))


@pytest.mark.parametrize("artifact", REQUIRED_ARTIFACTS)
def test_final_manifest_rejects_any_tampered_required_artifact(tmp_path: Path, artifact: str) -> None:
    """No required GT, signal, observation or source file may drift after finalization."""
    _complete_final_manifest(tmp_path)
    (tmp_path / artifact).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="final artifact SHA mismatch"):
        assert_final_manifest_ready(tmp_path)


def test_final_manifest_rejects_one_missing_required_artifact_hash(tmp_path: Path) -> None:
    """A manifest with even one omitted authority key cannot authorize Phase-2."""
    _complete_final_manifest(tmp_path)
    payload = json.loads((tmp_path / "phase1_2_final_manifest.json").read_text())
    del payload["artifact_sha256"]["4ch/observations.npz"]
    (tmp_path / "phase1_2_final_manifest.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="artifact_sha256"):
        assert_final_manifest_ready(tmp_path)


def test_frozen_geometry_drives_placeholder_free_parameter_handoff(tmp_path: Path) -> None:
    """Formal XCAT files must come from the selected frozen geometry, not the template."""
    geometry = json.loads((Path(__file__).parents[1] / "xcat3d/cardiac_roi_geometry.template.json").read_text())
    geometry["xcat_version_build_compatibility"] = "fixture-xcat"
    block = geometry["xcat_geometry_block"]
    block.update({"array_size": 256, "startslice": 500, "endslice": 780})
    frozen = tmp_path / "cardiac_roi_geometry.json"; frozen.write_text(json.dumps(geometry))
    handoff = Path(prepare_xcat_parameter_hand_off(tmp_path, geometry_spec=frozen)["handoff_root"])
    parameter = (handoff / "parameter_templates/t1_phase1_2.par").read_text()
    assert "array_size = 256" in parameter and "startslice = 500" in parameter and "REQUIRED_" not in parameter
