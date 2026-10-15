from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.xcat3d.prepare_xcat_parameters import freeze_xcat_parameter_hand_off
from paper_experiments.joint_vs_2dfit.xcat3d.scout_geometry import analyze_xcat_scout, inspect_xcat_scout_context, make_xcat_scout_geometry, propose_final_xcat_geometry


def test_context_reports_missing_local_mrxcat_without_guessing(tmp_path: Path) -> None:
    report = inspect_xcat_scout_context(tmp_path / "context", search_roots=(tmp_path / "empty",))
    assert report["mrxcat"]["status"] == "UNAVAILABLE_LOCALLY"
    assert report["xcat_version_build"]["status"] == "UNRESOLVED_EXTERNAL_INFO"


def test_unresolved_version_creates_candidate_not_freezable_scout(tmp_path: Path) -> None:
    report = make_xcat_scout_geometry(tmp_path / "scout")
    geometry = json.loads(Path(report["geometry_path"]).read_text())
    assert report["ready_to_freeze"] is False
    assert geometry["xcat_geometry_block"]["array_size"] == 480
    assert geometry["xcat_geometry_block"]["startslice"] == 950
    assert geometry["xcat_geometry_block"]["endslice"] == 1350
    assert geometry["xcat_geometry_block"]["subvoxel_index"] == 2
    assert report["raw_shape_zyx"] == [401, 480, 480]
    assert report["matrix_coverage_mm_xyz"] == [480.0, 480.0, 401.0]
    assert report["voxel_center_extent_mm_xyz"] == [479.0, 479.0, 400.0]
    with pytest.raises(ValueError, match="placeholders"):
        freeze_xcat_parameter_hand_off(tmp_path / "handoff", geometry_spec=report["geometry_path"])


def test_verified_version_creates_formally_valid_scout_geometry(tmp_path: Path) -> None:
    report = make_xcat_scout_geometry(tmp_path / "scout", xcat_version_build="XCAT fixture build")
    assert report["ready_to_freeze"] is True
    assert Path(report["geometry_path"]).name == "cardiac_roi_geometry.scout_v1.json"


def test_static_discovery_uses_template_filenames(tmp_path: Path) -> None:
    static = tmp_path / "XCAT_input_ap.dat"; static.write_bytes(b"metadata only")
    report = inspect_xcat_scout_context(tmp_path / "context", search_roots=(tmp_path,))
    item = report["static_inputs"]["ap_filename"]
    assert item["status"] == "FOUND"
    assert item["matches"][0]["path"] == str(static.resolve())


def _prepared_root(tmp_path: Path) -> Path:
    root = tmp_path / "prepared"
    for stack, affine in {"sax": np.eye(4), "2ch": np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float), "4ch": np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], float)}.items():
        path = root / "DYZ" / stack; path.mkdir(parents=True, exist_ok=True)
        (path / "manifest.json").write_text(json.dumps({"groups": [{"group_idx": 0, "tr_ms": 2.61, "vps": 87, "image_shape_rows_cols": [2, 2], "pixel_spacing_rc_mm": [1.0, 1.0], "slice_thickness_mm": 1.0, "affine_lps_rc": affine.tolist()}]}))
    return root


def _small_scout_geometry(tmp_path: Path) -> Path:
    report = make_xcat_scout_geometry(tmp_path / "scout", xcat_version_build="fixture")
    payload = json.loads(Path(report["geometry_path"]).read_text())
    payload["xcat_geometry_block"]["array_size"] = 8
    path = tmp_path / "small_scout.json"; path.write_text(json.dumps(payload))
    return path


def test_analyze_scout_maps_raw_z_to_absolute_xcat_slices(tmp_path: Path) -> None:
    geometry = _small_scout_geometry(tmp_path)
    raw = np.zeros((401, 8, 8), np.float32); raw[135:279, :, :] = 1
    mask = tmp_path / "mask.npy"; np.save(mask, raw)
    report = analyze_xcat_scout(geometry, mask, dtype="float32", prepared_root=_prepared_root(tmp_path), output_root=tmp_path / "analysis")
    assert report["lv_mask"]["absolute_xcat_slice_min_max"] == [1085, 1228]
    assert report["template_subject"] == "DYZ"


def test_proposal_rejects_geometry_sha_mismatch_and_keeps_or_grows_480(tmp_path: Path) -> None:
    report = make_xcat_scout_geometry(tmp_path / "scout", xcat_version_build="fixture")
    geometry = Path(report["geometry_path"])
    qc = {"scout_only": True, "geometry_sha256": report["geometry_sha256"], "lv_mask": {"sha256": "a" * 64, "canonical_bbox_xyz": [[200, 200, 150], [220, 220, 250]]}, "template_subject": "DYZ", "template_manifest_sha256": {stack: "b" * 64 for stack in ("sax", "2ch", "4ch")}, "acquisition_footprint_world_min_mm": [-100, -100, -100], "acquisition_footprint_world_max_mm": [100, 100, 100]}
    qc_path = tmp_path / "qc.json"; qc_path.write_text(json.dumps(qc))
    proposal = propose_final_xcat_geometry(qc_path, geometry, safety_margin_mm=20.0, output_root=tmp_path / "proposal")
    assert proposal["recommended_array_size"] == 480
    changed = json.loads(geometry.read_text()); changed["xcat_geometry_block"]["startslice"] = 949
    other = tmp_path / "other.json"; other.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="geometry SHA256"):
        propose_final_xcat_geometry(qc_path, other, safety_margin_mm=20.0, output_root=tmp_path / "bad")


def test_analyze_scout_rejects_nonbinary_or_wrong_sized_raw_mask(tmp_path: Path) -> None:
    geometry = _small_scout_geometry(tmp_path)
    bad = np.zeros((401, 8, 8), np.float32); bad[135:279] = 0.25
    path = tmp_path / "nonbinary.npy"; np.save(path, bad)
    with pytest.raises(ValueError, match="not safely binary"):
        analyze_xcat_scout(geometry, path, dtype="float32", prepared_root=_prepared_root(tmp_path), output_root=tmp_path / "bad")
    short = tmp_path / "short.bin"; short.write_bytes(b"\x00" * 4)
    with pytest.raises(ValueError, match="element count"):
        analyze_xcat_scout(geometry, short, dtype="float32", prepared_root=_prepared_root(tmp_path), output_root=tmp_path / "short")


def test_proposal_grows_array_when_480_is_insufficient(tmp_path: Path) -> None:
    report = make_xcat_scout_geometry(tmp_path / "scout", xcat_version_build="fixture")
    qc = {"scout_only": True, "geometry_sha256": report["geometry_sha256"], "lv_mask": {"sha256": "a" * 64, "canonical_bbox_xyz": [[200, 200, 150], [220, 220, 250]]}, "template_subject": "DYZ", "template_manifest_sha256": {stack: "b" * 64 for stack in ("sax", "2ch", "4ch")}, "acquisition_footprint_world_min_mm": [-300, -300, -100], "acquisition_footprint_world_max_mm": [300, 300, 100]}
    path = tmp_path / "qc.json"; path.write_text(json.dumps(qc))
    proposal = propose_final_xcat_geometry(path, report["geometry_path"], safety_margin_mm=20.0, output_root=tmp_path / "proposal")
    assert proposal["recommended_array_size"] > 480
    assert proposal["recommended_array_size"] % 16 == 0
