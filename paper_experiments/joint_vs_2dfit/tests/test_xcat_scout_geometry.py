from __future__ import annotations

import json
from pathlib import Path

from paper_experiments.joint_vs_2dfit.xcat3d.scout_geometry import inspect_xcat_scout_context, make_xcat_scout_geometry


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


def test_verified_version_creates_formally_valid_scout_geometry(tmp_path: Path) -> None:
    report = make_xcat_scout_geometry(tmp_path / "scout", xcat_version_build="XCAT fixture build")
    assert report["ready_to_freeze"] is True
    assert Path(report["geometry_path"]).name == "cardiac_roi_geometry.scout_v1.json"
