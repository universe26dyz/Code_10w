from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_experiments.joint_vs_2dfit.xcat3d.validate_quantitative_gt import assert_phase1_2_manifest_ready
from paper_experiments.joint_vs_2dfit.xcat3d.validate_quantitative_gt import check_active_phase1_2_wiring
from paper_experiments.joint_vs_2dfit.xcat3d.check_phase1_2_local_acceptance import local_acceptance_report


def test_phase1_2_manifest_gate_requires_passing_geometry_qc(tmp_path: Path) -> None:
    manifest = tmp_path / "synthetic_source_manifest.json"
    manifest.write_text(json.dumps({"phase": "1.2", "source_family": "single_source_xcat_3d_quantitative", "scientific_gt_geometry_gate": False}))
    with pytest.raises(ValueError, match="geometry gate"):
        assert_phase1_2_manifest_ready(manifest)
    manifest.write_text(json.dumps({"phase": "1.2", "scientific_gt_geometry_gate": True, "source_family": "single_source_xcat_3d_quantitative"}))
    assert assert_phase1_2_manifest_ready(manifest)["scientific_gt_geometry_gate"] is True


def test_active_wiring_has_no_invalid_composite_route() -> None:
    root = Path(__file__).resolve().parents[3]
    report = check_active_phase1_2_wiring(root)
    assert report["passed"], report["violations"]


def test_local_acceptance_remains_explicitly_unverified_without_xcat_files() -> None:
    report = local_acceptance_report(Path(__file__).resolve().parents[3])
    assert report["LOCAL_CODE_ACCEPTANCE"] == "PASS"
    assert report["EXTERNAL_XCAT_INPUT_GATE"] == "UNVERIFIED"
    assert report["FORMAL_SYNTHETIC_READY"] == "FAIL"
