from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.xcat3d.canonicalize import canonicalize_volume
from paper_experiments.joint_vs_2dfit.xcat3d.finalize import finalize_phase1_2
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


def test_finalizer_refuses_missing_artifacts(tmp_path: Path) -> None:
    (tmp_path / "synthetic_source_manifest.json").write_text(json.dumps({"phase": "1.2", "source_family": "single_source_xcat_3d_quantitative", "scientific_gt_geometry_gate": True}))
    with pytest.raises(FileNotFoundError, match="required Phase-1.2 artifact"):
        finalize_phase1_2(tmp_path)
