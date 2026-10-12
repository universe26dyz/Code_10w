from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.mrxcat.define_gt_tissue_maps import build_gt_maps
from paper_experiments.joint_vs_2dfit.simulation.build_geometry import apply_global_rigid_alignment
from paper_experiments.joint_vs_2dfit.simulation.generate_clean_dataset import acquisition_times_from_timing9
from paper_experiments.joint_vs_2dfit.simulation.simulate_hhz_signals import generate_signal_field


def test_signal_support_excludes_xcat_air_but_keeps_documented_tissue() -> None:
    maps = build_gt_maps(np.array([[0, 1, 5, 13, 15, 16, 31, 99]], dtype=np.int16))
    assert maps.signal_support_mask.tolist() == [[False, True, True, True, False, False, True, True]]
    assert maps.t1_ms[0, 0] == 0.0
    assert maps.t2_ms[0, 0] == 0.0


def test_signal_generation_is_exactly_zero_outside_support() -> None:
    support = np.array([True, False])
    signals = generate_signal_field(
        np.array([1410.0, 0.0], dtype=np.float32),
        np.array([50.0, 0.0], dtype=np.float32),
        np.ones(2, dtype=np.float32),
        np.arange(1, 10, dtype=np.float32),
        signal_support_mask=support,
    )
    assert np.any(signals[0] != 0.0)
    assert np.array_equal(signals[1], np.zeros(10, dtype=np.float32))


def test_one_global_rigid_transform_preserves_relative_geometry() -> None:
    source = {
        "sax": {"groups": [{"group_idx": 0, "image_shape_rows_cols": [2, 2], "pixel_spacing_rc_mm": [1.0, 1.0], "slice_thickness_mm": 8.0,
                            "affine_lps_rc": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 8, 0], [0, 0, 0, 1]]}]},
        "2ch": {"groups": [{"group_idx": 0, "image_shape_rows_cols": [2, 2], "pixel_spacing_rc_mm": [1.0, 1.0], "slice_thickness_mm": 8.0,
                            "affine_lps_rc": [[0, 0, 8, 5], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]}]},
    }
    aligned, qc = apply_global_rigid_alignment(source, np.array([10.0, 20.0, 30.0]), np.array([0.0, 0.0, 1.0]))
    assert abs(qc["sax_normal_dot_lv_long_axis"]) > 0.999999
    assert np.isclose(np.linalg.det(np.asarray(qc["rotation_matrix"])), 1.0)
    before = np.array(source["2ch"]["groups"][0]["affine_lps_rc"])[:3, 3] - np.array(source["sax"]["groups"][0]["affine_lps_rc"])[:3, 3]
    after = np.array(aligned["2ch"][0]["affine_lps_rc"])[:3, 3] - np.array(aligned["sax"][0]["affine_lps_rc"])[:3, 3]
    assert np.isclose(np.linalg.norm(before), np.linalg.norm(after))


def test_acquisition_times_round_trip_to_selected_timing() -> None:
    timing9 = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])
    acq = acquisition_times_from_timing9(timing9, tr_ms=2.61, vps=87)
    assert acq.shape == (10,)
    assert acq[0] == 0.0
    assert np.all(np.diff(acq) > 0.0)
    np.testing.assert_allclose(np.diff(acq) - 2.61 * 87, timing9 + np.array([0, 0, 0, 150, 0, 0, 35, 45, 55]))


def test_phase1_1_output_contract_if_present() -> None:
    root = Path(__file__).resolve().parents[1] / "simulation/outputs/phase1_1_clean_mrxcat"
    if not root.exists():
        pytest.skip("Phase-1.1 output is generated after source-only validation")
    field = np.load(root / "clean_signal_field.npz")
    assert np.array_equal(field["signals"][~field["signal_support_mask"]], np.zeros((int((~field["signal_support_mask"]).sum()), 10), dtype=np.float32))
    manifest = json.loads((root / "synthetic_source_manifest.json").read_text())
    assert manifest["mesh_primary_myocardium"]["isotropic_spacing_mm"] == 1.0
    assert manifest["geometry_alignment_qc"]["global_transform_only"] is True
    assert manifest["acquisition_time_ms"]["round_trip_verified"] is True
    for stack in ("sax", "2ch", "4ch"):
        data = np.load(root / stack / "observations.npz")
        assert np.all(np.diff(data["acquisition_time_ms"][:10]) > 0.0)
        assert np.all(np.isfinite(data["images"]))
        assert np.all(data["vps"] == 87)
