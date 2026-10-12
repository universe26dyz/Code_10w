from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from paper_experiments.joint_vs_2dfit.mrxcat.define_gt_tissue_maps import build_gt_maps
from paper_experiments.joint_vs_2dfit.simulation.build_geometry import apply_global_rigid_alignment
from paper_experiments.joint_vs_2dfit.simulation.generate_clean_dataset import acquisition_times_from_timing9
from paper_experiments.joint_vs_2dfit.simulation.simulate_hhz_signals import generate_signal_field
from paper_experiments.joint_vs_2dfit.simulation.audit_phase1_final_gate import comparison_metrics, choose_accepted_level, meets_plateau
from paper_experiments.joint_vs_2dfit.simulation.validate_hhz_parity import build_reference_config, validate_matlab_reference
from paper_experiments.joint_vs_2dfit.simulation.finalize_phase1_matlab_parity import build_final_gate, parity_passes


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


def test_final_convergence_metrics_use_foreground_myo_and_p99() -> None:
    reference = np.ones((4, 4, 10), dtype=float)
    value = reference.copy()
    value[0, 0] += 0.02
    myo = np.zeros((4, 4), dtype=bool)
    myo[1:3, 1:3] = True
    metrics = comparison_metrics(value, reference, myo)
    assert metrics["foreground_pixel_count"] == 16
    assert metrics["myocardium_intersecting_pixel_count"] == 4
    assert metrics["whole_foreground_normalized_max_abs"] == pytest.approx(0.02)
    assert metrics["myocardium_intersecting_pixel_max_abs"] == pytest.approx(0.0)
    assert metrics["p99_normalized_abs"] == pytest.approx(0.02)


def test_final_convergence_selects_lowest_global_plateau() -> None:
    passing = {"normalized_rmse": 0.004, "p99_normalized_abs": 0.009}
    failing = {"normalized_rmse": 0.006, "p99_normalized_abs": 0.001}
    assert meets_plateau(passing)
    metrics = {"6x6x9_vs_8x8x13": {stack: passing for stack in ("sax", "2ch", "4ch")}}
    assert choose_accepted_level(metrics)[0] == (6, 6, 9)
    metrics["6x6x9_vs_8x8x13"]["2ch"] = failing
    metrics["8x8x13_vs_10x10x17"] = {stack: passing for stack in ("sax", "2ch", "4ch")}
    assert choose_accepted_level(metrics)[0] == (8, 8, 13)


def test_external_matlab_reference_validator_reads_export_contract(tmp_path: Path) -> None:
    config = build_reference_config(np.arange(1.0, 10.0), cases=2)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    raw = generate_signal_field(np.asarray(config["T1"]), np.asarray(config["T2"]), np.asarray(config["B1"]), np.asarray(config["timing9_ms"])[0])
    reference = tmp_path / "reference.mat"
    savemat(reference, {"signal_raw": raw})
    result = validate_matlab_reference(reference, config_path)
    assert result["status"] == "pass"
    assert result["max_abs_raw_error"] == 0.0


def test_final_gate_requires_passing_external_matlab_parity() -> None:
    prior = {"renderer_convergence_status": "pass", "accepted_quadrature": {"inplane_samples_row_col": [6, 6], "thickness_samples": 9}, "frozen_contract": {"verified": True}, "relevant_sha256": {"clean_signal_field.npz": "frozen"}}
    passed = {"status": "pass", "max_abs_raw_error": 1e-5, "min_normalized_cosine": 0.999999, "all_finite": True}
    assert parity_passes(passed)
    gate = build_final_gate(prior, passed, {"reference_mat_sha256": "reference"})
    assert gate["matlab_parity_status"] == "pass"
    assert gate["formal_synthetic_reconstruction_ready"] is True
    failed = {**passed, "all_finite": False}
    assert not parity_passes(failed)
    with pytest.raises(ValueError, match="acceptance criteria"):
        build_final_gate(prior, failed, {})


def test_external_matlab_final_gate_contract_if_present() -> None:
    root = Path(__file__).resolve().parents[1] / "simulation/outputs/phase1_1_clean_mrxcat"
    gate_path = root / "phase1_final_gate_matlab_parity.json"
    if not gate_path.exists():
        pytest.skip("external MATLAB finalization artifact is generated after MATLAB reference transfer")
    gate = json.loads(gate_path.read_text())
    assert gate["matlab_parity_status"] == "pass"
    assert gate["formal_synthetic_reconstruction_ready"] is True
    assert gate["matlab_validator_result"]["max_abs_raw_error"] <= 1e-5
    assert gate["matlab_validator_result"]["min_normalized_cosine"] >= 0.999999
    assert gate["matlab_validator_result"]["all_finite"] is True
