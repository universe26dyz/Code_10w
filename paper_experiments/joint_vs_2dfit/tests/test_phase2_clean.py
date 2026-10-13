from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import (
    APPROVED_DECODER_SHA256,
    BASELINE_POSE_POLICY,
    FROZEN_GRID,
    assert_phase1_ready,
    compare_observation_hashes,
    validate_proposed_subject,
)
from paper_experiments.joint_vs_2dfit.phase2_clean.dictionary import build_dictionary, match_fingerprints, frozen_axis
from paper_experiments.joint_vs_2dfit.phase2_clean.evaluation import synthetic_metrics
from paper_experiments.joint_vs_2dfit.reference.migrate_verified_legacy_roi import _bundle_audit_entry


def test_phase2_frozen_dictionary_contract_and_on_grid_recovery() -> None:
    assert FROZEN_GRID["t1_ms"] == (20, 20, 500, 505, 5, 1500, 1520, 20, 2500)
    assert frozen_axis(FROZEN_GRID["t2_ms"])[-1] == 200
    dictionary = build_dictionary(np.array([1410.0]), np.array([50.0]), np.array([1.0]), np.arange(1.0, 10.0), 2.61, 87)
    result = match_fingerprints(dictionary["fingerprints"], dictionary["fingerprints"])
    assert result.tolist() == [0]


def test_phase2_contract_rejects_hash_mismatch_and_dyl() -> None:
    with pytest.raises(ValueError, match="observation hash mismatch"):
        compare_observation_hashes({"sax": "a"}, {"sax": "b"})
    with pytest.raises(ValueError, match="VPS=85"):
        validate_proposed_subject("DYL", 2.61, 85)
    assert APPROVED_DECODER_SHA256 == "f83280d4f73bfc9e0540b9606ee20bf83395339ad4e54dd0fb7be93ed9c9a7a9"
    assert BASELINE_POSE_POLICY["t1_t2_independent"] is True


def test_phase2_gate_and_mesh_evaluator_primary_full_mask() -> None:
    assert_phase1_ready({"renderer_convergence_status": "pass", "matlab_parity_status": "pass", "formal_synthetic_reconstruction_ready": True})
    reference = np.array([[[1000.0, 1100.0], [1200.0, 1300.0]]])
    prediction = reference + 10.0
    full = np.array([[[True, True], [True, False]]])
    report = synthetic_metrics(reference, prediction, full, np.ones_like(full, bool))
    assert report["primary_region"] == "full_mesh_myocardium"
    assert report["coverage_fraction"] == 1.0
    assert report["mae"] == 10.0


def test_phase06_subject_report_adapter_and_server_scripts() -> None:
    entry = {"legacy_roi": {"status": "PASS", "source_directory": "/legacy", "composite_reference": {"path": "/x"}, "manual_masks": {}}}
    assert _bundle_audit_entry(entry)["status"] == "PASS"
    root = Path(__file__).resolve().parents[1] / "phase2_clean/server_commands"
    names = ("server_preflight_phase2.sh", "run_phase2_synthetic_proposed.sh", "run_phase2_synthetic_fit_first.sh", "run_phase2_real_proposed.sh", "run_phase2_real_fit_first.sh", "evaluate_phase2.sh")
    for name in names:
        text = (root / name).read_text()
        assert "set -euo pipefail" in text and "not auto-launched" in text or name == "server_preflight_phase2.sh"
    assert "sha256sum" in (root / "server_preflight_phase2.sh").read_text()
