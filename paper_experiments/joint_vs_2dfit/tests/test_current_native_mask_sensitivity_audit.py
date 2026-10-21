"""Sensitivity domains remain current-native and never require historical masks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def test_current_native_sensitivity_audit_uses_primary_and_frozen_l100_only(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import mask_sensitivity_audit as audit

    values = np.ones((2, 2), float)
    result = audit.compare_mask_sensitivity(values, values, np.array([[True, False], [True, True]]), np.ones((2, 2), bool), source_support=np.ones((2, 2), bool))
    assert set(audit.MASKS) == {"primary_native_mask", "phase4r_l100"}
    assert result["primary_native_mask"]["metrics"]["rmse"] == 0.0
    root = tmp_path / "input"; target = root / "plane.npz"; target.parent.mkdir()
    np.savez_compressed(target, reference=values, prediction=values, primary_native_mask=np.ones((2, 2), bool), phase4r_l100=np.ones((2, 2), bool), source_support=np.ones((2, 2), bool))
    output = audit.run_mask_sensitivity_audit(root, tmp_path / "audit")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema"] == "phase4r_current_native_mask_sensitivity_audit/v1"
