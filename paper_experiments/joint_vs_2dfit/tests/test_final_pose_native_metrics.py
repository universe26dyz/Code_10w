"""Common-support primary-mask contracts for final native endpoints."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.phase2_clean import current_native_central_component_mask as native
from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256


def _mask_root(root: Path) -> Path:
    output = root / "masks"; planes = []
    for parameter in ("t1", "t2"):
        target = output / "planes" / parameter / "sax" / "group003.npz"; target.parent.mkdir(parents=True, exist_ok=True)
        values, support = np.ones((2, 2), float), np.ones((2, 2), bool)
        np.savez_compressed(target, source_support=support, phase4r_l100=support, base_native_support=support, primary_native_mask=support, reference=values)
        planes.append({"parameter": parameter, "stack": "sax", "original_group_idx": 3, "shape_rc": [2, 2], "output": {"path": str(target.relative_to(output)), "sha256": sha256(target)}})
    (output / "manifest.json").write_text(json.dumps({"schema": "phase4r_current_native_central_component_mask/v1", "planes": planes, "selection_rule": "test"}), encoding="utf-8")
    return output


def test_final_metrics_freeze_current_native_primary_on_single_common_support(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_pose_native_metrics as metrics

    direct = tmp_path / "direct"; (direct / "planes" / "sax").mkdir(parents=True)
    (direct / "manifest.json").write_text(json.dumps({"schema": "direct_native_prediction_bundle/v1", "endpoint": "central", "planes": [{"stack": "sax", "original_group_idx": 3, "path": "planes/sax/group003.npz"}]}), encoding="utf-8")
    np.savez_compressed(direct / "planes" / "sax" / "group003.npz", t1_ms=np.full((2, 2), 900.0), t2_ms=np.full((2, 2), 70.0))
    fit = tmp_path / "fit"; (fit / "manifest.json").parent.mkdir(); (fit / "manifest.json").write_text(json.dumps({"schema": "fit_first_native_prediction_bundle/v2", "endpoint": "central"}), encoding="utf-8")
    for parameter, value in (("t1", 910.0), ("t2", 75.0)):
        target = fit / "planes" / parameter / "sax"; target.mkdir(parents=True)
        np.savez_compressed(target / "group003.npz", prediction_ms=np.full((2, 2), value), valid=np.ones((2, 2), bool))
    cache = {"group_idx": np.array([3]), "source_support": np.ones((1, 2, 2), bool), "t1_ms": np.full((1, 2, 2), 1000.0), "t2_ms": np.full((1, 2, 2), 80.0)}
    monkeypatch.setattr(metrics, "_load_all_verified_native_cache_groups", lambda *args: cache)
    sparse = tmp_path / "sparse.json"; sparse.write_text(json.dumps({"schema_version": "phase4r_sparse_v1", "subject": "DYL_2", "condition": "L100"}), encoding="utf-8")
    output = metrics.build_final_pose_native_metrics(direct_bundle_root=direct, fit_first_bundle_root=fit, r0_native_maps_root=tmp_path, sparse_manifest=sparse, mask_root=_mask_root(tmp_path), output=tmp_path / "out")
    aggregate = json.loads((output / "aggregate_metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert aggregate["primary_mask"] == "primary_native_mask"
    assert aggregate["primary_aggregate"]["t1"]["common_support_voxels"] == 4
    assert manifest["primary_support_equation"] == "primary_native_mask AND finite(reference) AND proposed_valid AND fit_first_valid AND finite(proposed, fit_first)"
    assert manifest["sensitivity_masks"] == ["phase4r_l100"]
    assert manifest["subject"] == "DYL_2"


def test_final_metrics_fail_closed_on_legacy_mask_provenance(tmp_path: Path) -> None:
    root = _mask_root(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8")); manifest["schema"] = "scmr_central_component_mask/v2"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="phase4r_current_native"):
        native.load_validated_current_native_mask_plane(root, parameter="t1", stack="sax", original_group_idx=3, shape_rc=(2, 2), source_support=np.ones((2, 2), bool))


def test_current_native_mask_loader_rejects_native_plane_shape_mismatch(tmp_path: Path) -> None:
    root = _mask_root(tmp_path)
    with pytest.raises(ValueError, match="shape does not match reference"):
        native.load_validated_current_native_mask_plane(root, parameter="t1", stack="sax", original_group_idx=3, shape_rc=(2, 3), source_support=np.ones((2, 3), bool))
