"""CPU contract for explicit-mask direct-native native-plane metrics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def test_direct_native_metrics_only_consumes_explicit_phase4r_and_scmr_masks(monkeypatch, tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import direct_native_metrics as metrics

    bundle = tmp_path / "bundle"; (bundle / "planes" / "sax").mkdir(parents=True)
    (bundle / "manifest.json").write_text(json.dumps({"schema": "direct_native_prediction_bundle/v1", "planes": [{"stack": "sax", "original_group_idx": 3, "path": "planes/sax/group003.npz"}]}), encoding="utf-8")
    np.savez_compressed(bundle / "planes" / "sax" / "group003.npz", t1_ms=np.full((2, 2), 1000.0), t2_ms=np.full((2, 2), 80.0))
    masks = tmp_path / "masks"
    for parameter in ("t1", "t2"):
        target = masks / "planes" / parameter / "sax" / "group003.npz"; target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, phase4r_l100=np.ones((2, 2), bool), phase4r_l100_central_component=np.ones((2, 2), bool), scmr_style_central_component=np.array([[True, False], [True, True]]))
    sparse = tmp_path / "sparse.json"; sparse.write_text("{}", encoding="utf-8")
    cache = {"group_idx": np.array([3]), "source_support": np.ones((1, 2, 2), bool), "t1_ms": np.full((1, 2, 2), 900.0), "t2_ms": np.full((1, 2, 2), 70.0)}
    monkeypatch.setattr(metrics, "_load_all_verified_native_cache_groups", lambda *args: cache)
    output = metrics.build_direct_native_metrics(bundle_root=bundle, r0_native_maps_root=tmp_path, sparse_manifest=sparse, mask_root=masks, output=tmp_path / "out")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["metric_domains"] == ["scmr_style_central_component", "phase4r_l100", "phase4r_l100_central_component"]
    assert (output / "mask_sensitivity" / "aggregate_metrics.json").is_file()
