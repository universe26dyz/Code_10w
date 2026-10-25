"""Optional read-only regression against the immutable CYJ 1.1 artifacts."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.phase2_clean.final_native_grid_metrics import mask_overlap_counts


HISTORICAL_DEFAULT = Path(
    "/home/universe/SVR/data/Code_10w_v1/Code_10w_runs/paper_joint_vs_2dfit/"
    "formal_matched_q004_quality_reuse/CYJ_20261020/results/CYJ/L100_fold0/central/"
    "Q004S640_vs_fit-first"
)
DIRECT_AUDIT_DEFAULT = Path(
    "/home/universe/SVR/data/Code_10w_v1/Code_10w_runs/paper_joint_vs_2dfit/"
    "formal_matched_q004_quality_reuse/CYJ_20261020/direct_native_endpoint_audit/"
    "L100_fold0_central_v2_physical_ras_fix"
)


def _root(variable: str, default: Path) -> Path:
    return Path(os.environ.get(variable, str(default)))


def test_immutable_cyj_l100_formal_masks_and_metrics_regression() -> None:
    """Audit every directly matched historic 2CH/4CH plane without reconstruction."""
    historical = _root("CYJ_HISTORICAL_L100_ROOT", HISTORICAL_DEFAULT)
    direct_audit = _root("CYJ_DIRECT_NATIVE_AUDIT_ROOT", DIRECT_AUDIT_DEFAULT)
    if not (historical / "per_plane_metrics.json").is_file() or not (direct_audit / "manifest.json").is_file():
        pytest.skip("immutable CYJ 1.1 artifacts are not available on this host")
    planes = json.loads((historical / "per_plane_metrics.json").read_text(encoding="utf-8"))["planes"]
    totals = {name: 0 for name in ("native", "query_mask", "l100", "sparse", "recovered", "paired", "intersection", "union", "xor")}
    for plane in planes:
        parameter, stack, group = plane["parameter"], plane["stack"], int(plane["original_group_idx"])
        legacy_path = historical / "masks_native" / parameter / f"{stack}_group{group}.npz"
        audit_path = direct_audit / "planes" / parameter / stack / f"group{group:03d}.npz"
        assert legacy_path.is_file() and audit_path.is_file()
        with np.load(legacy_path, allow_pickle=False) as legacy, np.load(audit_path, allow_pickle=False) as direct:
            native, l100 = np.asarray(legacy["native_source"], bool), np.asarray(legacy["l100_native"], bool)
            sparse, recovered = np.asarray(legacy["sparse_region"], bool), np.asarray(legacy["recovered_only"], bool)
            assert np.array_equal(np.asarray(legacy["l100_full"], bool), native & l100)
            assert np.array_equal(recovered, (native & l100) & ~np.asarray(legacy["sparse_native"], bool))
            overlap = mask_overlap_counts(l100, np.asarray(direct["l100_native"], bool))
        totals["native"] += int(native.sum()); totals["query_mask"] += int(l100.sum()); totals["l100"] += int((native & l100).sum())
        totals["sparse"] += int(sparse.sum()); totals["recovered"] += int(recovered.sum())
        totals["paired"] += int(plane["regions"]["l100_full"]["paired_support_voxel_count"])
        totals["intersection"] += int(overlap["intersection_voxels"]); totals["union"] += int(overlap["union_voxels"]); totals["xor"] += int(overlap["xor_voxels"])
        for method in ("proposed", "fit_first"):
            assert all(np.isfinite(plane["regions"]["l100_full"][method][metric]) for metric in ("rmse", "mae", "bias"))
    assert len(planes) == 54
    assert totals == {"native": 788554, "query_mask": 660230, "l100": 623705, "sparse": 612718, "recovered": 10987, "paired": 623705, "intersection": 644088, "union": 687559, "xor": 43471}
