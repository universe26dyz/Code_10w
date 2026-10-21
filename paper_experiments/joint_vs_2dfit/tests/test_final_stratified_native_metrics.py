"""Contracts for all-source observed/held-out native metric strata."""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256


def _endpoint(root: Path, method: str) -> Path:
    root.mkdir(parents=True)
    for parameter, value in (("t1", 900.0), ("t2", 70.0)):
        path = root / (f"{parameter.upper()}_3D.nii.gz" if method == "Proposed" else f"{parameter}/volume_ms.nii.gz")
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.full((4, 4, 4), value, np.float32), np.eye(4)), str(path))
    schema = "phase4r_formal_proposed_export/v1" if method == "Proposed" else "phase4r_fixed_l100_fit_first_export_bundle/v1"
    (root / "manifest.json").write_text(json.dumps({"schema": schema, "subject": "DYL_2", "condition": "L50_fold0", "endpoint": "central"}), encoding="utf-8")
    return root


def _mask_root(root: Path, groups: dict[str, list[int]]) -> Path:
    planes = []
    for parameter in ("t1", "t2"):
        for stack, ids in groups.items():
            for group in ids:
                path = root / "planes" / parameter / stack / f"group{group:03d}.npz"; path.parent.mkdir(parents=True, exist_ok=True)
                support = np.ones((2, 2), bool)
                np.savez_compressed(path, source_support=support, phase4r_l100=support, base_native_support=support, primary_native_mask=support, reference=np.ones((2, 2)))
                planes.append({"parameter": parameter, "stack": stack, "original_group_idx": group, "shape_rc": [2, 2], "output": {"path": str(path.relative_to(root)), "sha256": sha256(path)}})
    (root / "manifest.json").write_text(json.dumps({"schema": "phase4r_current_native_central_component_mask/v1", "planes": planes, "selection_rule": "test"}), encoding="utf-8")
    return root


def test_stratified_evaluator_pools_only_exact_paired_support_and_disjoint_regions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_stratified_native_metrics as stratified

    groups = {"sax": [0], "2ch": [0, 1], "4ch": [0, 1]}
    sparse = tmp_path / "sparse.json"
    sparse.write_text(json.dumps({"subject": "DYL_2", "condition": "L50", "phase": 0, "status": "ELIGIBLE", "stacks": {"sax": {}, "2ch": {"retained_original_group_ids": [0], "heldout_original_group_ids": [1]}, "4ch": {"retained_original_group_ids": [0], "heldout_original_group_ids": [1]}}}), encoding="utf-8")
    def cache(stack: str) -> dict[str, np.ndarray]:
        ids = groups[stack]
        return {"group_idx": np.asarray(ids), "source_support": np.ones((len(ids), 2, 2), bool), "affine_lps_rc": np.repeat(np.eye(4)[None], len(ids), axis=0), "t1_ms": np.full((len(ids), 2, 2), 1000.0), "t2_ms": np.full((len(ids), 2, 2), 80.0)}
    monkeypatch.setattr(stratified, "_load_all_verified_native_cache_groups", lambda _root, _manifest, stack: cache(stack))
    def reproject(values: np.ndarray, _affine: np.ndarray, _plane: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        valid = np.ones(shape, bool)
        if np.isclose(np.nanmean(values), 70.0) or np.isclose(np.nanmean(values), 900.0):
            valid.flat[-1] = False
        return np.full(shape, float(np.nanmean(values))), valid
    monkeypatch.setattr(stratified, "reproject_volume_to_native_plane", reproject)
    output = stratified.build_final_stratified_native_metrics(proposed_endpoint_root=_endpoint(tmp_path / "proposed", "Proposed"), fit_first_endpoint_root=_endpoint(tmp_path / "fit", "Fit-first"), native_mask_root=_mask_root(tmp_path / "masks", groups), r0_native_maps_root=tmp_path / "r0", sparse_manifest=sparse, output=tmp_path / "out")
    payload = json.loads((output / "aggregate_metrics.json").read_text(encoding="utf-8"))
    regions = payload["regions"]["t1"]
    assert {name: regions[name]["voxel_count"] for name in ("sax_observed", "lax_observed", "lax_heldout", "nonheldout_all", "all_lax", "all_source")} == {"sax_observed": 3, "lax_observed": 6, "lax_heldout": 6, "nonheldout_all": 9, "all_lax": 12, "all_source": 15}
    assert all(record["parameters"]["t1"]["common_support_voxels"] == 3 for record in payload["per_plane_records"])
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["no_final_pose_voxels"] is True

def test_sparse_group_classification_uses_manifest_identities_without_overlap() -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.final_stratified_native_metrics import classify_stratified_groups

    sparse = {
        "condition": "L50", "phase": 0,
        "stacks": {
            "sax": {"retained_original_group_ids": [0, 1]},
            "2ch": {"retained_original_group_ids": [0, 2], "heldout_original_group_ids": [1, 3]},
            "4ch": {"retained_original_group_ids": [4], "heldout_original_group_ids": [5]},
        },
    }
    regions, statuses = classify_stratified_groups(sparse, {"sax": [0, 1], "2ch": [0, 1, 2, 3], "4ch": [4, 5]})
    assert regions["sax_observed"] == {"sax": [0, 1]}
    assert regions["lax_observed"] == {"2ch": [0, 2], "4ch": [4]}
    assert regions["lax_heldout"] == {"2ch": [1, 3], "4ch": [5]}
    assert regions["all_source"] == {"sax": [0, 1], "2ch": [0, 1, 2, 3], "4ch": [4, 5]}
    assert statuses[("sax", 0)] == "observed"
    assert statuses[("2ch", 1)] == "held-out"


def test_l100_marks_all_source_observed_and_heldout_as_not_applicable() -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.final_stratified_native_metrics import classify_stratified_groups

    regions, statuses = classify_stratified_groups({"condition": "L100", "stacks": {}}, {"sax": [0], "2ch": [1], "4ch": [2]})
    assert regions["lax_heldout"] is None
    assert regions["nonheldout_all"] == regions["all_source"]
    assert set(statuses.values()) == {"observed"}
