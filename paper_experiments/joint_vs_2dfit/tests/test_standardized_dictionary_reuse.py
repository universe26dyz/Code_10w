"""CPU contracts for immutable standardization-aware native-map reuse."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256


def _observation(path: Path, groups: list[int]) -> None:
    rows = [(group, weight) for group in groups for weight in range(10)]
    group_idx = np.asarray([row[0] for row in rows], np.int64)
    weight_idx = np.asarray([row[1] for row in rows], np.int64)
    affine = np.repeat(np.eye(4)[None], len(rows), axis=0)
    affine[:, 2, 3] = group_idx
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, images=group_idx[:, None, None].astype(np.float32), masks=np.ones((len(rows), 1, 1), bool), group_idx=group_idx, weight_idx=weight_idx, affine_lps_rc=affine, pixel_spacing_rc_mm=np.ones((len(rows), 2)), slice_thickness_mm=np.ones(len(rows)), timing9_ms=np.tile(np.arange(9.0), (len(rows), 1)), tr_ms=np.full(len(rows), 2.61), vps=np.full(len(rows), 87))


def _standardized_observation(path: Path, original_groups: list[int]) -> None:
    _observation(path, original_groups)
    with np.load(path, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    payload["group_idx"] = np.repeat(np.arange(len(original_groups), dtype=np.int64), 10)
    np.savez_compressed(path, **payload)


def _native(root: Path, observations: Path, groups: list[int]) -> None:
    root.mkdir(parents=True)
    values = np.asarray(groups, np.float32)[:, None, None]
    np.savez_compressed(root / "native_maps.npz", t1_ms=values, t2_ms=values + 1, b1=values + 2, group_idx=np.asarray(groups))
    affine = np.repeat(np.eye(4)[None], len(groups), axis=0); affine[:, 2, 3] = groups
    np.savez_compressed(root / "geometry.npz", affine_lps_rc=affine, pixel_spacing_rc_mm=np.ones((len(groups), 2)), slice_thickness_mm=np.ones(len(groups)))
    (root / "manifest.json").write_text(json.dumps({"source_observations": {"path": str(observations), "sha256": sha256(observations)}, "native_maps_sha256": sha256(root / "native_maps.npz"), "geometry_sha256": sha256(root / "geometry.npz")}), encoding="utf-8")


def test_audit_reindexes_each_native_map_by_original_group_identity(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.reuse_standardized_native_dictionary_maps import audit_standardized_native_dictionary_maps

    full, standardized, native = tmp_path / "full", tmp_path / "standardized", tmp_path / "native"
    stacks = {"sax": [0, 1], "2ch": [3, 4], "4ch": [7, 8]}
    records = {}
    for stack, groups in stacks.items():
        _observation(full / stack / "observations.npz", groups)
        _standardized_observation(standardized / stack / "observations.npz", list(reversed(groups)))
        _native(native / stack, full / stack / "observations.npz", groups)
        records[stack] = {"source_observation": {"path": str(full / stack / "observations.npz"), "sha256": sha256(full / stack / "observations.npz")}, "output_observation": {"path": str(standardized / stack / "observations.npz"), "sha256": sha256(standardized / stack / "observations.npz")}, "source_count": 2, "output_count": 2, "retained_original_group_ids": list(reversed(groups)), "derived_to_original_group_idx": {"0": groups[1], "1": groups[0]}}
    standardization = tmp_path / "standardization.json"
    standardization.write_text(json.dumps({"schema": "phase4r_common_slice_standardization/v1", "subject": "DYL_2", "stacks": records}), encoding="utf-8")
    out = audit_standardized_native_dictionary_maps(standardization, native, tmp_path / "reuse")
    manifest = json.loads((out / "reuse_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "REUSE_VERIFIED"
    with np.load(out / "2ch" / "native_maps.npz") as maps:
        assert maps["group_idx"].tolist() == [0, 1]
    assert manifest["stacks"]["2ch"]["standardized_to_original_group_idx"] == {"0": 4, "1": 3}
    with pytest.raises(FileExistsError):
        audit_standardized_native_dictionary_maps(standardization, native, out)


def test_audit_fails_closed_on_missing_or_mismatched_group(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.reuse_standardized_native_dictionary_maps import audit_standardized_native_dictionary_maps

    full, standardized, native = tmp_path / "full", tmp_path / "standardized", tmp_path / "native"
    records = {}
    for stack in ("sax", "2ch", "4ch"):
        _observation(full / stack / "observations.npz", [0, 1]); _observation(standardized / stack / "observations.npz", [0, 1]); _native(native / stack, full / stack / "observations.npz", [0])
        records[stack] = {"source_observation": {"path": str(full / stack / "observations.npz"), "sha256": sha256(full / stack / "observations.npz")}, "output_observation": {"path": str(standardized / stack / "observations.npz"), "sha256": sha256(standardized / stack / "observations.npz")}, "source_count": 2, "output_count": 2, "retained_original_group_ids": [0, 1], "derived_to_original_group_idx": {"0": 0, "1": 1}}
    path = tmp_path / "standardization.json"; path.write_text(json.dumps({"schema": "phase4r_common_slice_standardization/v1", "subject": "CYJ", "stacks": records}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing native group"):
        audit_standardized_native_dictionary_maps(path, native, tmp_path / "reuse")
    assert not (tmp_path / "reuse").exists()
