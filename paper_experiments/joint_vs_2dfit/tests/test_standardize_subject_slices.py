"""Immutable physical-order slice standardization contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


def _write(path: Path, count: int, *, normal_sign: float = -1.0) -> None:
    rows = [(group, weight) for group in range(count) for weight in range(10)]
    groups = np.asarray([group for group, _weight in rows], np.int64)
    weights = np.asarray([weight for _group, weight in rows], np.int64)
    affine = np.repeat(np.eye(4)[None], len(rows), axis=0)
    affine[:, 1, 1] = normal_sign
    affine[:, 2, 3] = groups.astype(float) * 5.0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, images=np.ones((len(rows), 2, 2), np.float32), masks=np.ones((len(rows), 2, 2), bool), group_idx=groups, weight_idx=weights, stack_idx=np.zeros(len(rows), np.int64), acquisition_time_ms=np.arange(len(rows), dtype=float), timing9_ms=np.tile(np.arange(1.0, 10.0), (len(rows), 1)), affine_lps_rc=affine, pixel_spacing_rc_mm=np.ones((len(rows), 2)), slice_thickness_mm=np.full(len(rows), 8.0), tr_ms=np.full(len(rows), 2.61), vps=np.full(len(rows), 87, np.int64))


def _source(root: Path, counts: tuple[int, int, int] = (14, 15, 12)) -> Path:
    for stack, count in zip(("sax", "2ch", "4ch"), counts):
        _write(root / "CYJ" / stack / "observations.npz", count)
    return root


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_standardizer_uses_signed_physical_order_and_symmetric_or_high_end_trim(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.standardize_subject_slices import standardize_subject_slices

    source = _source(tmp_path / "prepared", (14, 16, 12))
    before = {stack: _sha(source / "CYJ" / stack / "observations.npz") for stack in ("sax", "2ch", "4ch")}
    output = standardize_subject_slices(source, "CYJ", tmp_path / "standardized", git_sha="test")
    manifest = json.loads((output / "slice_standardization_manifest.json").read_text(encoding="utf-8"))
    assert {stack: manifest["stacks"][stack]["output_count"] for stack in ("sax", "2ch", "4ch")} == {"sax": 12, "2ch": 15, "4ch": 10}
    assert manifest["stacks"]["sax"]["dropped_original_group_ids"] == [0, 13]
    assert manifest["stacks"]["2ch"]["dropped_original_group_ids"] == [15]
    assert manifest["stacks"]["2ch"]["odd_trim_tie_break"] == "drop_highest_signed_coordinate_outer_group"
    assert manifest["stacks"]["2ch"]["stack_normal_lps"][2] > 0
    assert all(manifest["stacks"][stack]["geometry_audit"]["regular_gap"] is True for stack in ("sax", "2ch", "4ch"))
    with np.load(output / "2ch" / "observations.npz") as data:
        assert np.unique(data["group_idx"]).tolist() == list(range(15))
    assert before == {stack: _sha(source / "CYJ" / stack / "observations.npz") for stack in before}


def test_standardizer_rejects_noncontiguous_physical_positions_and_dry_run_writes_nothing(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.standardize_subject_slices import standardize_subject_slices

    source = _source(tmp_path / "prepared")
    broken = source / "CYJ" / "sax" / "observations.npz"
    with np.load(broken, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    payload["affine_lps_rc"][payload["group_idx"] == 4, 2, 3] += 2.0
    np.savez_compressed(broken, **payload)
    with pytest.raises(ValueError, match="regular physical gaps"):
        standardize_subject_slices(source, "CYJ", tmp_path / "out")
    clean = _source(tmp_path / "clean")
    result = standardize_subject_slices(clean, "CYJ", tmp_path / "dry", dry_run=True)
    assert result["status"] == "DRY_RUN"
    assert not (tmp_path / "dry").exists()


def test_dyl2_without_completed_standardization_audit_is_only_ready(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.standardize_subject_slices import standardized_subject_readiness

    assert standardized_subject_readiness("DYL_2", tmp_path / "missing") == {"subject": "DYL_2", "status": "READY_FOR_STANDARDIZATION"}
