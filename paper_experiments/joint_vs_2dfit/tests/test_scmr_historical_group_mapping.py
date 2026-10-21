"""Fail-closed contracts for historical SCMR-to-current group identities."""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.phase2_clean import scmr_historical_group_mapping as mapping


COUNTS = {"sax": 14, "2ch": 15, "4ch": 12}


def _cache(stack: str, *, duplicate_sax: bool = False) -> dict[str, object]:
    count = COUNTS[stack]
    ras = []
    for group in range(count):
        affine = np.eye(4); affine[:3, 3] = (100 * ("sax", "2ch", "4ch").index(stack), 0, group if not (duplicate_sax and stack == "sax") else 0)
        ras.append(affine)
    return {
        "group_idx": np.arange(count),
        "t1_ms": np.ones((count, 2, 2), np.float32),
        "t2_ms": np.ones((count, 2, 2), np.float32),
        "source_support": np.ones((count, 2, 2), bool),
        "affine_lps_rc": np.asarray([np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine for affine in ras]),
        "provenance": {"stack": stack, "r0": "frozen"},
    }


def _write(path: Path, affine: np.ndarray) -> str:
    nib.save(nib.Nifti1Image(np.ones((2, 2), np.float32), affine), str(path))
    return str(path)


def _manifest(tmp_path: Path, *, disagreement: bool = False, missing: bool = False) -> Path:
    records = []
    for parameter in ("T1", "T2"):
        for stack, count in COUNTS.items():
            for group in range(count):
                target = np.diag([-1.0, -1.0, 1.0, 1.0]) @ _cache(stack)["affine_lps_rc"][group]
                matched_group = group
                if disagreement and parameter == "T2" and stack == "sax" and group == 0:
                    matched_group = 1
                    target = np.diag([-1.0, -1.0, 1.0, 1.0]) @ _cache(stack)["affine_lps_rc"][matched_group]
                if missing and parameter == "T1" and stack == "sax" and group == 0:
                    target = target.copy(); target[0, 3] += 0.5
                records.append({"parameter": parameter, "stack": stack, "native_slice_id": 1000 + group, "original_ms": _write(tmp_path / f"{parameter}_{stack}_{group}.nii.gz", target)})
    path = tmp_path / "common_mask_manifest.json"; path.write_text(json.dumps({"subject_id": "CYJ", "slices": records}))
    return path


def _sparse(tmp_path: Path) -> Path:
    path = tmp_path / "sparse.json"; path.write_text(json.dumps({"subject": "CYJ", "condition": "L100"})); return path


def test_builds_82_explicit_mappings_without_assuming_slice_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mapping, "_load_all_verified_native_cache_groups", lambda _root, _sparse, stack: _cache(stack))
    output = mapping.build_scmr_historical_group_mapping(historical_common_mask_manifest=_manifest(tmp_path), r0_native_maps_root=tmp_path / "r0", sparse_manifest=_sparse(tmp_path), output=tmp_path / "mapping")
    payload = json.loads((output / "manifest.json").read_text())
    assert payload["schema"] == "scmr_historical_group_mapping/v1"
    assert len(payload["mappings"]) == 82
    assert payload["mappings"][0]["historical_native_slice_id"] >= 1000
    assert payload["t1_t2_consistency"]["passed"] is True
    with pytest.raises(FileExistsError):
        mapping.build_scmr_historical_group_mapping(historical_common_mask_manifest=_manifest(tmp_path), r0_native_maps_root=tmp_path / "r0", sparse_manifest=_sparse(tmp_path), output=output)


def test_discovery_reports_ambiguous_and_missing_groups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mapping, "_load_all_verified_native_cache_groups", lambda _root, _sparse, stack: _cache(stack, duplicate_sax=True))
    report = mapping.discover_scmr_historical_group_mapping(historical_common_mask_manifest=_manifest(tmp_path), r0_native_maps_root=tmp_path / "r0", sparse_manifest=_sparse(tmp_path))
    assert report["complete"] is False
    assert report["ambiguous"]
    assert all(item["candidate_original_group_idx"] for item in report["ambiguous"])


def test_discovery_reports_no_exact_match_without_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mapping, "_load_all_verified_native_cache_groups", lambda _root, _sparse, stack: _cache(stack))
    report = mapping.discover_scmr_historical_group_mapping(historical_common_mask_manifest=_manifest(tmp_path, missing=True), r0_native_maps_root=tmp_path / "r0", sparse_manifest=_sparse(tmp_path))
    assert report["complete"] is False
    assert any(item["reason"] == "no_exact_current_r0_group_match" for item in report["missing"])


def test_discovery_rejects_t1_t2_correspondence_disagreement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mapping, "_load_all_verified_native_cache_groups", lambda _root, _sparse, stack: _cache(stack))
    report = mapping.discover_scmr_historical_group_mapping(historical_common_mask_manifest=_manifest(tmp_path, disagreement=True), r0_native_maps_root=tmp_path / "r0", sparse_manifest=_sparse(tmp_path))
    assert report["complete"] is False
    assert report["t1_t2_consistency"]["passed"] is False
