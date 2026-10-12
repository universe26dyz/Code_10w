"""Contract tests for the Phase 0 read-only audit utilities."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    audit_dir = str(ROOT / "audit")
    if audit_dir not in sys.path:
        sys.path.insert(0, audit_dir)
    path = ROOT / "audit" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_observations(path: Path, *, groups: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = groups * 10
    images = np.zeros((count, 3, 4), dtype=np.float32)
    np.savez(
        path,
        images=images,
        masks=np.ones_like(images, dtype=bool),
        group_idx=np.repeat(np.arange(groups), 10),
        weight_idx=np.tile(np.arange(10), groups),
        stack_idx=np.zeros(count, dtype=np.int64),
        acquisition_time_ms=np.arange(count, dtype=np.float64),
        timing9_ms=np.tile(np.arange(9, dtype=np.float64), (count, 1)),
        affine_lps_rc=np.tile(np.eye(4), (count, 1, 1)),
        pixel_spacing_rc_mm=np.tile([1.2, 1.3], (count, 1)),
        slice_thickness_mm=np.full(count, 8.0),
        tr_ms=np.full(count, 2.61),
        vps=np.full(count, 87),
    )


def test_decoder_audit_rejects_missing_checkpoint(tmp_path: Path) -> None:
    module = _load("audit_decoder_checkpoint")
    with pytest.raises(FileNotFoundError):
        module.audit_checkpoint(tmp_path / "missing.pth")


def test_decoder_audit_rejects_non_approved_metadata(tmp_path: Path) -> None:
    module = _load("audit_decoder_checkpoint")
    metadata = {
        "architecture": "12-200-200-200-10",
        "protocol_hhz_v1": {"tr_ms": 2.61, "vps": 87},
        "parameter_ranges": {"t1_ms": [20, 2500]},
        "validation_status": "awaiting_manual_review",
        "approved_by_manual_review": False,
        "timing9_min_ms": [0] * 9,
        "timing9_max_ms": [1] * 9,
        "train_timing9_min_ms": [0] * 9,
        "train_timing9_max_ms": [1] * 9,
    }
    checkpoint = tmp_path / "decoder.json"
    checkpoint.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="approved"):
        module.audit_checkpoint(checkpoint, require_approved=True)


def test_decoder_audit_extracts_protocol_metadata(tmp_path: Path) -> None:
    module = _load("audit_decoder_checkpoint")
    metadata = {
        "architecture": "12-200-200-200-10",
        "protocol_hhz_v1": {"tr_ms": 2.61, "vps": 87, "fa_deg": [45, 45, 45]},
        "parameter_ranges": {"t1_ms": [20, 2500]},
        "validation_status": "approved_by_manual_review",
        "approved_by_manual_review": True,
        "timing9_min_ms": [0] * 9,
        "timing9_max_ms": [1] * 9,
        "train_timing9_min_ms": [0] * 9,
        "train_timing9_max_ms": [1] * 9,
    }
    checkpoint = tmp_path / "decoder.json"
    checkpoint.write_text(json.dumps(metadata), encoding="utf-8")
    report = module.audit_checkpoint(checkpoint)
    assert report["approved_for_paper"] is True
    assert report["protocol_hhz_v1"]["tr_ms"] == 2.61


def test_cohort_audit_counts_complete_ten_weight_groups(tmp_path: Path) -> None:
    module = _load("audit_real_cohort")
    _write_observations(tmp_path / "prepared" / "S1" / "sax" / "observations.npz")
    row = module.inspect_observations(tmp_path / "prepared" / "S1" / "sax" / "observations.npz")
    assert row["group_count"] == 2
    assert row["weights_per_group"] == [10]
    assert row["eligible_full_data"] is True


def test_current_audit_output_schema(tmp_path: Path) -> None:
    module = _load("audit_current_method")
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("num_images: 10\n", encoding="utf-8")
    report = module.build_report(repo_root=tmp_path, protocol_path=protocol, reference_paths_path=None)
    assert {"git", "hhz_protocol", "inr_ranges", "expected_observation_schema"} <= set(report)


def test_audits_do_not_modify_existing_runs(tmp_path: Path) -> None:
    module = _load("audit_real_cohort")
    run_file = tmp_path / "runs" / "historical.txt"
    run_file.parent.mkdir(parents=True)
    run_file.write_text("preserve", encoding="utf-8")
    _write_observations(tmp_path / "prepared" / "S1" / "sax" / "observations.npz")
    module.audit_cohort(tmp_path / "prepared", preprocessed_root=None, native_reference_root=None, myocardium_mask_root=None)
    assert run_file.read_text(encoding="utf-8") == "preserve"
