import importlib.util
import json
from pathlib import Path

import pytest
import torch

from mlp.modules.module_05_signal_decoder.provenance import sha256_file
from online_helpers import write_functional_checkpoint


MLP_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location("approve_formal_checkpoint", MLP_ROOT / "scripts" / "approve_formal_checkpoint.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _candidate(tmp_path):
    source = tmp_path / "candidate.pth"; write_functional_checkpoint(source)
    payload = torch.load(source, weights_only=False)
    payload.update({"functional_fixture": False, "scientific_checkpoint": True, "formal_candidate": True,
                    "validation_status": "unvalidated", "dataset_schema": "mlp_rr_synthetic/v1", "dataset_split_mode": "rhythm"})
    torch.save(payload, source)
    return source


def _reports(tmp_path, source, *, coverage="PASS", recommendation="eligible_for_human_review", checkpoint_sha=None):
    validation = tmp_path / "validation.json"
    validation.write_text(json.dumps({"checkpoint_sha256": sha256_file(source), "validation_status": "awaiting_manual_review"}), encoding="utf-8")
    real = tmp_path / "real_timing_validation.json"
    real.write_text(json.dumps({"schema": "rr_real_vps87_validation/v1", "checkpoint_sha256": checkpoint_sha or sha256_file(source),
                                "training_domain_coverage_status": coverage, "approval_recommendation": recommendation,
                                "validation_status": "awaiting_manual_review"}), encoding="utf-8")
    return validation, real


def test_approval_copies_review_metadata_only_after_both_validation_gates_pass(tmp_path):
    module = _load_script(); source = _candidate(tmp_path); validation, real = _reports(tmp_path, source); approved = tmp_path / "approved.pth"
    result = module.approve_formal_checkpoint(source, validation, real, approved, "reviewed by physicist")
    assert result["validation_status"] == "approved_by_manual_review"
    assert torch.load(source, weights_only=False)["validation_status"] == "unvalidated"
    saved = torch.load(approved, weights_only=False)
    assert saved["source_checkpoint_sha256"] == sha256_file(source)
    assert saved["validation_report_sha256"] == sha256_file(validation)
    assert saved["real_timing_validation_report_sha256"] == sha256_file(real)
    assert saved["approval_review_note"] == "reviewed by physicist"


@pytest.mark.parametrize(("coverage", "recommendation"), [
    ("OUTSIDE_TRAINING_DOMAIN", "do_not_approve"),
    ("PASS", "do_not_approve"),
])
def test_approval_rejects_non_eligible_real_timing_report(tmp_path, coverage, recommendation):
    module = _load_script(); source = _candidate(tmp_path); validation, real = _reports(tmp_path, source, coverage=coverage, recommendation=recommendation)
    with pytest.raises(ValueError, match="real VPS=87 timing/fidelity"):
        module.approve_formal_checkpoint(source, validation, real, tmp_path / "approved.pth", "reviewed")


def test_approval_rejects_real_timing_report_for_different_checkpoint(tmp_path):
    module = _load_script(); source = _candidate(tmp_path); validation, real = _reports(tmp_path, source, checkpoint_sha="0" * 64)
    with pytest.raises(ValueError, match="Real timing validation report checkpoint_sha256"):
        module.approve_formal_checkpoint(source, validation, real, tmp_path / "approved.pth", "reviewed")
