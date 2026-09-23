import importlib.util
import json
import sys
from pathlib import Path

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


def test_approval_copies_only_review_metadata_after_sha_match(tmp_path):
    module = _load_script()
    source = tmp_path / "candidate.pth"; write_functional_checkpoint(source)
    payload = torch.load(source, weights_only=False)
    payload.update({"functional_fixture": False, "scientific_checkpoint": True, "formal_candidate": True, "validation_status": "unvalidated"})
    torch.save(payload, source)
    report = tmp_path / "validation.json"
    report.write_text(json.dumps({"checkpoint_sha256": sha256_file(source), "validation_status": "awaiting_manual_review"}), encoding="utf-8")
    approved = tmp_path / "approved.pth"
    result = module.approve_formal_checkpoint(source, report, approved, "reviewed by physicist")
    assert result["validation_status"] == "approved_by_manual_review"
    assert torch.load(source, weights_only=False)["validation_status"] == "unvalidated"
    saved = torch.load(approved, weights_only=False)
    assert saved["source_checkpoint_sha256"] == sha256_file(source)
    assert saved["approval_review_note"] == "reviewed by physicist"
