"""CPU-only evidence-gate and five-subject registry contracts."""
from __future__ import annotations

import json
from pathlib import Path


def test_registry_uses_exact_canonical_subjects_and_keeps_gate_closed_without_server_evidence(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.five_subject_readiness_registry import build_five_subject_registry

    registry = build_five_subject_registry(tmp_path / "registry", evidence_root=tmp_path / "evidence")
    payload = json.loads((registry / "registry.json").read_text(encoding="utf-8"))
    assert payload["subjects"] == ["CYJ", "DYL_2", "DYZ", "HHZ", "HJL"]
    assert payload["all_subjects_ready_for_3d_reconstruction"] is False
    assert payload["launch_gate"]["state"] == "CLOSED"
    dyl2 = json.loads((registry / "subjects" / "DYL_2" / "subject_manifest.json").read_text(encoding="utf-8"))
    assert dyl2["stage"] == "BLOCKED"
    assert "CMRServer04 DNS" not in " ".join(dyl2["blockers"])


def test_readiness_requires_verified_standardization_reuse_dependency_and_server(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.five_subject_readiness_registry import readiness_from_evidence

    evidence = {"standardization": {"status": "STANDARDIZATION_COMPLETE", "valid": True}, "dictionary_reuse": {"status": "REUSE_VERIFIED"}, "frozen_dependencies": {"decoder_sha256_valid": True, "b6_sha256_valid": True}, "server": {"verified": False}}
    result = readiness_from_evidence("CYJ", evidence)
    assert result["stage"] == "BLOCKED"
    assert "reconstruction server availability is not verified" in result["blockers"]
    evidence["server"]["verified"] = True
    assert readiness_from_evidence("CYJ", evidence)["stage"] == "READY_FOR_3D_RECONSTRUCTION"


def test_dyl2_reported_admission_is_labeled_as_handoff_evidence(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.five_subject_readiness_registry import build_five_subject_registry

    registry = build_five_subject_registry(tmp_path / "registry", evidence_root=tmp_path / "evidence")
    payload = json.loads((registry / "subjects" / "DYL_2" / "subject_manifest.json").read_text(encoding="utf-8"))
    assert payload["raw_audit"]["status"] == "PASS"
    assert payload["raw_audit"]["evidence_type"] == "handoff_report"
    assert payload["protocol_admission"]["evidence_type"] == "handoff_report"
    assert payload["prepared"]["sax"]["status"] == "UNKNOWN_NOT_INSPECTED"
