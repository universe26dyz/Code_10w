"""Perform the explicit human approval transition for one formal MLP checkpoint."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from mlp.modules.module_05_signal_decoder.provenance import sha256_file


def approve_formal_checkpoint(checkpoint_path: str | Path, validation_report: str | Path, real_timing_validation: str | Path, approved_output: str | Path, review_note: str) -> dict[str, object]:
    checkpoint_path, validation_report, real_timing_validation, approved_output = Path(checkpoint_path), Path(validation_report), Path(real_timing_validation), Path(approved_output)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Formal candidate checkpoint does not exist: {checkpoint_path}")
    if not validation_report.is_file():
        raise FileNotFoundError(f"Validation report does not exist: {validation_report}")
    if not real_timing_validation.is_file():
        raise FileNotFoundError(f"Real timing validation report does not exist: {real_timing_validation}")
    if approved_output.exists():
        raise FileExistsError(f"Approved checkpoint output already exists: {approved_output}")
    if not isinstance(review_note, str) or not review_note.strip():
        raise ValueError("--review-note must be non-empty for explicit manual approval.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Formal candidate checkpoint must be a mapping.")
    if bool(checkpoint.get("functional_fixture")) or not bool(checkpoint.get("formal_candidate")) or checkpoint.get("validation_status") != "unvalidated":
        raise ValueError("Approval requires one unvalidated non-functional formal candidate checkpoint.")
    with validation_report.open(encoding="utf-8") as handle:
        report = json.load(handle)
    source_sha = sha256_file(checkpoint_path)
    if not isinstance(report, dict) or report.get("validation_status") != "awaiting_manual_review":
        raise ValueError("Approval requires a validation report with validation_status='awaiting_manual_review'.")
    if report.get("checkpoint_sha256") != source_sha:
        raise ValueError("Validation report checkpoint_sha256 does not match the formal candidate checkpoint.")
    with real_timing_validation.open(encoding="utf-8") as handle:
        real_report = json.load(handle)
    expected_real_report = {
        "schema": "rr_real_vps87_validation/v1",
        "training_domain_coverage_status": "PASS",
        "approval_recommendation": "eligible_for_human_review",
        "validation_status": "awaiting_manual_review",
    }
    if not isinstance(real_report, dict) or any(real_report.get(key) != value for key, value in expected_real_report.items()):
        raise ValueError("Approval requires a PASS real VPS=87 timing/fidelity validation report eligible for human review.")
    if real_report.get("checkpoint_sha256") != source_sha:
        raise ValueError("Real timing validation report checkpoint_sha256 does not match the formal candidate checkpoint.")
    approved = dict(checkpoint)
    approved.update({
        "validation_status": "approved_by_manual_review",
        "approval_review_note": review_note.strip(),
        "approval_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint_sha256": source_sha,
        "validation_report_sha256": sha256_file(validation_report),
        "real_timing_validation_report_sha256": sha256_file(real_timing_validation),
    })
    approved_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(approved, approved_output)
    return {"approved_output": str(approved_output), "validation_status": approved["validation_status"], "source_checkpoint_sha256": source_sha, "real_timing_validation_report_sha256": approved["real_timing_validation_report_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-report", required=True)
    parser.add_argument("--real-timing-validation", required=True)
    parser.add_argument("--approved-output", required=True)
    parser.add_argument("--review-note", required=True)
    args = parser.parse_args()
    print(json.dumps(approve_formal_checkpoint(args.checkpoint, args.validation_report, args.real_timing_validation, args.approved_output, args.review_note), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
