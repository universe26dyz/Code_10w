"""Build a read-only-evidence registry for the canonical five-subject cohort."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import APPROVED_DECODER_SHA256, B6_CONFIG_SOURCE_SHA256, sha256, write_immutable_json
from .sparse import _git_sha
from .standardize_subject_slices import TARGET_COUNTS, validate_standardization_manifest


SCHEMA = "phase4r_five_subject_readiness_registry/v1"
SUBJECTS = ("CYJ", "DYL_2", "DYZ", "HHZ", "HJL")
ARTIFACT_STAGES = ("proposed_reconstruction", "fit_first_t1_reconstruction", "fit_first_t2_reconstruction", "poses", "central_endpoint", "psf128_endpoint", "native_bundles", "support_masks", "metrics", "figures", "summary")


def readiness_from_evidence(subject: str, evidence: dict[str, Any]) -> dict[str, Any]:
    """Compute a fail-closed launch state from independently verified evidence."""
    standardization = evidence.get("standardization", {})
    reuse = evidence.get("dictionary_reuse", {})
    dependencies = evidence.get("frozen_dependencies", {})
    server = evidence.get("server", {})
    blockers: list[str] = []
    if not (standardization.get("status") == "STANDARDIZATION_COMPLETE" and standardization.get("valid") is True):
        blockers.append("standardization evidence is missing, stale, or invalid")
    if reuse.get("status") != "REUSE_VERIFIED":
        blockers.append("dictionary-map reuse audit is not REUSE_VERIFIED")
    if not (dependencies.get("decoder_sha256_valid") is True and dependencies.get("b6_sha256_valid") is True):
        blockers.append("frozen decoder/B6 SHA256 evidence is incomplete")
    if server.get("verified") is not True:
        blockers.append("reconstruction server availability is not verified")
    return {"subject": subject, "stage": "READY_FOR_3D_RECONSTRUCTION" if not blockers else "BLOCKED", "blockers": blockers}


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256(path) if path.is_file() else None, "status": "PRESENT" if path.is_file() else "UNKNOWN_NOT_INSPECTED"}


def _prepared_stack(path: Path) -> dict[str, Any]:
    record = _artifact(path)
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            record.update({"group_count": int(len(np.unique(data["group_idx"]))), "observation_count": int(len(data["group_idx"]))})
    return record


def _subject_manifest_paths(evidence_root: Path, subject: str, name: str) -> list[Path]:
    return sorted(path for path in evidence_root.rglob(name) if subject in path.parts)


def _standardization_evidence(evidence_root: Path, subject: str) -> dict[str, Any]:
    paths = _subject_manifest_paths(evidence_root, subject, "slice_standardization_manifest.json")
    if len(paths) != 1:
        return {"status": "UNKNOWN_NOT_INSPECTED", "valid": False, "evidence_paths": [str(path) for path in paths]}
    try:
        payload = validate_standardization_manifest(paths[0], subject=subject)
    except (OSError, ValueError, KeyError, TypeError):
        return {"status": "BLOCKED", "valid": False, "manifest": _artifact(paths[0])}
    return {"status": "STANDARDIZATION_COMPLETE", "valid": True, "manifest": _artifact(paths[0]), "stacks": {stack: {"output_count": payload["stacks"][stack]["output_count"], "output_observation": payload["stacks"][stack]["output_observation"]} for stack in TARGET_COUNTS}}


def _reuse_evidence(evidence_root: Path, subject: str, standardization: dict[str, Any]) -> dict[str, Any]:
    paths = _subject_manifest_paths(evidence_root, subject, "reuse_manifest.json")
    if len(paths) != 1:
        return {"status": "UNKNOWN_NOT_INSPECTED", "evidence_paths": [str(path) for path in paths]}
    try:
        payload = json.loads(paths[0].read_text(encoding="utf-8"))
        valid = payload.get("schema") == "phase4r_standardized_native_dictionary_reuse/v1" and payload.get("subject") == subject and payload.get("status") == "REUSE_VERIFIED"
        valid = valid and all(payload.get("stacks", {}).get(stack, {}).get("status") == "REUSE_VERIFIED" for stack in TARGET_COUNTS)
        standard = standardization.get("manifest", {})
        valid = valid and payload.get("standardization_manifest", {}).get("sha256") == standard.get("sha256")
    except (OSError, ValueError, TypeError):
        valid = False
    return {"status": "REUSE_VERIFIED" if valid else "BLOCKED", "manifest": _artifact(paths[0])}


def _dyl2_evidence(evidence_root: Path) -> dict[str, Any]:
    base = evidence_root / "pre_reconstruction" / "DYL_2"
    prepared = base / "prepared_full"
    stacks = {stack: _prepared_stack(prepared / stack / "observations.npz") for stack in TARGET_COUNTS}
    native: dict[str, Any] = {}
    for stack in TARGET_COUNTS:
        native[stack] = {name: _artifact(base / "native_dictionary_full" / stack / name) for name in ("native_maps.npz", "geometry.npz", "manifest.json")}
    handoff = {"evidence_type": "handoff_report", "source": "CODEX 13 verified handoff", "file_reverified": False}
    return {"raw_audit": {"status": "PASS", **handoff}, "protocol_admission": {"status": "ADMITTED", **handoff}, "handoff": "READY_FOR_STANDARDIZATION", "prepared": stacks, "native_dictionary_full": native, "standardization": {"status": "PROPOSED_NOT_APPLIED", "valid": False, "slice_plan": {"2ch_drop_original_group": 0, "policy": "drop highest signed-coordinate endpoint; retain 15 contiguous groups"}}, "dictionary_reuse": {"status": "UNKNOWN_NOT_INSPECTED"}, "frozen_dependencies": {"decoder_expected_sha256": APPROVED_DECODER_SHA256, "b6_expected_sha256": B6_CONFIG_SOURCE_SHA256, "decoder_sha256_valid": False, "b6_sha256_valid": False}, "server": {"verified": False, "transfer": "BLOCKED", "detail": "CMRServer04 DNS did not resolve; only rsync dry-run occurred"}}


def _subject_evidence(subject: str, evidence_root: Path) -> dict[str, Any]:
    if subject == "DYL_2":
        evidence = _dyl2_evidence(evidence_root)
    else:
        evidence = {"raw_audit": {"status": "UNKNOWN_NOT_INSPECTED"}, "protocol_admission": {"status": "UNKNOWN_NOT_INSPECTED"}, "prepared": {stack: {"status": "UNKNOWN_NOT_INSPECTED"} for stack in TARGET_COUNTS}, "native_dictionary_full": {stack: {"status": "UNKNOWN_NOT_INSPECTED"} for stack in TARGET_COUNTS}, "standardization": {"status": "UNKNOWN_NOT_INSPECTED", "valid": False}, "dictionary_reuse": {"status": "UNKNOWN_NOT_INSPECTED"}, "frozen_dependencies": {"decoder_expected_sha256": APPROVED_DECODER_SHA256, "b6_expected_sha256": B6_CONFIG_SOURCE_SHA256, "decoder_sha256_valid": False, "b6_sha256_valid": False}, "server": {"verified": False, "transfer": "UNKNOWN_NOT_INSPECTED"}}
    standardization = _standardization_evidence(evidence_root, subject)
    if standardization["status"] != "UNKNOWN_NOT_INSPECTED":
        evidence["standardization"] = standardization
    evidence["dictionary_reuse"] = _reuse_evidence(evidence_root, subject, evidence["standardization"])
    return evidence


def _manifest(subject: str, evidence: dict[str, Any]) -> dict[str, Any]:
    evaluated = readiness_from_evidence(subject, evidence)
    if subject == "DYL_2" and evidence["standardization"]["status"] == "PROPOSED_NOT_APPLIED":
        stage, blockers = "READY_FOR_STANDARDIZATION", ["standardization and dictionary-map reuse audit have not been applied", "CMRServer04 DNS blocked target-server verification"]
    else:
        stage, blockers = evaluated["stage"], evaluated["blockers"]
    return {"schema": SCHEMA, "subject": subject, "stage": stage, "raw_audit": evidence["raw_audit"], "protocol_admission": evidence["protocol_admission"], "prepared": evidence["prepared"], "native_dictionary_full": evidence["native_dictionary_full"], "standardization": evidence["standardization"], "dictionary_map_reuse": evidence["dictionary_reuse"], "frozen_dependencies": evidence["frozen_dependencies"], "transfer_server": evidence["server"], "artifacts": {stage_name: {"status": "UNKNOWN_NOT_INSPECTED"} for stage_name in ARTIFACT_STAGES}, "code_sha": _git_sha(), "inspection_time_utc": datetime.now(timezone.utc).isoformat(), "blockers": blockers, "next_action": "run read-only manifest audit" if stage == "BLOCKED" else "run approved standardization dry-run"}


def build_five_subject_registry(registry_root: str | Path, *, evidence_root: str | Path) -> Path:
    """Write a new immutable registry; never scans or modifies experiment output."""
    target, evidence_root = Path(registry_root), Path(evidence_root)
    if target.exists():
        raise FileExistsError(f"refusing overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        manifests: dict[str, Any] = {}
        for subject in SUBJECTS:
            manifest = _manifest(subject, _subject_evidence(subject, evidence_root))
            manifests[subject] = manifest
            write_immutable_json(staging / "subjects" / subject / "subject_manifest.json", manifest)
        readiness = [manifest["stage"] == "READY_FOR_3D_RECONSTRUCTION" for manifest in manifests.values()]
        registry = {"schema": SCHEMA, "subjects": list(SUBJECTS), "standardized_targets": TARGET_COUNTS, "subject_manifests": {subject: str((Path("subjects") / subject / "subject_manifest.json")) for subject in SUBJECTS}, "subject_stages": {subject: manifests[subject]["stage"] for subject in SUBJECTS}, "code_sha": _git_sha(), "all_subjects_ready_for_3d_reconstruction": all(readiness), "launch_gate": {"state": "OPEN" if all(readiness) else "CLOSED", "reasons": [f"{subject}: {manifests[subject]['stage']}" for subject in SUBJECTS if manifests[subject]["stage"] != "READY_FOR_3D_RECONSTRUCTION"]}}
        write_immutable_json(staging / "registry.json", registry)
        write_immutable_json(staging / "subject_stack_manifest_common_12_15_10_v1.json", {"schema": SCHEMA, "subjects": list(SUBJECTS), "targets": TARGET_COUNTS})
        (staging / "README.md").write_text("# Five-subject readiness registry\n\nThis registry contains references and hashes only; it does not authorize execution.\n", encoding="utf-8")
        (staging / "registry.md").write_text("# Five-subject launch gate\n\n" + "\n".join(f"- {subject}: {manifests[subject]['stage']}" for subject in SUBJECTS) + "\n", encoding="utf-8")
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-root", required=True); parser.add_argument("--evidence-root", required=True)
    args = parser.parse_args(); print(build_five_subject_registry(args.registry_root, evidence_root=args.evidence_root))


if __name__ == "__main__":
    main()
