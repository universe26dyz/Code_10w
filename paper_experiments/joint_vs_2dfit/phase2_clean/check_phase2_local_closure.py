"""Actual local authority for the Phase-2 final execution closure."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .contracts import APPROVED_DECODER_SHA256, B6_CONFIG_SOURCE_SHA256, CANONICAL_APPROVED_DECODER_DECLARED_PATH, CANONICAL_APPROVED_DECODER_RELATIVE_PATH, ELIGIBLE_REAL_SUBJECTS, HISTORICAL_NESVOR_EXECUTABLE_DEFAULT, HISTORICAL_NESVOR_VERSION, sha256

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[3]
CODE_ROOT = ROOT.parents[2]
REFERENCE_ROOT = Path("/home/universe/SVR/data/Code_10w_v1")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read(name: str) -> str:
    return (ROOT / name).read_text()


def _check_references() -> None:
    refs = json.loads(_read("final_real_reference_manifest.json"))
    inputs = json.loads(_read("phase2_input_manifest.json"))
    _require(set(refs["subjects"]) == set(ELIGIBLE_REAL_SUBJECTS), "final reference cohort is not CYJ/DYZ/HHZ/HJL")
    for subject, record in refs["subjects"].items():
        _require(record.get("subject_id") == subject, f"missing subject_id for {subject}")
        _require(record.get("protocol") == "current HHZ/TRAD", f"non-current protocol for {subject}")
        _require(record.get("dictionary_grid_provenance") and record.get("roi_migration_provenance"), f"incomplete provenance for {subject}")
        _require(record.get("sax_observation_sha256") == inputs["real_observation_sha256"][subject]["sax"]["sha256"], f"source SAX SHA drift for {subject}")
        for key in ("native_reference", "myocardium_bundle"):
            item = record[key]
            _require("relative_path" in item and not Path(item["relative_path"]).is_absolute(), f"non-relocatable {subject} {key}")
            path = REFERENCE_ROOT / item["relative_path"]
            _require(path.is_file(), f"missing local {subject} {key}: {path}")
            _require(sha256(path) == item["sha256"], f"local SHA mismatch for {subject} {key}")
            if key == "native_reference":
                native = json.loads(path.read_text())
                _require(native.get("current_protocol_source_observations", {}).get("sha256") == record["sax_observation_sha256"], f"internal native-reference SAX provenance mismatch for {subject}")


def _check_source_contracts() -> None:
    preflight = _read("preflight_actual.py")
    fit = _read("fit_first.py")
    baseline = _read("baseline.py")
    evaluator = _read("evaluation.py")
    proposed = _read("proposed.py")
    historical = json.loads(_read("historical_nesvor_v5_128_w1_contract.json"))
    approved = json.loads((CODE_ROOT / "paper_experiments/joint_vs_2dfit/audit/outputs/approved_decoder.json").read_text())
    _require("assert " not in preflight, "formal preflight contains assert safety gate")
    for token in ("sha256(p)", "reference source SAX SHA mismatch", "native-reference internal source SAX SHA mismatch", "relative_path", "B6_CONFIG_SOURCE_SHA256"):
        _require(token in preflight, f"preflight lacks actual-file gate: {token}")
    for token in ("verify_approved_decoder", "CANONICAL_DECODER_RELATIVE", "CANONICAL_APPROVED_DECODER_DECLARED_PATH", "canonical_path_plus_sha256"):
        _require(token in preflight, f"preflight lacks canonical decoder authority: {token}")
    _require(approved.get("approved_for_paper") is True and approved.get("approved_by_manual_review") is True and approved.get("validation_status") == "approved_by_manual_review", "approved decoder audit is not manually approved")
    _require(approved.get("sha256") == APPROVED_DECODER_SHA256 and approved.get("resolved_from_provenance", {}).get("declared_decoder_path") == CANONICAL_APPROVED_DECODER_DECLARED_PATH, "approved decoder audit disagrees with canonical authority")
    _require("nesvor reconstruct" not in baseline and "build_independent_nesvor_commands" in baseline, "baseline has duplicated NeSVoR command authority")
    for token in ("HISTORICAL_NESVOR_EXECUTABLE_DEFAULT", "'2500'", "'200'", "'--thicknesses','8','8','8'", "'--registration','stack'", "--scanner-space", "--no-slice-scale", "--single-precision", "'--batch-size','2000'", "'--n-iter','10000'", "'--n-samples','256'", "'--n-inference-samples','128'", "'--pixel-variance-weight','1'", "'--weight-pixel-var-smooth','0'", "'--seed','20260823'", " && python -m"):
        _require(token in fit, f"fit-first contract missing {token}")
    _require("--inference-samples" not in fit and "; python" not in fit and "--initial-transforms" not in fit, "fit-first can hide NeSVoR failure, use the wrong CLI, or reuse pose")
    _require(historical["runtime"]["executable"] == HISTORICAL_NESVOR_EXECUTABLE_DEFAULT and historical["runtime"]["version"] == HISTORICAL_NESVOR_VERSION, "historical runtime contract drifted")
    _require(historical["historical_explicit_flags"]["n_inference_samples"] == 128 and historical["historical_resolved_defaults"]["n_samples"] == 256, "historical explicit/default distinction drifted")
    _require("args.mode == 'real'" in evaluator and "'subject_level'" in evaluator and "not 3-D ground-truth accuracy" in evaluator, "real evaluator is incomplete")
    _require("1410.0" in evaluator and "50.0" in evaluator and "not_applicable_constant_reference" in evaluator, "synthetic constant-GT metrics drifted")
    for token in ("Q002S640_B6_roi_same_anchor_mse", "--anchor-batch-size','640'", "--signal-simulator"):
        _require(token in proposed, f"proposed Q002 contract missing {token}")


def _check_scripts() -> None:
    directory = ROOT / "server_commands"
    names = ("server_preflight_phase2.sh", "run_phase2_synthetic_proposed.sh", "run_phase2_synthetic_fit_first.sh", "run_phase2_real_proposed.sh", "run_phase2_real_fit_first.sh", "evaluate_phase2.sh")
    for name in names:
        path = directory / name
        text = path.read_text()
        _require("set -euo pipefail" in text, f"{name} lacks strict shell mode")
        _require("TEMPLATE ONLY" not in text and "--contract" not in text, f"{name} contains stale placeholder contract")
        subprocess.run(["bash", "-n", str(path)], check=True)
    preflight = (directory / "server_preflight_phase2.sh").read_text()
    _require("assert " not in preflight and "--reference-root \"$DATA\"" in preflight and "PHASE2_NESVOR_EXECUTABLE" in preflight and "historical_nesvor_runtime.py" in preflight and "import nesvor" not in preflight and "PHASE2_APPROVED_DECODER" in preflight and "CANONICAL_DECODER" in preflight and "CANDIDATES" not in preflight and "approved_decoder_duplicate_sha_matches" in preflight, "server preflight lacks canonical decoder authority or isolated historical runtime gate")
    for name in names[1:5]:
        text = (directory / name).read_text()
        _require("server_preflight_phase2.sh" in text and "phase2_clean.runner" not in text, f"{name} uses stale manifest self-comparison")
    for name in ("run_phase2_synthetic_proposed.sh", "run_phase2_real_proposed.sh"):
        text = (directory / name).read_text()
        _require('["approved_decoder"]["path"]' in text and "approved_decoder_path" not in text, f"{name} does not consume the actual preflight decoder record")
    for name in ("run_phase2_synthetic_fit_first.sh", "run_phase2_real_fit_first.sh"):
        _require('--nesvor-executable "$NESVOR_EXEC"' in (directory / name).read_text(), f"{name} does not consume verified historical NeSVoR")
    evaluation = (directory / "evaluate_phase2.sh").read_text()
    for token in ("PROPOSED_SYN_ROOT", "T1_3D.nii.gz", "volume_ms.nii.gz", "CYJ", "DYZ", "HHZ", "HJL", "--mode real"):
        _require(token in evaluation, f"evaluation script lacks {token}")


def _check_readiness() -> None:
    readiness = json.loads(_read("phase2_execution_readiness.json"))
    for key in ("references_ready", "synthetic_evaluator_ready", "real_evaluator_ready", "formal_scripts_ready", "local_acceptance"):
        _require(readiness.get(key) is True, f"local readiness false: {key}")
    for key in ("actual_observation_hash_gate", "approved_decoder_gate", "b6_config_source_gate", "historical_nesvor_runtime_gate", "server_environment_gate", "formal_reconstruction_ready"):
        _require(readiness.get(key) is False, f"server gate falsely marked true: {key}")
    acceptance = json.loads(_read("phase2_local_acceptance.json"))
    _require(acceptance.get("local_acceptance") == "PASS" and acceptance.get("server_acceptance") == "UNVERIFIED", "acceptance artifact not finalized")


def main() -> None:
    _require(APPROVED_DECODER_SHA256 == "f83280d4f73bfc9e0540b9606ee20bf83395339ad4e54dd0fb7be93ed9c9a7a9", "approved decoder SHA drifted")
    _require(B6_CONFIG_SOURCE_SHA256 == "15981fbe93d2c70e14755ea756a02856c9383481e7d92d239619268a628a29b5", "B6 source SHA drifted")
    _require(HISTORICAL_NESVOR_EXECUTABLE_DEFAULT == "/home/dengyz/miniconda3/envs/nesvor/bin/nesvor" and HISTORICAL_NESVOR_VERSION == "0.6.0rc1", "historical NeSVoR authority drifted")
    _require(CANONICAL_APPROVED_DECODER_RELATIVE_PATH == "Code_10w_runs/CYJ_decoder_comparison_20260924_clean_v1/inputs/mlp/signal_simulator_approved.pth", "canonical decoder relative path drifted")
    _check_references()
    _check_source_contracts()
    _check_scripts()
    _check_readiness()
    print("LOCAL_ACCEPTANCE=PASS")


if __name__ == "__main__":
    main()
