from __future__ import annotations

import json
from pathlib import Path
import subprocess


def _write(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _commit_repository(root: Path) -> None:
    for command in (
        ["git", "init"], ["git", "config", "user.email", "test@example.invalid"],
        ["git", "config", "user.name", "Test"], ["git", "add", "."], ["git", "commit", "-m", "runtime configuration"],
    ):
        subprocess.run(command, cwd=root, check=True, capture_output=True)


def test_materializes_generic_and_subject_specific_runtime_configuration(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.materialize_runtime_configuration import materialize_runtime_configuration

    archive, code, subject = tmp_path / "archive", tmp_path / "code", "SUBJECT_X"
    _write(archive / "99_integrity" / "archive_manifest.json", json.dumps({
        "schema_version": "scmr_v5_multimap_reproducibility_archive/v1",
        "subject_examples": [subject],
    }))
    _write(archive / "03_subject_examples" / subject / "01_inputs" / "deployment_manifest.json", "{}\n")
    _write(archive / "03_subject_examples" / subject / "01_inputs" / "sparse_inputs" / "L100" / "sparse_manifest.json", "{}\n")
    _write(archive / "03_subject_examples" / subject / "02_outputs" / "proposed" / "L100" / "config_resolved.yaml")
    _write(archive / "03_subject_examples" / subject / "02_outputs" / "proposed" / "L100" / "experiment_manifest.json", "{}\n")
    _write(archive / "03_subject_examples" / subject / "02_outputs" / "proposed" / "L100" / "run_provenance.json", "{}\n")
    for relative in (
        "mlp/configs/protocol_scmr_v5_10_100.yaml",
        "mlp/configs/recon_server_train_scmr_v5_generic.yaml",
        "configs/subject_stack_manifest.example.json",
        "paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_scmr_v5_fit_first.sh",
        "paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/paired_final_pose.sh",
        "paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/common_native_metrics_qc.sh",
        "paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/common_native_all_stack_roi_metrics.sh",
    ):
        _write(code / relative)
    _commit_repository(code)

    target = materialize_runtime_configuration(archive, code)

    assert target == archive / "02_runtime_configuration"
    assert (target / "00_common" / "protocol_scmr_v5_10_100.yaml").is_file()
    assert (target / "01_launchers" / "run_scmr_v5_fit_first.sh").is_file()
    assert (target / "02_subject_examples" / subject / "proposed" / "L100" / "run_provenance.json").is_file()
    records = (target / "runtime_configuration_manifest.jsonl").read_text(encoding="utf-8")
    assert "sha256" in records
    assert len(json.loads((target / "runtime_configuration_metadata.json").read_text(encoding="utf-8"))["code_commit"]) == 40


def test_refuses_runtime_configuration_without_required_proposed_provenance(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.materialize_runtime_configuration import materialize_runtime_configuration

    archive, code, subject = tmp_path / "archive", tmp_path / "code", "SUBJECT_X"
    _write(archive / "99_integrity" / "archive_manifest.json", json.dumps({
        "schema_version": "scmr_v5_multimap_reproducibility_archive/v1",
        "subject_examples": [subject],
    }))
    _write(archive / "03_subject_examples" / subject / "01_inputs" / "deployment_manifest.json", "{}\n")
    for relative in (
        "mlp/configs/protocol_scmr_v5_10_100.yaml",
        "mlp/configs/recon_server_train_scmr_v5_generic.yaml",
        "configs/subject_stack_manifest.example.json",
        "paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_scmr_v5_fit_first.sh",
        "paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/paired_final_pose.sh",
        "paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/common_native_metrics_qc.sh",
        "paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/common_native_all_stack_roi_metrics.sh",
    ):
        _write(code / relative)
    _commit_repository(code)

    try:
        materialize_runtime_configuration(archive, code)
    except FileNotFoundError as exc:
        assert "sparse_manifest.json" in str(exc)
    else:
        raise AssertionError("expected incomplete provenance to be refused")
