"""Local CPU contracts for the formal Experiment 1.2 Q004/K1 wrapper."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / "phase2_clean" / "server_commands" / "run_cyj_experiment_1_2_k1_psf_v2.sh"


def _args(tmp_path: Path) -> list[str]:
    return ["--selected-case-id", "S2CH100_4CH50_fold0", "--profile-id", "S2CH100_4CH50", "--subject-id", "CYJ", "--prepared-root", str(tmp_path / "prepared"), "--sparse-manifest", str(tmp_path / "sparse_manifest.json"), "--expected-manifest-sha256", "a" * 64, "--expected-generator-git-sha", "b" * 40, "--b6-model", str(tmp_path / "b6.json"), "--expected-b6-sha256", "c" * 64, "--decoder", str(tmp_path / "decoder.pt"), "--protocol", str(tmp_path / "protocol.yaml"), "--expected-protocol-sha256", "d" * 64, "--output", str(tmp_path / "output")]


def _fake_python(tmp_path: Path) -> Path:
    fake = tmp_path / "fake_python.sh"
    fake.write_text("#!/usr/bin/env bash\nif [[ $1 == - ]]; then exit 0; fi\noutput=\nwhile [[ $# -gt 0 ]]; do\n  if [[ $1 == --output ]]; then output=$2; shift 2; continue; fi\n  shift\ndone\nmkdir -p \"$output\"\ntouch \"$output/model.pt\" \"$output/experiment_manifest.json\" \"$output/q002_route_manifest.json\" \"$output/RESULT_SUMMARY.md\"\n", encoding="utf-8")
    fake.chmod(0o755)
    return fake


def test_k1_runner_defaults_to_dry_run_and_prints_the_fixed_formal_command(tmp_path: Path) -> None:
    result = subprocess.run(["bash", str(SCRIPT), *_args(tmp_path)], text=True, capture_output=True, env={**os.environ, "PYTHON": str(_fake_python(tmp_path))})
    assert result.returncode == 0
    assert "validated formal Q004 K1 admission" in result.stdout
    assert "--experiment-id Q004S640_K1_B6_roi_same_anchor_mse" in result.stdout
    assert "--anchor-batch-size 640" in result.stdout and "--device cuda:0" in result.stdout
    assert "--expected-sparse-manifest-sha256" in result.stdout and not (tmp_path / "output").exists()


def test_k1_runner_execute_uses_the_fixed_command_and_refuses_existing_output(tmp_path: Path) -> None:
    env = {**os.environ, "PYTHON": str(_fake_python(tmp_path))}
    result = subprocess.run(["bash", str(SCRIPT), "--execute", *_args(tmp_path)], text=True, capture_output=True, env=env)
    assert result.returncode == 0
    assert all((tmp_path / "output" / name).is_file() for name in ("model.pt", "experiment_manifest.json", "q002_route_manifest.json", "RESULT_SUMMARY.md"))
    existing = subprocess.run(["bash", str(SCRIPT), "--dry-run", *_args(tmp_path)], text=True, capture_output=True, env=env)
    assert existing.returncode != 0 and "fresh" in existing.stderr


def test_k1_runner_rejects_missing_and_unknown_arguments(tmp_path: Path) -> None:
    missing = subprocess.run(["bash", str(SCRIPT), "--dry-run"], text=True, capture_output=True)
    unknown = subprocess.run(["bash", str(SCRIPT), "--dry-run", "--unknown"], text=True, capture_output=True)
    assert missing.returncode != 0 and "missing" in missing.stderr
    assert unknown.returncode != 0 and "unknown argument" in unknown.stderr


def test_k1_runner_uses_strict_shared_admission_not_a_second_validator() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "validate_sparse_profile_manifest(manifest, subject=subject, strict=True" in source
    assert "validate_formal_sparse_reconstruction_dependencies" in source
    assert "Q004S640_K1_B6_roi_same_anchor_mse" in source
    assert "--anchor-batch-size 640" in source
