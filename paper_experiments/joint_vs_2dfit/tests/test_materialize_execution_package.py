from __future__ import annotations

import json
from pathlib import Path
import subprocess


def _write(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _git_commit(root: Path) -> None:
    for command in (
        ["git", "init"], ["git", "config", "user.email", "test@example.invalid"],
        ["git", "config", "user.name", "Test"], ["git", "add", "."], ["git", "commit", "-m", "execution package"],
    ):
        subprocess.run(command, cwd=root, check=True, capture_output=True)


def test_materializes_visible_code_models_runtime_and_pipeline_map(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.materialize_execution_package import materialize_execution_package

    archive, code, assets = tmp_path / "archive", tmp_path / "code", tmp_path / "assets"
    _write(archive / "99_integrity" / "archive_manifest.json", json.dumps({
        "schema_version": "scmr_v5_multimap_reproducibility_archive/v1", "subject_examples": ["SUBJECT_X"],
    }))
    _write(code / "trad" / "scripts" / "local_preprocess_all.m")
    _write(code / "paper_experiments" / "joint_vs_2dfit" / "phase2_clean" / "server_commands" / "run_scmr_v5_fit_first.sh")
    _git_commit(code)
    _write(assets / "decoder.pth")
    _write(assets / "b6.pt")
    _write(assets / "nesvor" / "__init__.py")
    _write(assets / "bin" / "nesvor", "#!/usr/bin/env bash\n")
    _write(assets / "cr_dreme_freeze.txt")
    _write(assets / "nesvor_freeze.txt")
    _write(assets / "local_matlab.txt")
    _write(assets / "server_matlab.txt")

    result = materialize_execution_package(
        archive_root=archive, code_root=code, decoder=assets / "decoder.pth", b6_model=assets / "b6.pt",
        nesvor_package_root=assets / "nesvor", nesvor_executable=assets / "bin" / "nesvor",
        cr_dreme_pip_freeze=assets / "cr_dreme_freeze.txt", nesvor_pip_freeze=assets / "nesvor_freeze.txt",
        local_matlab_report=assets / "local_matlab.txt", server_matlab_report=assets / "server_matlab.txt",
    )

    assert result == archive / "01_execution_code"
    assert (result / "code" / "trad" / "scripts" / "local_preprocess_all.m").is_file()
    assert (result / "01_model_assets" / "approved_decoder.pth").is_file()
    assert (result / "02_external_runtime" / "nesvor_package" / "__init__.py").is_file()
    assert (result / "00_pipeline_code_map" / "01_local_dicom_preprocessing.md").is_file()
    assert "sha256" in (result / "99_integrity" / "execution_manifest.jsonl").read_text(encoding="utf-8")
