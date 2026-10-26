"""CPU-only contracts for the independent CYJ1.1 r3 plan."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _observation(path: Path, groups: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        group_idx=np.repeat(np.arange(groups, dtype=np.int64), 10),
        weight_idx=np.tile(np.arange(10, dtype=np.int64), groups),
    )


def _inputs(tmp_path: Path, *, subject: str = "DYZ", source_manifests: bool = True):
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_r3_reproduction import R3Inputs

    source = tmp_path / "full_prepared" / subject
    for stack, groups in {"sax": 12, "2ch": 16, "4ch": 10}.items():
        _observation(source / stack / "observations.npz", groups)
        if source_manifests:
            (source / stack / "manifest.json").write_text("{}", encoding="utf-8")
    assets = tmp_path / "assets"
    assets.mkdir()
    b6 = assets / "b6.pt"; b6.write_bytes(b"b6")
    simulator = assets / "signal_simulator.pth"; simulator.write_bytes(b"simulator")
    nesvor = assets / "nesvor"; nesvor.write_text("#!/bin/sh\n", encoding="utf-8"); nesvor.chmod(0o755)
    historical = tmp_path / "dad9698b"; historical.mkdir()
    return R3Inputs(
        subject=subject,
        source_prepared_root=source,
        analysis_root=tmp_path / "ismrm_five_subject_analysis_v1",
        historical_worktree=historical,
        b6_model=b6,
        signal_simulator=simulator,
        nesvor_executable=nesvor,
        expected_b6_model_sha256=_sha256(b6),
        expected_signal_simulator_sha256=_sha256(simulator),
    )


def _dad_head(_path: Path) -> str:
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_r3_reproduction import DAD9698_COMMIT

    return DAD9698_COMMIT


def _step(plan: dict[str, object], step_id: str) -> dict[str, object]:
    return next(step for step in plan["steps"] if step["id"] == step_id)


def test_r3_plan_builds_the_three_case_historical_dag_without_final5_reuse(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_r3_reproduction import build_r3_plan

    inputs = _inputs(tmp_path)
    plan = build_r3_plan(inputs, git_head=_dad_head)

    assert plan["schema"] == "cyj11_r3_reproduction_plan/v1"
    assert plan["identity"] == {"subject": "DYZ", "cases": ["L100_fold0", "L50_fold0", "L25_fold0"]}
    assert not inputs.analysis_root.exists()
    assert "final5" not in str(plan).lower()
    assert _step(plan, "copy_prepared")["destination"].endswith("prepared/DYZ")
    l100_baseline = _step(plan, "fit_first_L100_fold0")["argv"]
    assert "--reuse-native-maps-root" not in l100_baseline
    assert "--execute" in l100_baseline
    for case in ("L50_fold0", "L25_fold0"):
        baseline = _step(plan, f"fit_first_{case}")["argv"]
        assert baseline[baseline.index("--reuse-native-maps-root") + 1].endswith("native_cache/DYZ")
        assert baseline[baseline.index("--sparse-manifest") + 1].endswith(f"sparse_inputs/DYZ/{case}/sparse_manifest.json")
        fit_endpoint = _step(plan, f"fit_first_central_{case}")["argv"]
        assert fit_endpoint[fit_endpoint.index("--l100-fit-first-root") + 1].endswith("reconstruction/fit_first/DYZ/L100_fold0")
        assert fit_endpoint[fit_endpoint.index("--r0-native-maps-root") + 1].endswith("native_cache/DYZ")


def test_r3_plan_marks_dyl2_missing_source_manifests_without_fabricating_them(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_r3_reproduction import build_r3_plan

    plan = build_r3_plan(_inputs(tmp_path, subject="DYL_2", source_manifests=False), git_head=_dad_head)

    assert plan["source"]["source_manifest_absent"] is True
    assert plan["source"]["manifests"] == {"sax": None, "2ch": None, "4ch": None}


def test_r3_plan_reuses_matching_prepared_input_already_in_unified_root(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_r3_reproduction import build_r3_plan

    inputs = _inputs(tmp_path)
    shutil.copytree(inputs.source_prepared_root, inputs.analysis_root / "prepared" / "DYZ")

    plan = build_r3_plan(inputs, git_head=_dad_head)

    assert _step(plan, "copy_prepared")["kind"] == "reuse_prepared"
    assert plan["source"]["prepared_destination_mode"] == "reused_after_sha256_verification"


def test_r3_plan_rejects_wrong_historical_head_and_existing_case_destination(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_r3_reproduction import build_r3_plan

    inputs = _inputs(tmp_path)
    try:
        build_r3_plan(inputs, git_head=lambda _path: "0" * 40)
    except ValueError as error:
        assert "dad9698b" in str(error)
    else:
        raise AssertionError("wrong historical worktree must be rejected")
    occupied = inputs.analysis_root / "reconstruction" / "proposed" / "DYZ" / "L50_fold0"
    occupied.mkdir(parents=True)
    try:
        build_r3_plan(inputs, git_head=_dad_head)
    except FileExistsError as error:
        assert "non-empty r3 destination" in str(error)
    else:
        raise AssertionError("occupied r3 case destination must be rejected")


def test_r3_runner_dry_run_binds_subject_without_creating_analysis_root(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/bin/sh\nif [ \"$3\" = rev-parse ]; then echo dad9698b5b2b5cd98267a6d23f1eb866b54791d3; exit 0; fi\nexit 2\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    repo = Path(__file__).resolve().parents[3]
    script = repo / "paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_cyj11_r3_reproduction.sh"
    env = {
        **os.environ,
        "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
        "REPO": str(repo),
        "CYJ11_R3_PYTHON": os.sys.executable,
        "CYJ11_R3_SUBJECT": "DYZ",
        "CYJ11_R3_SOURCE_PREPARED_ROOT": str(inputs.source_prepared_root),
        "CYJ11_R3_ANALYSIS_ROOT": str(inputs.analysis_root),
        "CYJ11_R3_HISTORICAL_WORKTREE": str(inputs.historical_worktree),
        "CYJ11_R3_B6_MODEL": str(inputs.b6_model),
        "CYJ11_R3_SIGNAL_SIMULATOR": str(inputs.signal_simulator),
        "CYJ11_R3_NESVOR_EXECUTABLE": str(inputs.nesvor_executable),
        "CYJ11_R3_EXPECTED_B6_SHA256": inputs.expected_b6_model_sha256,
        "CYJ11_R3_EXPECTED_SIGNAL_SIMULATOR_SHA256": inputs.expected_signal_simulator_sha256,
    }
    result = subprocess.run(["bash", str(script), "--dry-run"], cwd=repo, env=env, text=True, capture_output=True, check=True)
    assert '"subject": "DYZ"' in result.stdout
    assert not inputs.analysis_root.exists()


def _r2_case(root: Path, subject: str, case: str) -> None:
    for relative in (
        Path("sparse_inputs") / subject / case / "sparse_manifest.json",
        Path("proposed") / subject / case / "T1_3D.nii.gz",
        Path("fit_first") / subject / case / "input_manifest.json",
        Path("formal_endpoints/Q002S640") / subject / case / "central" / "manifest.json",
        Path("formal_endpoints/Fit-first") / subject / case / "central" / "manifest.json",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative.as_posix(), encoding="utf-8")
    (root / "reproduction_plan.json").write_text("{}", encoding="utf-8")
    (root / "commands.log").write_text("r2 command\n", encoding="utf-8")


def test_r2_import_copies_only_supported_case_artifacts_and_writes_migration(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_analysis_archive import R2_CASES, import_r2_subject

    source = tmp_path / "r2"; subject = "DYZ"
    for case in R2_CASES:
        _r2_case(source / subject / case, subject, case)
    metrics = source / subject / R2_CASES[0] / "metrics" / "must_not_copy.txt"
    metrics.parent.mkdir(parents=True); metrics.write_text("metrics", encoding="utf-8")
    analysis_root = tmp_path / "analysis"
    record = import_r2_subject(source, analysis_root, subject)

    assert record["subject"] == subject
    assert (analysis_root / "reconstruction/proposed/DYZ/SAX100_2CH0_4CH0/T1_3D.nii.gz").is_file()
    assert (analysis_root / "endpoints/central/fit_first/DYZ/S2CH100_4CH50_fold1/manifest.json").is_file()
    assert (analysis_root / "jobs/r2/DYZ/SAX100_2CH0_4CH0.reproduction_plan.json").is_file()
    assert (analysis_root / "logs/r2/DYZ.SAX100_2CH0_4CH0.commands.log").is_file()
    assert not (analysis_root / "metrics").exists()
    assert (analysis_root / "migrations/r2_import/DYZ.json").is_file()


def test_git_archive_reuses_only_the_same_immutable_commit(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_analysis_archive import archive_git_commit

    repository = Path(__file__).resolve().parents[3]
    output = tmp_path / "snapshot"
    first = archive_git_commit(repository, "HEAD", output)
    second = archive_git_commit(repository, first["commit"], output)

    assert second == first
