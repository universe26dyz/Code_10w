"""CPU-only contracts for the independent CYJ1.1 r3 plan."""
from __future__ import annotations

import hashlib
from pathlib import Path

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
