"""CPU contracts for the formal five-subject batch orchestrator."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_experiments.joint_vs_2dfit.phase2_clean import final5_batch_runner as runner


def _registry(tmp_path: Path, *, drop: tuple[str, str] | None = None) -> Path:
    records = []
    for subject in runner.SUBJECTS:
        for case_id in runner.EXPECTED_CASES:
            if (subject, case_id) == drop:
                continue
            manifest = tmp_path / "sparse" / subject / case_id / "sparse_manifest.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(json.dumps({"subject": subject, "profile_id": case_id.split("_fold")[0], "case_id": case_id}))
            records.append({"subject": subject, "case_id": case_id, "sparse_manifest": str(manifest), "sparse_manifest_sha256": runner.sha256(manifest), "generator_git_sha": "b" * 40, "prepared_root": str(manifest.parent)})
    path = tmp_path / "registry.json"; path.write_text(json.dumps({"records": records}))
    return path


def _validated(path: str | Path, *, subject: str, **_kwargs: object) -> dict[str, object]:
    raw = json.loads(Path(path).read_text())
    assert raw["subject"] == subject
    return {**raw, "prepared_root": str(Path(path).parent)}


def test_registry_admission_requires_exact_five_by_fourteen_matrix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "validate_sparse_profile_manifest", _validated)
    admitted = runner.load_final5_registry(_registry(tmp_path))
    assert len(admitted) == 70
    assert {(item.subject, item.case_id) for item in admitted} == {(subject, case_id) for subject in runner.SUBJECTS for case_id in runner.EXPECTED_CASES}
    with pytest.raises(ValueError, match="70"):
        runner.load_final5_registry(_registry(tmp_path / "missing", drop=("HJL", "L100")))


def test_plan_is_read_only_and_blocks_sparse_downstream_until_subject_l100(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "validate_sparse_profile_manifest", _validated)
    calls: list[object] = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    plan = runner.build_batch_plan(_registry(tmp_path), tmp_path / "final", device="0")
    assert calls == []
    l100 = plan["cases"]["CYJ/L100"]
    sparse = plan["cases"]["CYJ/L50_fold0"]
    assert l100["stages"]["proposed_reconstruction"]["state"] == "READY"
    assert sparse["stages"]["sparse_metrics"]["state"] == "BLOCKED_DEPENDENCY"
    assert sparse["metric_kind"] == "held-out LAX original-native metric using subject frozen L100 computational mask"


def test_existing_invalid_output_is_blocked_without_delete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "validate_sparse_profile_manifest", _validated)
    registry = _registry(tmp_path)
    invalid = tmp_path / "final/04_reconstruction/proposed/CYJ/L100"; invalid.mkdir(parents=True)
    plan = runner.build_batch_plan(registry, tmp_path / "final", device="0")
    assert plan["cases"]["CYJ/L100"]["stages"]["proposed_reconstruction"]["state"] == "BLOCKED_EXISTING_INVALID"
    assert invalid.is_dir()


def test_commands_are_serial_gpu_and_sparse_fit_uses_subject_l100_mask(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "validate_sparse_profile_manifest", _validated)
    cases = runner.load_final5_registry(_registry(tmp_path))
    sparse = next(item for item in cases if item.subject == "DYL_2" and item.case_id == "L50_fold0")
    commands = runner.case_commands(sparse, tmp_path / "final", device="0", python="python", nesvor="nesvor")
    assert commands["proposed_reconstruction"][0] == "python"
    assert commands["proposed_reconstruction"][commands["proposed_reconstruction"].index("--device") + 1] == "cuda:0"
    assert commands["fit_first_reconstruction"]["env"]["DEVICE"] == "0"
    assert "central_only" in commands["proposed_reconstruction"]
    assert set(commands) == {"proposed_reconstruction", "fit_first_reconstruction"}
    assert runner.MAX_GPU_JOBS == 1


def test_l100_dispatch_uses_subprocess_only_for_reconstructions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = runner.CaseRecord("CYJ", "L100", "L100", None, 1, tmp_path / "sparse.json", "a" * 64, "b" * 40, tmp_path)
    stages = ("proposed_reconstruction", "fit_first_reconstruction", "proposed_endpoint", "fit_first_endpoint", "proposed_native_bundle", "fit_first_native_bundle", "l100_native_support", "current_native_mask", "l100_metrics")
    monkeypatch.setattr(runner, "load_final5_registry", lambda _registry: [record])
    monkeypatch.setattr(runner, "case_commands", lambda *_args, **_kwargs: {"proposed_reconstruction": ["proposed"], "fit_first_reconstruction": {"argv": ["fit"], "env": {}}})
    monkeypatch.setattr(runner, "build_batch_plan", lambda *_args, **_kwargs: {"cases": {"CYJ/L100": {"stages": {stage: {"state": "READY"} for stage in stages}}}})
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(runner, "_run_stage", lambda stage, *_args: calls.append(("subprocess", stage)) or 0)
    monkeypatch.setattr(runner, "_run_direct_stage", lambda stage, *_args, **_kwargs: calls.append(("direct", stage)))
    runner.execute_batch(tmp_path / "registry.json", tmp_path / "final", device="0", python="python", nesvor="nesvor", subject="CYJ", case="L100")
    assert calls == [("subprocess", "proposed_reconstruction"), ("subprocess", "fit_first_reconstruction"), *(("direct", stage) for stage in stages[2:])]


def test_sparse_fit_first_endpoint_is_direct_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = runner.CaseRecord("CYJ", "L50_fold0", "L50", 0, 2, tmp_path / "sparse.json", "a" * 64, "b" * 40, tmp_path)
    stages = ("proposed_reconstruction", "fit_first_reconstruction", "proposed_endpoint", "fit_first_endpoint", "sparse_metrics")
    monkeypatch.setattr(runner, "load_final5_registry", lambda _registry: [record])
    monkeypatch.setattr(runner, "case_commands", lambda *_args, **_kwargs: {"proposed_reconstruction": ["proposed"], "fit_first_reconstruction": {"argv": ["fit"], "env": {}}})
    monkeypatch.setattr(runner, "build_batch_plan", lambda *_args, **_kwargs: {"cases": {"CYJ/L50_fold0": {"stages": {stage: {"state": "READY"} for stage in stages}}}})
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(runner, "_run_stage", lambda stage, *_args: calls.append(("subprocess", stage)) or 0)
    monkeypatch.setattr(runner, "_run_direct_stage", lambda stage, *_args, **_kwargs: calls.append(("direct", stage)))
    runner.execute_batch(tmp_path / "registry.json", tmp_path / "final", device="0", python="python", nesvor="nesvor", subject="CYJ", case="L50_fold0")
    assert ("direct", "fit_first_endpoint") in calls and ("subprocess", "fit_first_endpoint") not in calls
