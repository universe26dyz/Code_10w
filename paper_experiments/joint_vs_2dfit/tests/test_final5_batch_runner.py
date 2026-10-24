"""CPU contracts for the formal five-subject batch orchestrator."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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
    strict_load = runner.load_final5_registry
    strict_load_calls: list[Path] = []
    monkeypatch.setattr(runner, "load_final5_registry", lambda registry: strict_load_calls.append(Path(registry)) or strict_load(registry))
    calls: list[object] = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    plan = runner.build_batch_plan(_registry(tmp_path), tmp_path / "final", device="0")
    assert calls == []
    assert len(strict_load_calls) == 1
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


def test_pre_reindex_fix_fit_first_bundle_v5_cannot_skip_valid(tmp_path: Path) -> None:
    record = runner.CaseRecord("CYJ", "L100", "L100", None, 1, tmp_path / "sparse.json", "a" * 64, "b" * 40, tmp_path)
    bundle = runner._paths(tmp_path / "final", record)["fit_bundle"]; bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text(json.dumps({"schema": "fit_first_native_prediction_bundle/v5"}))
    result = runner._validate_stage_output("fit_first_native_bundle", record, tmp_path / "final")
    assert not result.valid and result.state == "BLOCKED_EXISTING_INVALID"


def test_commands_are_serial_gpu_and_sparse_fit_uses_subject_l100_mask(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "validate_sparse_profile_manifest", _validated)
    cases = runner.load_final5_registry(_registry(tmp_path))
    sparse = next(item for item in cases if item.subject == "DYL_2" and item.case_id == "L50_fold0")
    commands = runner.case_commands(sparse, tmp_path / "final", device="0", python="python", nesvor="nesvor")
    assert commands["proposed_reconstruction"][0] == "python"
    assert commands["proposed_reconstruction"][commands["proposed_reconstruction"].index("--device") + 1] == "cuda:0"
    assert commands["fit_first_reconstruction"]["env"]["DEVICE"] == "0"
    assert commands["fit_first_reconstruction"]["argv"][1].endswith("run_scmr_one_stage_fit_first.sh")
    assert "--signal-simulator" in commands["proposed_reconstruction"]
    assert "--expected-decoder-sha256" not in commands["proposed_reconstruction"]
    for flag in ("--expected-b6-model-sha256", "--expected-protocol-sha256", "--expected-sparse-generator-git-sha", "--expected-sparse-manifest-sha256"):
        assert flag in commands["proposed_reconstruction"]
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
    completed: set[str] = set()
    monkeypatch.setattr(runner, "_validate_stage_locally", lambda stage, *_args: runner.ValidationResult(stage in completed, "SKIP_VALID" if stage in completed else "READY", "test"))
    monkeypatch.setattr(runner, "_run_stage", lambda stage, *_args: calls.append(("subprocess", stage)) or completed.add(stage) or 0)
    monkeypatch.setattr(runner, "_run_direct_stage", lambda stage, *_args, **_kwargs: calls.append(("direct", stage)) or completed.add(stage))
    runner.execute_batch(tmp_path / "registry.json", tmp_path / "final", device="0", python="python", nesvor="nesvor", subject="CYJ", case="L100")
    assert calls == [("subprocess", "proposed_reconstruction"), ("subprocess", "fit_first_reconstruction"), *(("direct", stage) for stage in stages[2:])]


def test_sparse_fit_first_endpoint_is_direct_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = runner.CaseRecord("CYJ", "L50_fold0", "L50", 0, 2, tmp_path / "sparse.json", "a" * 64, "b" * 40, tmp_path)
    stages = ("proposed_reconstruction", "fit_first_reconstruction", "proposed_endpoint", "fit_first_endpoint", "sparse_metrics")
    monkeypatch.setattr(runner, "load_final5_registry", lambda _registry: [record])
    monkeypatch.setattr(runner, "case_commands", lambda *_args, **_kwargs: {"proposed_reconstruction": ["proposed"], "fit_first_reconstruction": {"argv": ["fit"], "env": {}}})
    monkeypatch.setattr(runner, "build_batch_plan", lambda *_args, **_kwargs: {"cases": {"CYJ/L50_fold0": {"stages": {stage: {"state": "READY"} for stage in stages}}}})
    calls: list[tuple[str, str]] = []
    completed: set[str] = set()
    monkeypatch.setattr(runner, "_validate_stage_locally", lambda stage, *_args: runner.ValidationResult(stage in completed, "SKIP_VALID" if stage in completed else "READY", "test"))
    monkeypatch.setattr(runner, "_run_stage", lambda stage, *_args: calls.append(("subprocess", stage)) or completed.add(stage) or 0)
    monkeypatch.setattr(runner, "_run_direct_stage", lambda stage, *_args, **_kwargs: calls.append(("direct", stage)) or completed.add(stage))
    runner.execute_batch(tmp_path / "registry.json", tmp_path / "final", device="0", python="python", nesvor="nesvor", subject="CYJ", case="L50_fold0")
    assert ("direct", "fit_first_endpoint") in calls and ("subprocess", "fit_first_endpoint") not in calls


def test_sparse_metric_requires_l100_mask_and_both_case_endpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "validate_sparse_profile_manifest", _validated)
    monkeypatch.setattr(runner, "validate_proposed_reconstruction", lambda path, record: runner.ValidationResult(record.case_id != "L50_fold0", "SKIP_VALID" if record.case_id != "L50_fold0" else "READY", "test"))
    monkeypatch.setattr(runner, "validate_fit_first_reconstruction", lambda _path, _record: runner.ValidationResult(True, "SKIP_VALID", "test"))
    def schema(path: Path, expected: str) -> runner.ValidationResult:
        if expected == "final_sparse_native_bundle_metrics/v2":
            return runner.ValidationResult(False, "READY", "metric absent")
        if expected == "phase4r_formal_proposed_export/v1" and "L50_fold0" in str(path):
            return runner.ValidationResult(False, "READY", "endpoint absent")
        return runner.ValidationResult(True, "SKIP_VALID", "test")
    monkeypatch.setattr(runner, "_schema", schema)
    plan = runner.build_batch_plan(_registry(tmp_path), tmp_path / "final")
    assert plan["cases"]["CYJ/L50_fold0"]["stages"]["sparse_metrics"]["state"] == "BLOCKED_DEPENDENCY"


def _ready_plan(record: runner.CaseRecord, stages: tuple[str, ...]) -> dict[str, object]:
    return {"cases": {f"{record.subject}/{record.case_id}": {"stages": {stage: {"state": "READY", "reason": "absent"} for stage in stages}}}}


def test_execute_builds_full_preflight_once_and_revalidates_only_current_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    current = runner.CaseRecord("CYJ", "L100", "L100", None, 1, tmp_path / "sparse.json", "a" * 64, "b" * 40, tmp_path)
    other = runner.CaseRecord("HJL", "L100", "L100", None, 1, tmp_path / "other.json", "a" * 64, "b" * 40, tmp_path)
    stages = ("proposed_reconstruction", "fit_first_reconstruction", "proposed_endpoint", "fit_first_endpoint", "proposed_native_bundle", "fit_first_native_bundle", "l100_native_support", "current_native_mask", "l100_metrics")
    monkeypatch.setattr(runner, "load_final5_registry", lambda _registry: [current, other])
    preflight_calls: list[object] = []
    monkeypatch.setattr(runner, "build_batch_plan", lambda *_args, **_kwargs: preflight_calls.append(1) or _ready_plan(current, stages))
    monkeypatch.setattr(runner, "case_commands", lambda *_args, **_kwargs: {"proposed_reconstruction": ["proposed"], "fit_first_reconstruction": {"argv": ["fit"], "env": {}}})
    completed: set[str] = set()
    local_calls: list[tuple[str, str]] = []

    def local(stage: str, record: runner.CaseRecord, *_args: object) -> runner.ValidationResult:
        local_calls.append((stage, record.subject))
        return runner.ValidationResult(stage in completed, "SKIP_VALID" if stage in completed else "READY", "test")

    monkeypatch.setattr(runner, "_validate_stage_locally", local)
    monkeypatch.setattr(runner, "_run_stage", lambda stage, *_args: completed.add(stage) or 0)
    monkeypatch.setattr(runner, "_run_direct_stage", lambda stage, *_args, **_kwargs: completed.add(stage))
    runner.execute_batch(tmp_path / "registry.json", tmp_path / "final", device="0", python="python", nesvor="nesvor", subject="CYJ", case="L100")
    assert len(preflight_calls) == 1
    assert {subject for _stage, subject in local_calls} == {"CYJ"}
    assert len(local_calls) == 2 * len(stages)


def test_sparse_runtime_validation_reads_only_explicit_dependencies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sparse = runner.CaseRecord("CYJ", "L50_fold0", "L50", 0, 2, tmp_path / "sparse.json", "a" * 64, "b" * 40, tmp_path)
    l100 = runner.CaseRecord("CYJ", "L100", "L100", None, 1, tmp_path / "l100.json", "a" * 64, "b" * 40, tmp_path)
    calls: list[tuple[str, str]] = []

    def output(stage: str, record: runner.CaseRecord, *_args: object) -> runner.ValidationResult:
        calls.append((stage, record.case_id))
        if stage == "fit_first_endpoint":
            return runner.ValidationResult(False, "BLOCKED_EXISTING_INVALID", "stale endpoint")
        return runner.ValidationResult(True, "SKIP_VALID", "valid")

    monkeypatch.setattr(runner, "_validate_stage_output", output)
    result = runner._validate_stage_locally("sparse_metrics", sparse, tmp_path / "final", {("CYJ", "L100"): l100, ("CYJ", "L50_fold0"): sparse})
    assert result.state == "BLOCKED_DEPENDENCY"
    assert calls == [("sparse_metrics", "L50_fold0"), ("current_native_mask", "L100"), ("proposed_endpoint", "L50_fold0"), ("fit_first_endpoint", "L50_fold0")]


def test_old_fit_first_endpoint_schema_is_not_admitted(tmp_path: Path) -> None:
    endpoint = tmp_path / "endpoint"; endpoint.mkdir()
    (endpoint / "manifest.json").write_text(json.dumps({"schema": "phase4r_fixed_l100_fit_first_export_bundle/v1"}))
    record = runner.CaseRecord("CYJ", "L100", "L100", None, 1, tmp_path / "sparse.json", "a" * 64, "b" * 40, tmp_path)
    result = runner.validate_fit_first_endpoint(endpoint, record, {("CYJ", "L100"): record}, tmp_path / "final")
    assert not result.valid and result.state == "BLOCKED_EXISTING_INVALID"


def test_two_stage_fit_first_route_cannot_skip_formal_scmr_reconstruction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "two_stage"; path.mkdir()
    monkeypatch.setattr(runner, "resolve_fit_first_route", lambda _path: SimpleNamespace(route_kind="intensity_preserving_fit_first", route_schema="phase4r_intensity_preserving_fit_first/v2"))
    record = runner.CaseRecord("CYJ", "L100", "L100", None, 1, tmp_path / "sparse.json", "a" * 64, "b" * 40, tmp_path)
    result = runner.validate_fit_first_reconstruction(path, record)
    assert not result.valid and result.state == "BLOCKED_EXISTING_INVALID"


@pytest.mark.parametrize(("field", "value"), [
    ("subject", "DYL_2"), ("case_id", "L50_fold0"), ("profile_id", "L50"), ("fold_id", 0), ("fold_count", 2),
    ("sparse_manifest.path", "/wrong/sparse.json"), ("sparse_manifest.sha256", "0" * 64),
    ("sparse_manifest.generator_git_sha", "0" * 40),
])
def test_scmr_reconstruction_rejects_case_record_or_sparse_provenance_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object) -> None:
    sparse = tmp_path / "sparse.json"; sparse.write_text("{}")
    record = runner.CaseRecord("CYJ", "L100", "L100", None, 1, sparse, runner.sha256(sparse), "a" * 40, tmp_path)
    path = tmp_path / "fit"; path.mkdir()
    manifest = {"subject": record.subject, "case_id": record.case_id, "profile_id": record.profile_id, "fold_id": record.fold_id, "fold_count": record.fold_count, "sparse_manifest": {"path": str(sparse.resolve()), "sha256": record.sparse_manifest_sha256, "generator_git_sha": record.generator_git_sha}}
    if "." in field:
        parent, child = field.split("."); manifest[parent][child] = value
    else:
        manifest[field] = value
    (path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(runner, "resolve_fit_first_route", lambda _path: SimpleNamespace(route_kind="scmr_one_stage_fit_first", route_schema="phase4r_scmr_one_stage_fit_first/v1"))
    result = runner.validate_fit_first_reconstruction(path, record)
    assert not result.valid and result.state == "BLOCKED_EXISTING_INVALID"


def test_scmr_one_stage_fit_first_endpoint_is_admitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = runner.CaseRecord("CYJ", "L100", "L100", None, 1, tmp_path / "sparse.json", "a" * 64, "b" * 40, tmp_path)
    root, paths = tmp_path / "final", runner._paths(tmp_path / "final", record)
    source = paths["fit_first"]; source.mkdir(parents=True)
    models, results = {}, {}
    for parameter in ("t1", "t2"):
        model = source / f"nesvor/{parameter}/model.pt"; result = source / f"nesvor/{parameter}/result.json"
        model.parent.mkdir(parents=True, exist_ok=True); model.write_bytes(parameter.encode()); result.write_text("{}")
        models[parameter], results[parameter] = model, result
    monkeypatch.setattr(runner, "resolve_fit_first_route", lambda path: SimpleNamespace(route_kind="scmr_one_stage_fit_first", route_schema="phase4r_scmr_one_stage_fit_first/v1", models=models, results=results))
    parameters = {}
    for parameter in ("t1", "t2"):
        volume, mask = paths["fit_endpoint"] / parameter / "volume_ms.nii.gz", paths["fit_endpoint"] / parameter / "query_effective_mask.nii.gz"
        volume.parent.mkdir(parents=True, exist_ok=True); volume.write_bytes(parameter.encode()); mask.write_bytes(b"mask")
        parameters[parameter] = {"schema": "phase4r_fixed_l100_fit_first_export/v2", "subject": "CYJ", "condition": "L100", "endpoint": "central", "parameter": parameter, "output_volume_ms": {"path": str(volume.resolve()), "sha256": runner.sha256(volume)}, "effective_query_mask": {"path": str(mask.resolve()), "sha256": runner.sha256(mask)}}
        (paths["fit_endpoint"] / parameter / "manifest.json").write_text(json.dumps(parameters[parameter]))
    route = {"route_schema": "phase4r_scmr_one_stage_fit_first/v1", "source_fit_first_root": str(source.resolve()), "formal_reconstruction_model": {parameter: {"path": str(models[parameter].resolve()), "sha256": runner.sha256(models[parameter])} for parameter in models}, "parameter_results": {parameter: {"path": str(results[parameter].resolve()), "sha256": runner.sha256(results[parameter])} for parameter in results}}
    for parameter in ("t1", "t2"):
        parameters[parameter]["fit_first_routes"] = {"l100": route, "sparse": route}
        (paths["fit_endpoint"] / parameter / "manifest.json").write_text(json.dumps(parameters[parameter]))
    (paths["fit_endpoint"] / "manifest.json").write_text(json.dumps({"schema": "phase4r_fixed_l100_fit_first_export_bundle/v2", "subject": "CYJ", "condition": "L100", "method": "Fit-first", "endpoint": "central", "fit_first_routes": {"l100": route, "sparse": route}, "parameters": parameters}))
    result = runner.validate_fit_first_endpoint(paths["fit_endpoint"], record, {("CYJ", "L100"): record}, root)
    assert result.valid and result.state == "SKIP_VALID"


def test_old_l100_and_sparse_metric_schemas_are_not_admitted(tmp_path: Path) -> None:
    l100 = runner.CaseRecord("CYJ", "L100", "L100", None, 1, tmp_path / "l100.json", "a" * 64, "b" * 40, tmp_path)
    sparse = runner.CaseRecord("CYJ", "L50_fold0", "L50", 0, 2, tmp_path / "sparse.json", "a" * 64, "b" * 40, tmp_path)
    root = tmp_path / "final"
    for record, path_key, schema in ((l100, "l100_metrics", "final_pose_native_bundle_metrics/v4"), (sparse, "sparse_metrics", "final_sparse_native_bundle_metrics/v1")):
        path = runner._paths(root, record)[path_key]; path.mkdir(parents=True)
        (path / "manifest.json").write_text(json.dumps({"schema": schema}))
    assert runner.validate_l100_metric(runner._paths(root, l100)["l100_metrics"], l100, root).state == "BLOCKED_EXISTING_INVALID"
    assert runner.validate_sparse_metric(runner._paths(root, sparse)["sparse_metrics"], sparse, root).state == "BLOCKED_EXISTING_INVALID"


def test_sparse_metric_without_effective_query_mask_support_is_not_admitted(tmp_path: Path) -> None:
    record = runner.CaseRecord("CYJ", "L50_fold0", "L50", 0, 2, tmp_path / "sparse.json", "a" * 64, "b" * 40, tmp_path)
    root, path = tmp_path / "final", runner._paths(tmp_path / "final", record)["sparse_metrics"]
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps({"schema": "final_sparse_native_bundle_metrics/v2", "subject": "CYJ", "condition": "L50_fold0", "primary_support_equation": "primary_native_mask AND finite(reference)"}))
    assert runner.validate_sparse_metric(path, record, root).state == "BLOCKED_EXISTING_INVALID"


def test_execute_blocks_dependents_when_successful_subprocess_fails_artifact_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = runner.CaseRecord("CYJ", "L100", "L100", None, 1, tmp_path / "sparse.json", "a" * 64, "b" * 40, tmp_path)
    stages = ("proposed_reconstruction", "fit_first_reconstruction", "proposed_endpoint", "fit_first_endpoint", "proposed_native_bundle", "fit_first_native_bundle", "l100_native_support", "current_native_mask", "l100_metrics")
    monkeypatch.setattr(runner, "load_final5_registry", lambda _registry: [record])
    monkeypatch.setattr(runner, "build_batch_plan", lambda *_args, **_kwargs: _ready_plan(record, stages))
    monkeypatch.setattr(runner, "case_commands", lambda *_args, **_kwargs: {"proposed_reconstruction": ["proposed"], "fit_first_reconstruction": {"argv": ["fit"], "env": {}}})
    proposed_checks = 0

    def local(stage: str, *_args: object) -> runner.ValidationResult:
        nonlocal proposed_checks
        if stage == "proposed_reconstruction":
            proposed_checks += 1
            return runner.ValidationResult(False, "READY" if proposed_checks == 1 else "BLOCKED_EXISTING_INVALID", "missing output")
        if stage in ("proposed_endpoint", "proposed_native_bundle", "l100_native_support"):
            return runner.ValidationResult(False, "BLOCKED_DEPENDENCY", "proposed reconstruction invalid")
        return runner.ValidationResult(True, "SKIP_VALID", "valid")

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(runner, "_validate_stage_locally", local)
    monkeypatch.setattr(runner, "_run_stage", lambda stage, *_args: calls.append(("subprocess", stage)) or 0)
    monkeypatch.setattr(runner, "_run_direct_stage", lambda stage, *_args, **_kwargs: calls.append(("direct", stage)))
    result = runner.execute_batch(tmp_path / "registry.json", tmp_path / "final", device="0", python="python", nesvor="nesvor", subject="CYJ", case="L100")
    assert calls == [("subprocess", "proposed_reconstruction")]
    assert result["cases"]["CYJ/L100"]["stages"]["proposed_reconstruction"]["state"] == "FAILED"
    assert result["cases"]["CYJ/L100"]["stages"]["proposed_endpoint"]["state"] == "BLOCKED_DEPENDENCY"


def test_execute_resume_refuses_preexisting_invalid_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = runner.CaseRecord("CYJ", "L100", "L100", None, 1, tmp_path / "sparse.json", "a" * 64, "b" * 40, tmp_path)
    monkeypatch.setattr(runner, "load_final5_registry", lambda _registry: [record])
    monkeypatch.setattr(runner, "build_batch_plan", lambda *_args, **_kwargs: {"cases": {"CYJ/L100": {"stages": {"proposed_reconstruction": {"state": "BLOCKED_EXISTING_INVALID", "reason": "partial"}}}}})
    with pytest.raises(RuntimeError, match="preflight"):
        runner.execute_batch(tmp_path / "registry.json", tmp_path / "final", device="0", python="python", nesvor="nesvor")


def test_execute_keeps_preflight_valid_artifacts_as_skip_valid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = runner.CaseRecord("CYJ", "L100", "L100", None, 1, tmp_path / "sparse.json", "a" * 64, "b" * 40, tmp_path)
    stages = ("proposed_reconstruction", "fit_first_reconstruction", "proposed_endpoint", "fit_first_endpoint", "proposed_native_bundle", "fit_first_native_bundle", "l100_native_support", "current_native_mask", "l100_metrics")
    monkeypatch.setattr(runner, "load_final5_registry", lambda _registry: [record])
    monkeypatch.setattr(runner, "build_batch_plan", lambda *_args, **_kwargs: {"cases": {"CYJ/L100": {"stages": {stage: {"state": "SKIP_VALID", "reason": "valid"} for stage in stages}}}})
    monkeypatch.setattr(runner, "_validate_stage_locally", lambda *_args: pytest.fail("valid preflight artifact was revalidated during execute"))
    result = runner.execute_batch(tmp_path / "registry.json", tmp_path / "final", device="0", python="python", nesvor="nesvor")
    assert {value["state"] for value in result["cases"]["CYJ/L100"]["stages"].values()} == {"SKIP_VALID"}
