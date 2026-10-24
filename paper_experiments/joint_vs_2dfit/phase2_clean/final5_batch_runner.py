"""Serial, resumable orchestration for the formal five-subject Phase4R batch.

This module owns operational state only.  It never deletes, repairs, or
overwrites a reconstruction artifact; published artifact validation is always
the source of truth for resume decisions.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import sha256
from .fit_first_intensity_preserving import audit_intensity_preserving_final_pose
from .fit_first_route_contract import resolve_fit_first_route
from .matched_output import formalize_proposed_export, export_fixed_l100_fit_first
from .direct_native_bundles import build_direct_native_prediction_bundle
from .fit_first_native_bundles import build_fit_first_native_prediction_bundle
from .direct_native_endpoint_audit import run_direct_native_endpoint_audit
from .current_native_central_component_mask import build_current_native_central_component_mask
from .final_pose_native_metrics import build_final_pose_native_metrics
from .final_sparse_native_metrics import build_final_sparse_native_metrics
from .sparse_profile_manifest import validate_sparse_profile_manifest

SUBJECTS = ("CYJ", "DYL_2", "DYZ", "HHZ", "HJL")
EXPECTED_CASES = (
    "L100", "L50_fold0", "L50_fold1", "L25_fold0", "L25_fold1", "L25_fold2", "L25_fold3",
    "SAX100_2CH0_4CH0", "SAX100_2CH0_4CH100", "SAX100_2CH100_4CH0",
    "S2CH50_4CH100_fold0", "S2CH50_4CH100_fold1", "S2CH100_4CH50_fold0", "S2CH100_4CH50_fold1",
)
MAX_GPU_JOBS = 1
SUBPROCESS_STAGES = ("proposed_reconstruction", "fit_first_reconstruction")
DIRECT_STAGES = ("proposed_endpoint", "fit_first_endpoint", "proposed_native_bundle", "fit_first_native_bundle", "l100_native_support", "current_native_mask", "l100_metrics", "sparse_metrics")
PROPOSED_B6 = "/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/CYJ_decoder_comparison_20260924_clean_v1/runs/frozen_mlp_B6/model.pt"
PROPOSED_B6_SHA256 = "15981fbe93d2c70e14755ea756a02856c9383481e7d92d239619268a628a29b5"
DECODER = "/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/CYJ_decoder_comparison_20260924_clean_v1/inputs/mlp/signal_simulator_approved.pth"
DECODER_SHA256 = "f83280d4f73bfc9e0540b9606ee20bf83395339ad4e54dd0fb7be93ed9c9a7a9"
PROTOCOL = "/data/dengyz/code/Code_10w/trad/configs/protocol_hhz_v1.yaml"
PROTOCOL_SHA256 = "5b6bcf41f7766b29808d664f18b6def8894696ffb4ae3a55cfcba5532cdea5c3"


@dataclass(frozen=True)
class CaseRecord:
    subject: str
    case_id: str
    profile_id: str
    fold_id: int | None
    fold_count: int | None
    sparse_manifest: Path
    sparse_manifest_sha256: str
    generator_git_sha: str
    prepared_root: Path


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    state: str
    reason: str


@dataclass(frozen=True)
class DeviceSpec:
    gpu_index: str
    torch_device: str
    nesvor_device: str


def normalize_device(value: str) -> DeviceSpec:
    value = str(value)
    index = value[5:] if value.startswith("cuda:") else value
    if not index.isdigit():
        raise ValueError("formal final5 batch device must be GPU index N or cuda:N")
    return DeviceSpec(index, f"cuda:{index}", index)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _paths(root: Path, record: CaseRecord) -> dict[str, Path]:
    subject, case = record.subject, record.case_id
    return {
        "proposed": root / "04_reconstruction" / "proposed" / subject / case,
        "fit_first": root / "04_reconstruction" / "fit_first_intensity_preserving" / subject / case,
        "proposed_endpoint": root / "05_formal_endpoints" / "central" / "proposed" / subject / case,
        "fit_endpoint": root / "05_formal_endpoints" / "central" / "fit_first" / subject / case,
        "direct_bundle": root / "05_formal_endpoints" / "central" / "proposed_native_bundle" / subject / "L100",
        "fit_bundle": root / "05_formal_endpoints" / "central" / "fit_first_native_bundle" / subject / "L100",
        "native_support": root / "06_metrics" / "native_support" / subject / "L100" / "direct_native_endpoint_audit",
        "native_mask": root / "06_metrics" / "current_native_mask" / subject / "L100",
        "l100_metrics": root / "06_metrics" / "final_pose_native" / subject / "L100",
        "sparse_metrics": root / "06_metrics" / "final_sparse_native" / subject / case,
    }


def load_final5_registry(registry: str | Path) -> list[CaseRecord]:
    """Read the sole case source and refuse any matrix other than frozen 5×14."""
    payload = _read(Path(registry))
    rows = payload.get("records")
    if not isinstance(rows, list):
        raise ValueError("sparse_case_registry.json requires a records list")
    result: list[CaseRecord] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("registry record must be an object")
        try:
            subject, case_id = row["subject"], row["case_id"]
            manifest, manifest_sha, generator_sha, prepared = row["sparse_manifest"], row["sparse_manifest_sha256"], row["generator_git_sha"], row["prepared_root"]
        except KeyError as exc:
            raise ValueError(f"registry record missing {exc.args[0]}") from exc
        if not isinstance(subject, str) or not isinstance(case_id, str) or (subject, case_id) in seen:
            raise ValueError("registry has invalid or duplicate subject/case identity")
        if sha256(Path(manifest)) != manifest_sha:
            raise ValueError(f"{subject}/{case_id}: registry sparse manifest SHA256 mismatch")
        sparse = validate_sparse_profile_manifest(Path(manifest), subject=subject, strict=True, expected_generator_git_sha=generator_sha, expected_manifest_sha256=manifest_sha)
        if sparse.get("case_id") != case_id or Path(str(sparse.get("prepared_root"))).resolve() != Path(prepared).resolve():
            raise ValueError(f"{subject}/{case_id}: registry identity/prepared_root disagrees with sparse manifest")
        seen.add((subject, case_id))
        result.append(CaseRecord(subject, case_id, str(sparse.get("profile_id")), sparse.get("fold_id"), sparse.get("fold_count"), Path(manifest), manifest_sha, generator_sha, Path(prepared)))
    expected = {(subject, case) for subject in SUBJECTS for case in EXPECTED_CASES}
    if seen != expected:
        raise ValueError(f"formal registry must contain exactly 70 frozen subject-cases; got {len(seen)}")
    return sorted(result, key=lambda item: (SUBJECTS.index(item.subject), EXPECTED_CASES.index(item.case_id)))


def _invalid(path: Path, reason: str) -> ValidationResult:
    return ValidationResult(False, "BLOCKED_EXISTING_INVALID" if path.exists() else "READY", reason)


def validate_proposed_reconstruction(path: Path, record: CaseRecord) -> ValidationResult:
    if not path.exists():
        return _invalid(path, "output absent")
    required = (path / "model.pt", path / "T1_3D.nii.gz", path / "T2_3D.nii.gz", path / "support_mask_3D.nii.gz")
    manifest = path / "experiment_manifest.json"
    if not all(item.is_file() for item in required) or not manifest.is_file():
        return _invalid(path, "Proposed output is incomplete")
    data = _read(manifest)
    samples = data.get("training_psf_samples", data.get("route", {}).get("sampling", {}).get("shared_training_psf_samples"))
    if data.get("experiment_id") != "Q002S640_B6_roi_same_anchor_mse" or int(samples or -1) != 8:
        return _invalid(path, "Proposed route is not Q002S640/K8")
    if data.get("subject", record.subject) != record.subject or data.get("case_id", data.get("condition", record.case_id)) not in (record.case_id, "L100_fold0" if record.case_id == "L100" else record.case_id):
        return _invalid(path, "Proposed subject/case identity mismatch")
    if data.get("endpoint_policy", "central_only") != "central_only":
        return _invalid(path, "Proposed endpoint policy is not central_only")
    return ValidationResult(True, "SKIP_VALID", "valid Q002S640/K8 central-only reconstruction")


def _fit_stage2_complete(path: Path) -> bool:
    return all((path / "stage2_reconstruction" / parameter / "model.pt").is_file() for parameter in ("t1", "t2")) and (path / "registered_quantitative_inputs_gate.json").is_file()


def validate_fit_first_reconstruction(path: Path, record: CaseRecord) -> ValidationResult:
    if not path.exists():
        return _invalid(path, "output absent")
    if _fit_stage2_complete(path) and not (path / "final_pose_audit.json").is_file():
        return ValidationResult(False, "AUDIT_ONLY", "Stage-2 is complete but final_pose_audit.json is absent")
    try:
        route = resolve_fit_first_route(path)
    except (FileNotFoundError, ValueError) as exc:
        return _invalid(path, str(exc))
    if route.route_kind != "intensity_preserving_fit_first":
        return _invalid(path, "historical Fit-first route is not a formal corrected baseline")
    return ValidationResult(True, "SKIP_VALID", "valid intensity-preserving Stage-2 Fit-first reconstruction")


def _schema(path: Path, expected: str) -> ValidationResult:
    if not path.exists():
        return _invalid(path, "output absent")
    try:
        if _read(path / "manifest.json").get("schema") != expected:
            return _invalid(path, f"expected {expected}")
    except (FileNotFoundError, ValueError) as exc:
        return _invalid(path, str(exc))
    return ValidationResult(True, "SKIP_VALID", f"valid {expected}")


def _stage(result: ValidationResult, *, dependency: bool = False) -> dict[str, str]:
    return {"state": "BLOCKED_DEPENDENCY" if dependency and result.state == "READY" else result.state, "reason": result.reason}


def build_batch_plan(registry: str | Path, final_root: str | Path, *, device: str = "0", records: list[CaseRecord] | None = None) -> dict[str, Any]:
    """Read-only artifact truth for plan, status, validation, and resume."""
    records, root = (load_final5_registry(registry) if records is None else list(records)), Path(final_root)
    cases: dict[str, Any] = {}
    l100_ready: dict[str, bool] = {}
    for record in records:
        if record.case_id != "L100":
            continue
        paths = _paths(root, record)
        proposed, fit = validate_proposed_reconstruction(paths["proposed"], record), validate_fit_first_reconstruction(paths["fit_first"], record)
        endpoints = _schema(paths["proposed_endpoint"], "phase4r_formal_proposed_export/v1"), _schema(paths["fit_endpoint"], "phase4r_fixed_l100_fit_first_export_bundle/v1")
        bundles = _schema(paths["direct_bundle"], "direct_native_prediction_bundle/v1"), _schema(paths["fit_bundle"], "fit_first_native_prediction_bundle/v4")
        support, mask, metric = _schema(paths["native_support"], "q004_direct_native_endpoint_audit/v2"), _schema(paths["native_mask"], "phase4r_current_native_central_component_mask/v1"), _schema(paths["l100_metrics"], "final_pose_native_bundle_metrics/v4")
        stages = {"proposed_reconstruction": _stage(proposed), "fit_first_reconstruction": _stage(fit), "proposed_endpoint": _stage(endpoints[0], dependency=not proposed.valid), "fit_first_endpoint": _stage(endpoints[1], dependency=not fit.valid), "proposed_native_bundle": _stage(bundles[0], dependency=not proposed.valid), "fit_first_native_bundle": _stage(bundles[1], dependency=not (fit.valid and endpoints[1].valid)), "l100_native_support": _stage(support, dependency=not (proposed.valid and endpoints[0].valid)), "current_native_mask": _stage(mask, dependency=not support.valid), "l100_metrics": _stage(metric, dependency=not (bundles[0].valid and bundles[1].valid and mask.valid))}
        l100_ready[record.subject] = all(value["state"] == "SKIP_VALID" for value in stages.values())
        cases[f"{record.subject}/{record.case_id}"] = {"profile_id": record.profile_id, "fold_id": record.fold_id, "metric_kind": "computational current-native common-support metric", "stages": stages}
    for record in records:
        if record.case_id == "L100":
            continue
        paths = _paths(root, record)
        proposed, fit = validate_proposed_reconstruction(paths["proposed"], record), validate_fit_first_reconstruction(paths["fit_first"], record)
        metric = _schema(paths["sparse_metrics"], "final_sparse_native_bundle_metrics/v1")
        proposed_endpoint = _schema(paths["proposed_endpoint"], "phase4r_formal_proposed_export/v1")
        fit_endpoint = _schema(paths["fit_endpoint"], "phase4r_fixed_l100_fit_first_export_bundle/v1")
        l100_record = next(item for item in records if item.subject == record.subject and item.case_id == "L100")
        l100_mask_ready = _schema(_paths(root, l100_record)["native_mask"], "phase4r_current_native_central_component_mask/v1").valid
        dependency = not (l100_mask_ready and proposed_endpoint.valid and fit_endpoint.valid)
        cases[f"{record.subject}/{record.case_id}"] = {"profile_id": record.profile_id, "fold_id": record.fold_id, "metric_kind": "held-out LAX original-native metric using subject frozen L100 computational mask", "stages": {"proposed_reconstruction": _stage(proposed), "fit_first_reconstruction": _stage(fit), "proposed_endpoint": _stage(proposed_endpoint, dependency=not proposed.valid), "fit_first_endpoint": _stage(fit_endpoint, dependency=not fit.valid), "sparse_metrics": _stage(metric, dependency=dependency)}}
    return {"schema": "phase4r_final5_batch_plan/v1", "registry": str(Path(registry).resolve()), "final_root": str(root.resolve()), "max_gpu_jobs": MAX_GPU_JOBS, "cases": cases, "device": str(device), "computational_metrics_only": True}


def _case_stages(record: CaseRecord) -> tuple[str, ...]:
    return ("proposed_reconstruction", "fit_first_reconstruction", "proposed_endpoint", "fit_first_endpoint", "proposed_native_bundle", "fit_first_native_bundle", "l100_native_support", "current_native_mask", "l100_metrics") if record.case_id == "L100" else ("proposed_reconstruction", "fit_first_reconstruction", "proposed_endpoint", "fit_first_endpoint", "sparse_metrics")


def _validate_stage_output(stage: str, record: CaseRecord, root: Path) -> ValidationResult:
    """Validate one published artifact without rescanning the frozen registry."""
    paths = _paths(root, record)
    if stage == "proposed_reconstruction":
        return validate_proposed_reconstruction(paths["proposed"], record)
    if stage == "fit_first_reconstruction":
        return validate_fit_first_reconstruction(paths["fit_first"], record)
    expected = {
        "proposed_endpoint": ("proposed_endpoint", "phase4r_formal_proposed_export/v1"),
        "fit_first_endpoint": ("fit_endpoint", "phase4r_fixed_l100_fit_first_export_bundle/v1"),
        "proposed_native_bundle": ("direct_bundle", "direct_native_prediction_bundle/v1"),
        "fit_first_native_bundle": ("fit_bundle", "fit_first_native_prediction_bundle/v4"),
        "l100_native_support": ("native_support", "q004_direct_native_endpoint_audit/v2"),
        "current_native_mask": ("native_mask", "phase4r_current_native_central_component_mask/v1"),
        "l100_metrics": ("l100_metrics", "final_pose_native_bundle_metrics/v4"),
        "sparse_metrics": ("sparse_metrics", "final_sparse_native_bundle_metrics/v1"),
    }
    try:
        path_key, schema = expected[stage]
    except KeyError as exc:
        raise RuntimeError(f"unknown final5 stage: {stage}") from exc
    return _schema(paths[path_key], schema)


def _stage_dependencies(stage: str, record: CaseRecord, records: dict[tuple[str, str], CaseRecord]) -> tuple[tuple[str, CaseRecord], ...]:
    same = lambda required: (required, record)
    if stage == "proposed_endpoint":
        return (same("proposed_reconstruction"),)
    if stage == "fit_first_endpoint":
        dependencies = [same("fit_first_reconstruction")]
        if record.case_id != "L100":
            dependencies.append(("fit_first_reconstruction", records[(record.subject, "L100")]))
        return tuple(dependencies)
    if stage == "proposed_native_bundle":
        return (same("proposed_reconstruction"),)
    if stage == "fit_first_native_bundle":
        return (same("fit_first_reconstruction"), same("fit_first_endpoint"))
    if stage == "l100_native_support":
        return (same("proposed_reconstruction"), same("proposed_endpoint"))
    if stage == "current_native_mask":
        return (same("l100_native_support"),)
    if stage == "l100_metrics":
        return (same("proposed_native_bundle"), same("fit_first_native_bundle"), same("current_native_mask"))
    if stage == "sparse_metrics":
        return (("current_native_mask", records[(record.subject, "L100")]), same("proposed_endpoint"), same("fit_first_endpoint"))
    return ()


def _validate_stage_locally(stage: str, record: CaseRecord, root: Path, records: dict[tuple[str, str], CaseRecord]) -> ValidationResult:
    """Read only the current output and its explicit, published dependencies."""
    result = _validate_stage_output(stage, record, root)
    if result.valid or result.state == "BLOCKED_EXISTING_INVALID":
        return result
    for dependency_stage, dependency_record in _stage_dependencies(stage, record, records):
        dependency = _validate_stage_output(dependency_stage, dependency_record, root)
        if not dependency.valid:
            return ValidationResult(False, "BLOCKED_DEPENDENCY", f"{dependency_record.subject}/{dependency_record.case_id} {dependency_stage}: {dependency.reason}")
    return result


def case_commands(record: CaseRecord, final_root: str | Path, *, device: str, python: str, nesvor: str) -> dict[str, Any]:
    root, paths = Path(final_root), _paths(Path(final_root), record)
    devices = normalize_device(device)
    proposed = [python, "-m", "quality_experiments.Q002_B6_roi_same_anchor_mse.run_q002_reconstruction", "--cropped-prepared-root", str(record.prepared_root), "--subject-id", record.subject, "--b6-model", PROPOSED_B6, "--expected-b6-model-sha256", PROPOSED_B6_SHA256, "--signal-simulator", DECODER, "--protocol", PROTOCOL, "--expected-protocol-sha256", PROTOCOL_SHA256, "--sparse-manifest", str(record.sparse_manifest), "--expected-sparse-generator-git-sha", record.generator_git_sha, "--expected-sparse-manifest-sha256", record.sparse_manifest_sha256, "--endpoint-policy", "central_only", "--output", str(paths["proposed"]), "--device", devices.torch_device, "--experiment-id", "Q002S640_B6_roi_same_anchor_mse", "--anchor-batch-size", "640"]
    fit_env = {"SUBJECT": record.subject, "PREPARED_ROOT": str(record.prepared_root), "STANDARDIZED_NATIVE_DICTIONARY_ROOT": str(root / "02_standardized_12_15_10" / "native_dictionary" / record.subject), "STANDARDIZED_REUSE_MANIFEST": str(root / "02_standardized_12_15_10" / "native_dictionary" / record.subject / "reuse_manifest.json"), "SPARSE_MANIFEST": str(record.sparse_manifest), "EXPECTED_GENERATOR_GIT_SHA": record.generator_git_sha, "EXPECTED_MANIFEST_SHA256": record.sparse_manifest_sha256, "NESVOR": nesvor, "OUTPUT": str(paths["fit_first"]), "DEVICE": devices.nesvor_device, "PYTHON": python}
    return {"proposed_reconstruction": proposed, "fit_first_reconstruction": {"argv": ["bash", str(Path(__file__).with_name("server_commands") / "run_intensity_preserving_fit_first.sh")], "env": fit_env}}


def _state_root(final_root: Path) -> Path:
    return final_root / "00_inventory" / "final5_batch_v1"


def _write_state(final_root: Path, plan: dict[str, Any], *, name: str) -> Path:
    root = _state_root(final_root); root.mkdir(parents=True, exist_ok=True)
    target = root / name
    target.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def _run_stage(stage: str, record: CaseRecord, command: Any, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    if stage == "fit_first_reconstruction":
        argv, env = command["argv"], {**os.environ, **command["env"]}
    else:
        argv, env = command, None
    started = _utc()
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(argv, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    return result.returncode


def _run_direct_stage(stage: str, record: CaseRecord, root: Path, *, device: str, nesvor: str) -> None:
    """Invoke existing immutable downstream builders; no scientific arguments live here."""
    paths = _paths(root, record)
    native = root / "02_standardized_12_15_10" / "native_dictionary" / record.subject
    if stage == "proposed_endpoint":
        formalize_proposed_export(subject=record.subject, condition=record.case_id, source_root=paths["proposed"], endpoint="central", output=paths["proposed_endpoint"])
    elif stage == "fit_first_endpoint":
        l100_fit = _paths(root, CaseRecord(record.subject, "L100", "L100", None, 1, record.sparse_manifest, record.sparse_manifest_sha256, record.generator_git_sha, record.prepared_root))["fit_first"]
        export_fixed_l100_fit_first(subject=record.subject, condition=record.case_id, l100_fit_first_root=l100_fit, sparse_fit_first_root=paths["fit_first"], sparse_manifest=record.sparse_manifest, r0_native_maps_root=native, nesvor_executable=nesvor, endpoint="central", output=paths["fit_endpoint"], device=normalize_device(device).nesvor_device)
    elif stage == "proposed_native_bundle":
        build_direct_native_prediction_bundle(checkpoint=paths["proposed"] / "model.pt", prepared_root=record.prepared_root, endpoint="central", output=paths["direct_bundle"], device=normalize_device(device).torch_device)
    elif stage == "fit_first_native_bundle":
        build_fit_first_native_prediction_bundle(baseline_root=paths["fit_first"], r0_native_maps_root=native, endpoint="central", endpoint_source_root=paths["fit_endpoint"], output=paths["fit_bundle"])
    elif stage == "l100_native_support":
        run_direct_native_endpoint_audit(checkpoint=paths["proposed"] / "model.pt", prepared_root=record.prepared_root, r0_native_maps_root=native, sparse_manifest=record.sparse_manifest, central_t1=paths["proposed_endpoint"] / "T1_3D.nii.gz", central_t2=paths["proposed_endpoint"] / "T2_3D.nii.gz", l100_t1=paths["proposed"] / "support_mask_3D.nii.gz", l100_t2=paths["proposed"] / "support_mask_3D.nii.gz", output=paths["native_support"], device=normalize_device(device).torch_device)
    elif stage == "current_native_mask":
        build_current_native_central_component_mask(r0_native_maps_root=native, sparse_manifest=record.sparse_manifest, l100_native_support_root=paths["native_support"], output=paths["native_mask"])
    elif stage == "l100_metrics":
        build_final_pose_native_metrics(direct_bundle_root=paths["direct_bundle"], fit_first_bundle_root=paths["fit_bundle"], r0_native_maps_root=native, sparse_manifest=record.sparse_manifest, mask_root=paths["native_mask"], output=paths["l100_metrics"])
    elif stage == "sparse_metrics":
        build_final_sparse_native_metrics(proposed_endpoint_root=paths["proposed_endpoint"], fit_first_endpoint_root=paths["fit_endpoint"], native_mask_root=paths["native_mask"], r0_native_maps_root=native, sparse_manifest=record.sparse_manifest, output=paths["sparse_metrics"])


def execute_batch(registry: str | Path, final_root: str | Path, *, device: str, python: str, nesvor: str, subject: str | None = None, case: str | None = None) -> dict[str, Any]:
    """Run serially after one strict preflight; never delete partial roots."""
    all_records = load_final5_registry(registry)
    root = Path(final_root)
    result = deepcopy(build_batch_plan(registry, root, device=device, records=all_records))
    invalid = [f"{key} {stage}" for key, item in result["cases"].items() for stage, value in item["stages"].items() if value["state"] == "BLOCKED_EXISTING_INVALID"]
    if invalid:
        _write_state(root, result, name="status.json")
        raise RuntimeError(f"execute preflight found BLOCKED_EXISTING_INVALID: {', '.join(invalid)}")
    records = all_records
    if subject is not None:
        records = [item for item in records if item.subject == subject]
    if case is not None:
        records = [item for item in records if item.case_id == case]
    records_by_key = {(item.subject, item.case_id): item for item in all_records}
    events = _state_root(root) / "events.jsonl"; events.parent.mkdir(parents=True, exist_ok=True)
    try:
        for record in records:
            commands = case_commands(record, root, device=device, python=python, nesvor=nesvor)
            case_plan = result["cases"][f"{record.subject}/{record.case_id}"]
            for stage in _case_stages(record):
                stage_plan = case_plan["stages"][stage]
                if stage_plan["state"] == "SKIP_VALID":
                    continue
                local = _validate_stage_locally(stage, record, root, records_by_key)
                stage_plan.update(_stage(local))
                if local.valid:
                    continue
                log = _state_root(root) / "logs" / record.subject / record.case_id / f"{stage}.log"
                if local.state == "AUDIT_ONLY" and stage == "fit_first_reconstruction":
                    try:
                        audit_intensity_preserving_final_pose(root=_paths(root, record)["fit_first"])
                        code = 0
                    except Exception as exc:
                        log.parent.mkdir(parents=True, exist_ok=True); log.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
                        code = 1
                elif local.state != "READY":
                    continue
                elif stage in SUBPROCESS_STAGES:
                    code = _run_stage(stage, record, commands[stage], log)
                elif stage in DIRECT_STAGES:
                    try:
                        _run_direct_stage(stage, record, root, device=device, nesvor=nesvor)
                        code = 0
                    except Exception as exc:
                        log.parent.mkdir(parents=True, exist_ok=True); log.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
                        code = 1
                else:
                    raise RuntimeError(f"unknown final5 stage: {stage}")
                if code == 0:
                    post = _validate_stage_locally(stage, record, root, records_by_key)
                    if post.valid:
                        stage_plan.update(_stage(post))
                    else:
                        stage_plan.update({"state": "FAILED", "reason": f"post-stage validation failed: {post.reason}"})
                else:
                    stage_plan.update({"state": "FAILED", "reason": f"stage returned {code}"})
                with events.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"stage": stage, "subject": record.subject, "case_id": record.case_id, "return_code": code, "validation_state": stage_plan["state"], "log_path": str(log), "finished_utc": _utc()}) + "\n")
    except KeyboardInterrupt:
        _write_state(root, result, name="status.json")
        raise
    _write_state(root, result, name="status.json")
    _write_state(root, result, name="summary.json")
    return result


def _print_status(plan: dict[str, Any]) -> None:
    print("SUBJECT CASE                         PROP FIT EP-P EP-F METRIC STATUS")
    counts: dict[str, int] = {}
    for key, item in plan["cases"].items():
        subject, case = key.split("/", 1); stages = item["stages"]
        values = list(stages.values()); state = "COMPLETE" if values and all(value["state"] == "SKIP_VALID" for value in values) else "BLOCKED" if any(value["state"].startswith("BLOCKED") for value in values) else "READY"
        counts[state] = counts.get(state, 0) + 1
        print(f"{subject:<7} {case:<28} {stages.get('proposed_reconstruction', {}).get('state', '-'): <4} {stages.get('fit_first_reconstruction', {}).get('state', '-'): <4} {stages.get('proposed_endpoint', {}).get('state', '-'): <4} {stages.get('fit_first_endpoint', {}).get('state', '-'): <4} {stages.get('l100_metrics', stages.get('sparse_metrics', {})).get('state', '-'): <6} {state}")
    print(f"TOTAL {len(plan['cases'])}")
    for label in ("COMPLETE", "READY", "FAILED", "BLOCKED"):
        print(f"{label} {counts.get(label, 0)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "execute", "status", "validate"))
    parser.add_argument("--registry", required=True); parser.add_argument("--final-root", required=True)
    parser.add_argument("--device", default="0"); parser.add_argument("--python", default=sys.executable); parser.add_argument("--nesvor", default="nesvor")
    parser.add_argument("--subject"); parser.add_argument("--case")
    args = parser.parse_args()
    if args.command == "execute":
        result = execute_batch(args.registry, args.final_root, device=args.device, python=args.python, nesvor=args.nesvor, subject=args.subject, case=args.case)
    else:
        result = build_batch_plan(args.registry, args.final_root, device=args.device)
        _write_state(Path(args.final_root), result, name="plan.json" if args.command == "plan" else "status.json")
    _print_status(result)


if __name__ == "__main__":
    main()
