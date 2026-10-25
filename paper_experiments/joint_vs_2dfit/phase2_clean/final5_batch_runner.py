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
from .fit_first_route_contract import EXP1_1_SCHEMA, resolve_fit_first_route
from .matched_output import formalize_proposed_export, export_fixed_l100_fit_first
from .direct_native_bundles import build_direct_native_prediction_bundle
from .fit_first_native_bundles import build_fit_first_native_prediction_bundle
from .direct_native_endpoint_audit import run_direct_native_endpoint_audit
from .current_native_central_component_mask import build_current_native_central_component_mask
from .final_pose_native_metrics import build_final_pose_native_metrics
from .final_sparse_native_metrics import build_final_sparse_native_metrics
from .final_native_grid_metrics import build_final_native_grid_metrics
from .sparse_profile_manifest import validate_sparse_profile_manifest

SUBJECTS = ("CYJ", "DYL_2", "DYZ", "HHZ", "HJL")
EXPECTED_CASES = (
    "L100", "L50_fold0", "L50_fold1", "L25_fold0", "L25_fold1", "L25_fold2", "L25_fold3",
    "SAX100_2CH0_4CH0", "SAX100_2CH0_4CH100", "SAX100_2CH100_4CH0",
    "S2CH50_4CH100_fold0", "S2CH50_4CH100_fold1", "S2CH100_4CH50_fold0", "S2CH100_4CH50_fold1",
)
MAX_GPU_JOBS = 1
FIT_FIRST_ENDPOINT_SCHEMA = "phase4r_fixed_l100_fit_first_export_bundle/v2"
FIT_FIRST_PARAMETER_ENDPOINT_SCHEMA = "phase4r_fixed_l100_fit_first_export/v2"
L100_METRIC_SCHEMA = "final_pose_native_bundle_metrics/v5"
SPARSE_METRIC_SCHEMA = "final_sparse_native_bundle_metrics/v2"
FINAL_NATIVE_GRID_METRIC_SCHEMA = "final5_native_grid_metric_bundle/v2"
FIT_FIRST_COMMON_SUPPORT_TERM = "fit_first_query_effective_mask"
FIT_FIRST_NATIVE_EFFECTIVE_MASK_CONTRACT = "formal endpoint effective query mask SHA-verified and mapped to each native plane with nearest-neighbour order=0; valid = volume_geometry_valid AND query_effective_mask AND finite(prediction)"
SUBPROCESS_STAGES = ("proposed_reconstruction", "fit_first_reconstruction")
DIRECT_STAGES = ("proposed_endpoint", "fit_first_endpoint", "proposed_native_bundle", "fit_first_native_bundle", "l100_native_support", "current_native_mask", "l100_metrics", "sparse_metrics", "final_native_grid_metrics")
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
        "fit_first": root / "04_reconstruction" / "fit_first_exp1_1" / subject / case,
        "proposed_endpoint": root / "05_formal_endpoints" / "central" / "proposed" / subject / case,
        "fit_endpoint": root / "05_formal_endpoints" / "central" / "fit_first" / subject / case,
        "direct_bundle": root / "05_formal_endpoints" / "central" / "proposed_native_bundle" / subject / "L100",
        "fit_bundle": root / "05_formal_endpoints" / "central" / "fit_first_native_bundle" / subject / "L100",
        "native_support": root / "06_metrics" / "native_support" / subject / "L100" / "direct_native_endpoint_audit",
        "native_mask": root / "06_metrics" / "current_native_mask" / subject / "L100",
        "l100_metrics": root / "06_metrics" / "final_pose_native" / subject / "L100",
        "sparse_metrics": root / "06_metrics" / "final_sparse_native" / subject / case,
        "final_native_grid_metrics": root / "06_metrics" / "final_native_grid" / subject / case,
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
    route = data.get("route") if isinstance(data.get("route"), dict) else {}
    sampling = route.get("sampling") if isinstance(route.get("sampling"), dict) else {}
    samples = data.get("training_psf_samples")
    if (
        data.get("experiment_id") != "Q002S640_B6_roi_same_anchor_mse"
        or sampling.get("anchor_batch_size") != 640
        or samples != 8
        or sampling.get("shared_training_psf_samples") != 8
    ):
        return _invalid(path, "Proposed route is not Q002S640/K8")
    if data.get("subject_id") != record.subject or ("subject" in data and data["subject"] != record.subject):
        return _invalid(path, "Proposed subject/case identity mismatch")
    if data.get("profile_id") != record.profile_id or data.get("case_id") != record.case_id or ("fold_id" in data and data["fold_id"] != record.fold_id):
        return _invalid(path, "Proposed subject/case identity mismatch")
    sparse = data.get("sparse_manifest")
    if not isinstance(sparse, dict) or Path(str(sparse.get("path", ""))).resolve() != record.sparse_manifest.resolve() or sparse.get("sha256") != record.sparse_manifest_sha256 or sparse.get("generator_git_sha") != record.generator_git_sha:
        return _invalid(path, "Proposed sparse-manifest provenance mismatch")
    dependencies = data.get("dependency_sha256")
    if not isinstance(dependencies, dict) or dependencies.get("B6_model") != PROPOSED_B6_SHA256 or dependencies.get("frozen_decoder") != DECODER_SHA256 or dependencies.get("protocol") != PROTOCOL_SHA256:
        return _invalid(path, "Proposed frozen dependency provenance mismatch")
    if data.get("endpoint_policy") != "central_only":
        return _invalid(path, "Proposed endpoint policy is not central_only")
    return ValidationResult(True, "SKIP_VALID", "valid Q002S640/K8 central-only reconstruction")


def validate_fit_first_reconstruction(path: Path, record: CaseRecord) -> ValidationResult:
    if not path.exists():
        return _invalid(path, "output absent")
    try:
        manifest = _read(path / "manifest.json")
        route = resolve_fit_first_route(path)
    except (FileNotFoundError, ValueError) as exc:
        return _invalid(path, str(exc))
    if route.route_kind != "exp1_1_fit_first" or route.route_schema != EXP1_1_SCHEMA:
        return _invalid(path, "Fit-first route is not the CYJ Experiment 1.1 frozen-map one-stage baseline")
    if (manifest.get("subject"), manifest.get("case_id"), manifest.get("profile_id"), manifest.get("fold_id"), manifest.get("fold_count")) != (record.subject, record.case_id, record.profile_id, record.fold_id, record.fold_count):
        return _invalid(path, "Fit-first CaseRecord identity/fold provenance mismatch")
    sparse = manifest.get("sparse_manifest")
    if not isinstance(sparse, dict) or Path(str(sparse.get("path", ""))).resolve() != record.sparse_manifest.resolve() or sparse.get("sha256") != record.sparse_manifest_sha256 or sparse.get("generator_git_sha") != record.generator_git_sha:
        return _invalid(path, "Fit-first sparse-manifest provenance mismatch")
    return ValidationResult(True, "SKIP_VALID", "valid SCMR one-stage Fit-first reconstruction")


def _schema(path: Path, expected: str) -> ValidationResult:
    if not path.exists():
        return _invalid(path, "output absent")
    try:
        if _read(path / "manifest.json").get("schema") != expected:
            return _invalid(path, f"expected {expected}")
    except (FileNotFoundError, ValueError) as exc:
        return _invalid(path, str(exc))
    return ValidationResult(True, "SKIP_VALID", f"valid {expected}")


def _matches_record(value: Any, path: Path) -> bool:
    return isinstance(value, dict) and path.is_file() and Path(str(value.get("path", ""))).resolve() == path.resolve() and value.get("sha256") == sha256(path)


def validate_proposed_endpoint(path: Path, record: CaseRecord, root: Path) -> ValidationResult:
    """Admit only a central Proposed export bound to this current checkpoint."""
    if not path.exists():
        return _invalid(path, "output absent")
    try:
        manifest = _read(path / "manifest.json")
        checkpoint = _paths(root, record)["proposed"] / "model.pt"
        if (manifest.get("schema"), manifest.get("subject"), manifest.get("condition"), manifest.get("method"), manifest.get("endpoint")) != ("phase4r_formal_proposed_export/v1", record.subject, record.case_id, "Proposed", "central"):
            return _invalid(path, "Proposed endpoint schema/identity mismatch")
        if not _matches_record(manifest.get("source_checkpoint"), checkpoint):
            return _invalid(path, "Proposed endpoint checkpoint provenance mismatch")
    except (FileNotFoundError, ValueError, TypeError) as exc:
        return _invalid(path, str(exc))
    return ValidationResult(True, "SKIP_VALID", "valid current central Proposed endpoint")


def validate_proposed_native_bundle(path: Path, record: CaseRecord, root: Path) -> ValidationResult:
    """Admit only a direct-native bundle bound to this current Proposed checkpoint."""
    if not path.exists():
        return _invalid(path, "output absent")
    try:
        manifest = _read(path / "manifest.json")
        checkpoint = _paths(root, record)["proposed"] / "model.pt"
        if manifest.get("schema") != "direct_native_prediction_bundle/v1" or manifest.get("endpoint") != "central":
            return _invalid(path, "Proposed native bundle schema/endpoint mismatch")
        if not _matches_record(manifest.get("checkpoint"), checkpoint):
            return _invalid(path, "Proposed native bundle checkpoint provenance mismatch")
    except (FileNotFoundError, ValueError, TypeError) as exc:
        return _invalid(path, str(exc))
    return ValidationResult(True, "SKIP_VALID", "valid central Proposed native bundle")


def validate_fit_first_native_bundle(path: Path, record: CaseRecord, root: Path) -> ValidationResult:
    """Admit only a v6 bundle bound to current Exp1.1 route and endpoint artifacts."""
    if not path.exists():
        return _invalid(path, "output absent")
    try:
        manifest = _read(path / "manifest.json")
        paths = _paths(root, record)
        endpoint_manifest = paths["fit_endpoint"] / "manifest.json"
        route_manifest = paths["fit_first"] / "manifest.json"
        if manifest.get("schema") != "fit_first_native_prediction_bundle/v6" or manifest.get("endpoint") != "central":
            return _invalid(path, "Fit-first native bundle schema/endpoint mismatch")
        if manifest.get("effective_mask_contract") != FIT_FIRST_NATIVE_EFFECTIVE_MASK_CONTRACT:
            return _invalid(path, "Fit-first native bundle effective-query-mask contract mismatch")
        if not _matches_record(manifest.get("endpoint_source_manifest"), endpoint_manifest):
            return _invalid(path, "Fit-first native bundle endpoint provenance mismatch")
        route = resolve_fit_first_route(paths["fit_first"])
        if route.route_kind != "exp1_1_fit_first" or route.route_schema != EXP1_1_SCHEMA or Path(route.root).resolve() != paths["fit_first"].resolve():
            return _invalid(path, "Fit-first native bundle route is not current CYJ Experiment 1.1")
        if not _matches_record(manifest.get("route_manifest"), route_manifest):
            return _invalid(path, "Fit-first native bundle route-manifest provenance mismatch")
        route_provenance = manifest.get("route")
        if not isinstance(route_provenance, dict) or route_provenance.get("route_kind") != "exp1_1_fit_first" or route_provenance.get("route_schema") != EXP1_1_SCHEMA or Path(str(route_provenance.get("source_fit_first_root", ""))).resolve() != paths["fit_first"].resolve():
            return _invalid(path, "Fit-first native bundle route identity provenance mismatch")
    except (FileNotFoundError, ValueError, TypeError) as exc:
        return _invalid(path, str(exc))
    return ValidationResult(True, "SKIP_VALID", "valid current Exp1.1 central Fit-first native bundle")


def validate_l100_native_support(path: Path, record: CaseRecord, root: Path) -> ValidationResult:
    """Admit only a support audit bound to current Proposed endpoint/checkpoint inputs."""
    if not path.exists():
        return _invalid(path, "output absent")
    try:
        manifest = _read(path / "manifest.json")
        paths = _paths(root, record)
        checkpoint, support_mask = paths["proposed"] / "model.pt", paths["proposed"] / "support_mask_3D.nii.gz"
        if manifest.get("schema") != "q004_direct_native_endpoint_audit/v2":
            return _invalid(path, "L100 native support schema mismatch")
        if not _matches_record(manifest.get("checkpoint"), checkpoint):
            return _invalid(path, "L100 native support checkpoint provenance mismatch")
        central, masks = manifest.get("central_volumes"), manifest.get("frozen_l100_masks")
        if not isinstance(central, dict) or not isinstance(masks, dict):
            return _invalid(path, "L100 native support endpoint/mask provenance is absent")
        for parameter in ("t1", "t2"):
            if not _matches_record(central.get(parameter), paths["proposed_endpoint"] / f"{parameter.upper()}_3D.nii.gz") or not _matches_record(masks.get(parameter), support_mask):
                return _invalid(path, f"L100 native support {parameter} endpoint/mask provenance mismatch")
    except (FileNotFoundError, ValueError, TypeError) as exc:
        return _invalid(path, str(exc))
    return ValidationResult(True, "SKIP_VALID", "valid current L100 native support audit")


def validate_current_native_mask(path: Path, record: CaseRecord, root: Path) -> ValidationResult:
    """Admit only a current native mask bound to this support audit and sparse profile."""
    if not path.exists():
        return _invalid(path, "output absent")
    try:
        manifest = _read(path / "manifest.json")
        paths = _paths(root, record)
        if (manifest.get("schema"), manifest.get("subject"), manifest.get("condition")) != ("phase4r_current_native_central_component_mask/v1", record.subject, "L100_fold0"):
            return _invalid(path, "Current native mask schema/identity mismatch")
        if not _matches_record(manifest.get("frozen_l100_native_support_manifest"), paths["native_support"] / "manifest.json"):
            return _invalid(path, "Current native mask L100 support provenance mismatch")
        if not _matches_record(manifest.get("sparse_manifest"), record.sparse_manifest):
            return _invalid(path, "Current native mask sparse-manifest provenance mismatch")
    except (FileNotFoundError, ValueError, TypeError) as exc:
        return _invalid(path, str(exc))
    return ValidationResult(True, "SKIP_VALID", "valid current native central-component mask")


def validate_fit_first_endpoint(path: Path, record: CaseRecord, records: dict[tuple[str, str], CaseRecord], root: Path) -> ValidationResult:
    """Admit only SCMR one-stage Fit-first endpoint provenance for this case."""
    if not path.exists():
        return _invalid(path, "output absent")
    try:
        manifest = _read(path / "manifest.json")
        if manifest.get("schema") != FIT_FIRST_ENDPOINT_SCHEMA:
            return _invalid(path, f"expected {FIT_FIRST_ENDPOINT_SCHEMA}")
        if manifest.get("subject") != record.subject or manifest.get("condition") != record.case_id or manifest.get("method") != "Fit-first" or manifest.get("endpoint") != "central":
            return _invalid(path, "Fit-first endpoint identity mismatch")
        routes = manifest.get("fit_first_routes")
        l100_record = records[(record.subject, "L100")]
        for name, source_record in (("l100", l100_record), ("sparse", record)):
            route_record = routes.get(name) if isinstance(routes, dict) else None
            source_root = _paths(root, source_record)["fit_first"]
            route = resolve_fit_first_route(source_root)
            if route.route_kind != "exp1_1_fit_first" or route.route_schema != EXP1_1_SCHEMA:
                return _invalid(path, f"{name}: Fit-first route is not CYJ Experiment 1.1 one-stage")
            if not isinstance(route_record, dict) or route_record.get("route_schema") != EXP1_1_SCHEMA or Path(str(route_record.get("source_fit_first_root", ""))).resolve() != source_root.resolve():
                return _invalid(path, f"{name}: endpoint route provenance mismatch")
            models, results = route_record.get("formal_reconstruction_model"), route_record.get("parameter_results")
            if not isinstance(models, dict) or not isinstance(results, dict) or any(not _matches_record(models.get(parameter), route.models[parameter]) or not _matches_record(results.get(parameter), route.results[parameter]) for parameter in ("t1", "t2")):
                return _invalid(path, f"{name}: endpoint reconstruction provenance mismatch")
        parameters = manifest.get("parameters")
        if not isinstance(parameters, dict):
            return _invalid(path, "Fit-first endpoint parameter provenance is absent")
        for parameter in ("t1", "t2"):
            item = parameters.get(parameter)
            volume, mask = path / parameter / "volume_ms.nii.gz", path / parameter / "query_effective_mask.nii.gz"
            if not isinstance(item, dict) or item.get("schema") != FIT_FIRST_PARAMETER_ENDPOINT_SCHEMA or item.get("parameter") != parameter:
                return _invalid(path, f"{parameter}: parameter endpoint schema mismatch")
            if any(item.get(field) != manifest.get(field) for field in ("subject", "condition", "endpoint")):
                return _invalid(path, f"{parameter}: parameter endpoint identity mismatch")
            if item.get("fit_first_routes") != routes:
                return _invalid(path, f"{parameter}: parameter/top-level Fit-first route provenance mismatch")
            if _read(path / parameter / "manifest.json") != item:
                return _invalid(path, f"{parameter}: embedded endpoint manifest disagrees with disk")
            if not _matches_record(item.get("output_volume_ms"), volume) or not _matches_record(item.get("effective_query_mask"), mask):
                return _invalid(path, f"{parameter}: endpoint volume/effective-mask provenance mismatch")
    except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
        return _invalid(path, str(exc))
    return ValidationResult(True, "SKIP_VALID", "valid SCMR one-stage Fit-first endpoint")


def validate_l100_metric(path: Path, record: CaseRecord, root: Path) -> ValidationResult:
    """Admit only v5 metrics with formal SCMR v6 Fit-first-bundle provenance."""
    if not path.exists():
        return _invalid(path, "output absent")
    try:
        manifest = _read(path / "manifest.json")
        paths = _paths(root, record)
        if manifest.get("schema") != L100_METRIC_SCHEMA or manifest.get("subject") != record.subject or manifest.get("condition") != "L100_fold0":
            return _invalid(path, "L100 metric schema/identity mismatch")
        if FIT_FIRST_COMMON_SUPPORT_TERM not in str(manifest.get("primary_support_equation", "")):
            return _invalid(path, "L100 metric does not record Fit-first effective query mask support")
        if not _matches_record(manifest.get("direct_bundle"), paths["direct_bundle"] / "manifest.json") or not _matches_record(manifest.get("fit_first_bundle"), paths["fit_bundle"] / "manifest.json") or not _matches_record(manifest.get("mask_manifest"), paths["native_mask"] / "manifest.json"):
            return _invalid(path, "L100 metric upstream bundle/mask provenance mismatch")
        aggregate_path = path / "aggregate_metrics.json"
        if not _matches_record(manifest.get("aggregate_metrics"), aggregate_path):
            return _invalid(path, "L100 metric aggregate provenance mismatch")
        aggregate = _read(aggregate_path)
        if aggregate.get("schema") != "final_pose_native_common_support_metrics/v4" or FIT_FIRST_COMMON_SUPPORT_TERM not in str(aggregate.get("primary_support_equation", "")):
            return _invalid(path, "L100 aggregate does not record mask-aware common support")
        fit_bundle = _read(paths["fit_bundle"] / "manifest.json")
        if fit_bundle.get("schema") != "fit_first_native_prediction_bundle/v6" or not _matches_record(fit_bundle.get("endpoint_source_manifest"), paths["fit_endpoint"] / "manifest.json"):
            return _invalid(path, "L100 Fit-first bundle endpoint provenance mismatch")
    except (FileNotFoundError, ValueError, TypeError) as exc:
        return _invalid(path, str(exc))
    return ValidationResult(True, "SKIP_VALID", "valid mask-aware v5 L100 metric")


def validate_sparse_metric(path: Path, record: CaseRecord, root: Path) -> ValidationResult:
    """Admit only v2 held-out metrics tied to the v2 Fit-first endpoint."""
    if not path.exists():
        return _invalid(path, "output absent")
    try:
        manifest = _read(path / "manifest.json")
        paths = _paths(root, record)
        if manifest.get("schema") != SPARSE_METRIC_SCHEMA or manifest.get("subject") != record.subject or manifest.get("condition") != record.case_id:
            return _invalid(path, "sparse metric schema/identity mismatch")
        if FIT_FIRST_COMMON_SUPPORT_TERM not in str(manifest.get("primary_support_equation", "")):
            return _invalid(path, "sparse metric does not record Fit-first effective query mask support")
        provenance = manifest.get("endpoint_provenance")
        if not isinstance(provenance, dict) or not _matches_record(provenance.get("proposed_manifest"), paths["proposed_endpoint"] / "manifest.json") or not _matches_record(provenance.get("fit_first_manifest"), paths["fit_endpoint"] / "manifest.json"):
            return _invalid(path, "sparse metric endpoint provenance mismatch")
        aggregate_path = path / "aggregate_metrics.json"
        if not _matches_record(manifest.get("aggregate_metrics"), aggregate_path):
            return _invalid(path, "sparse metric aggregate provenance mismatch")
        aggregate = _read(aggregate_path)
        if aggregate.get("schema") != "final_sparse_native_common_support_metrics/v2" or FIT_FIRST_COMMON_SUPPORT_TERM not in str(aggregate.get("primary_support_equation", "")):
            return _invalid(path, "sparse aggregate does not record mask-aware common support")
    except (FileNotFoundError, ValueError, TypeError) as exc:
        return _invalid(path, str(exc))
    return ValidationResult(True, "SKIP_VALID", "valid formal SCMR one-stage sparse metric")


def validate_final_native_grid_metric(path: Path, record: CaseRecord, root: Path) -> ValidationResult:
    """Admit only the non-morphological Final5 original-native-grid metric."""
    if not path.exists():
        return _invalid(path, "output absent")
    try:
        manifest = _read(path / "manifest.json")
        paths = _paths(root, record)
        if (manifest.get("schema"), manifest.get("subject"), manifest.get("condition")) != (FINAL_NATIVE_GRID_METRIC_SCHEMA, record.subject, record.case_id):
            return _invalid(path, "Final native-grid metric schema/identity mismatch")
        if manifest.get("legacy_current_native_mask_used") is not False or manifest.get("no_connected_component_filtering") is not True:
            return _invalid(path, "Final native-grid metric must not use central-component masking")
        provenance = manifest.get("endpoint_provenance")
        if not isinstance(provenance, dict) or not _matches_record(provenance.get("proposed_manifest"), paths["proposed_endpoint"] / "manifest.json") or not _matches_record(provenance.get("fit_first_manifest"), paths["fit_endpoint"] / "manifest.json"):
            return _invalid(path, "Final native-grid endpoint provenance mismatch")
        if not _matches_record(manifest.get("sparse_manifest"), record.sparse_manifest):
            return _invalid(path, "Final native-grid sparse-manifest provenance mismatch")
        aggregate_path = path / "aggregate_metrics.json"
        if not _matches_record(manifest.get("aggregate_metrics"), aggregate_path):
            return _invalid(path, "Final native-grid aggregate provenance mismatch")
        aggregate = _read(aggregate_path)
        equations = aggregate.get("support_equations", {})
        if aggregate.get("schema") != "final5_native_grid_common_support_metrics/v2" or "fit_first_query_effective_mask" not in str(equations.get("l100_mask", "")) or "fit_first_query_effective_mask" in str(equations.get("paired", "")):
            return _invalid(path, "Final native-grid aggregate common-support contract mismatch")
    except (FileNotFoundError, ValueError, TypeError) as exc:
        return _invalid(path, str(exc))
    return ValidationResult(True, "SKIP_VALID", "valid Final5 original-native-grid common-support metric")


_STAGE_ARTIFACT_PATH = {
    "proposed_reconstruction": "proposed",
    "fit_first_reconstruction": "fit_first",
    "proposed_endpoint": "proposed_endpoint",
    "fit_first_endpoint": "fit_endpoint",
    "proposed_native_bundle": "direct_bundle",
    "fit_first_native_bundle": "fit_bundle",
    "l100_native_support": "native_support",
    "current_native_mask": "native_mask",
    "l100_metrics": "l100_metrics",
    "sparse_metrics": "sparse_metrics",
    "final_native_grid_metrics": "final_native_grid_metrics",
}


def _stage(result: ValidationResult, *, dependency: bool = False) -> dict[str, str]:
    return {"state": "BLOCKED_DEPENDENCY" if dependency and result.state != "BLOCKED_EXISTING_INVALID" else result.state, "reason": result.reason}


def _with_artifact_paths(stages: dict[str, dict[str, str]], paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    for stage, item in stages.items():
        item["artifact_path"] = str(paths[_STAGE_ARTIFACT_PATH[stage]].resolve())
    return stages


def build_batch_plan(registry: str | Path, final_root: str | Path, *, device: str = "0", records: list[CaseRecord] | None = None) -> dict[str, Any]:
    """Read-only artifact truth for plan, status, validation, and resume."""
    records, root = (load_final5_registry(registry) if records is None else list(records)), Path(final_root)
    cases: dict[str, Any] = {}
    records_by_key = {(item.subject, item.case_id): item for item in records}
    l100_ready: dict[str, bool] = {}
    for record in records:
        if record.case_id != "L100":
            continue
        paths = _paths(root, record)
        proposed, fit = validate_proposed_reconstruction(paths["proposed"], record), validate_fit_first_reconstruction(paths["fit_first"], record)
        endpoints = validate_proposed_endpoint(paths["proposed_endpoint"], record, root), validate_fit_first_endpoint(paths["fit_endpoint"], record, records_by_key, root)
        metric = validate_final_native_grid_metric(paths["final_native_grid_metrics"], record, root)
        stages = {"proposed_reconstruction": _stage(proposed), "fit_first_reconstruction": _stage(fit), "proposed_endpoint": _stage(endpoints[0], dependency=not proposed.valid), "fit_first_endpoint": _stage(endpoints[1], dependency=not fit.valid), "final_native_grid_metrics": _stage(metric, dependency=not (endpoints[0].valid and endpoints[1].valid))}
        stages = _with_artifact_paths(stages, paths)
        l100_ready[record.subject] = all(value["state"] == "SKIP_VALID" for value in stages.values())
        cases[f"{record.subject}/{record.case_id}"] = {"profile_id": record.profile_id, "fold_id": record.fold_id, "metric_kind": "formal original-native-grid exact common-support metric", "stages": stages}
    for record in records:
        if record.case_id == "L100":
            continue
        paths = _paths(root, record)
        proposed, fit = validate_proposed_reconstruction(paths["proposed"], record), validate_fit_first_reconstruction(paths["fit_first"], record)
        metric = validate_final_native_grid_metric(paths["final_native_grid_metrics"], record, root)
        proposed_endpoint = validate_proposed_endpoint(paths["proposed_endpoint"], record, root)
        fit_endpoint = validate_fit_first_endpoint(paths["fit_endpoint"], record, records_by_key, root)
        dependency = not (proposed_endpoint.valid and fit_endpoint.valid)
        stages = {
            "proposed_reconstruction": _stage(proposed),
            "fit_first_reconstruction": _stage(fit),
            "proposed_endpoint": _stage(proposed_endpoint, dependency=not proposed.valid),
            "fit_first_endpoint": _stage(fit_endpoint, dependency=not fit.valid),
            "final_native_grid_metrics": _stage(metric, dependency=dependency),
        }
        cases[f"{record.subject}/{record.case_id}"] = {"profile_id": record.profile_id, "fold_id": record.fold_id, "metric_kind": "formal manifest-held-out original-native-grid exact common-support metric", "stages": _with_artifact_paths(stages, paths)}
    return {"schema": "phase4r_final5_batch_plan/v1", "registry": str(Path(registry).resolve()), "final_root": str(root.resolve()), "max_gpu_jobs": MAX_GPU_JOBS, "cases": cases, "device": str(device), "computational_metrics_only": True}


def _case_stages(record: CaseRecord) -> tuple[str, ...]:
    return ("proposed_reconstruction", "fit_first_reconstruction", "proposed_endpoint", "fit_first_endpoint", "final_native_grid_metrics")


def _validate_stage_output(stage: str, record: CaseRecord, root: Path, records: dict[tuple[str, str], CaseRecord] | None = None) -> ValidationResult:
    """Validate one published artifact without rescanning the frozen registry."""
    paths = _paths(root, record)
    if stage == "proposed_reconstruction":
        return validate_proposed_reconstruction(paths["proposed"], record)
    if stage == "fit_first_reconstruction":
        return validate_fit_first_reconstruction(paths["fit_first"], record)
    if stage == "fit_first_endpoint":
        if records is None:
            raise ValueError("Fit-first endpoint validation requires registry records")
        return validate_fit_first_endpoint(paths["fit_endpoint"], record, records, root)
    if stage == "proposed_endpoint":
        return validate_proposed_endpoint(paths["proposed_endpoint"], record, root)
    if stage == "proposed_native_bundle":
        return validate_proposed_native_bundle(paths["direct_bundle"], record, root)
    if stage == "fit_first_native_bundle":
        return validate_fit_first_native_bundle(paths["fit_bundle"], record, root)
    if stage == "l100_native_support":
        return validate_l100_native_support(paths["native_support"], record, root)
    if stage == "current_native_mask":
        return validate_current_native_mask(paths["native_mask"], record, root)
    if stage == "l100_metrics":
        return validate_l100_metric(paths["l100_metrics"], record, root)
    if stage == "sparse_metrics":
        return validate_sparse_metric(paths["sparse_metrics"], record, root)
    if stage == "final_native_grid_metrics":
        if records is None:
            raise ValueError("Final native-grid metric validation requires registry records")
        return validate_final_native_grid_metric(paths["final_native_grid_metrics"], record, root)
    expected = {
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
    if stage == "final_native_grid_metrics":
        return (same("proposed_endpoint"), same("fit_first_endpoint"))
    return ()


def _validate_stage_locally(stage: str, record: CaseRecord, root: Path, records: dict[tuple[str, str], CaseRecord]) -> ValidationResult:
    """Read only the current output and its explicit, published dependencies."""
    result = _validate_stage_output(stage, record, root, records)
    if result.state == "BLOCKED_EXISTING_INVALID":
        return result
    for dependency_stage, dependency_record in _stage_dependencies(stage, record, records):
        dependency = _validate_stage_output(dependency_stage, dependency_record, root, records)
        if not dependency.valid:
            return ValidationResult(False, "BLOCKED_DEPENDENCY", f"{dependency_record.subject}/{dependency_record.case_id} {dependency_stage}: {dependency.reason}")
    return result


def _runtime_dependency_reason(stage: str, record: CaseRecord, plan: dict[str, Any], records: dict[tuple[str, str], CaseRecord]) -> str | None:
    """Use the execute-process state after its one strict preflight."""
    for dependency_stage, dependency_record in _stage_dependencies(stage, record, records):
        try:
            dependency = plan["cases"][f"{dependency_record.subject}/{dependency_record.case_id}"]["stages"][dependency_stage]
        except KeyError:
            return f"missing in-memory dependency state: {dependency_record.subject}/{dependency_record.case_id} {dependency_stage}"
        if dependency.get("state") != "SKIP_VALID":
            return f"{dependency_record.subject}/{dependency_record.case_id} {dependency_stage}: {dependency.get('reason', dependency.get('state', 'not valid'))}"
    return None


def case_commands(record: CaseRecord, final_root: str | Path, *, device: str, python: str, nesvor: str) -> dict[str, Any]:
    root, paths = Path(final_root), _paths(Path(final_root), record)
    devices = normalize_device(device)
    proposed = [python, "-m", "quality_experiments.Q002_B6_roi_same_anchor_mse.run_q002_reconstruction", "--cropped-prepared-root", str(record.prepared_root), "--subject-id", record.subject, "--b6-model", PROPOSED_B6, "--expected-b6-model-sha256", PROPOSED_B6_SHA256, "--signal-simulator", DECODER, "--protocol", PROTOCOL, "--expected-protocol-sha256", PROTOCOL_SHA256, "--sparse-manifest", str(record.sparse_manifest), "--expected-sparse-generator-git-sha", record.generator_git_sha, "--expected-sparse-manifest-sha256", record.sparse_manifest_sha256, "--endpoint-policy", "central_only", "--output", str(paths["proposed"]), "--device", devices.torch_device, "--experiment-id", "Q002S640_B6_roi_same_anchor_mse", "--anchor-batch-size", "640"]
    fit_env = {"SUBJECT": record.subject, "STANDARDIZED_NATIVE_DICTIONARY_ROOT": str(root / "02_standardized_12_15_10" / "native_dictionary" / record.subject), "SPARSE_MANIFEST": str(record.sparse_manifest), "EXPECTED_GENERATOR_GIT_SHA": record.generator_git_sha, "EXPECTED_MANIFEST_SHA256": record.sparse_manifest_sha256, "NESVOR": nesvor, "OUTPUT": str(paths["fit_first"]), "DEVICE": devices.nesvor_device, "PYTHON": python}
    return {"proposed_reconstruction": proposed, "fit_first_reconstruction": {"argv": ["bash", str(Path(__file__).with_name("server_commands") / "run_exp1_1_fit_first.sh")], "env": fit_env}}


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
    elif stage == "final_native_grid_metrics":
        build_final_native_grid_metrics(proposed_endpoint_root=paths["proposed_endpoint"], fit_first_endpoint_root=paths["fit_endpoint"], r0_native_maps_root=native, sparse_manifest=record.sparse_manifest, output=paths["final_native_grid_metrics"])


def execute_batch(registry: str | Path, final_root: str | Path, *, device: str, python: str, nesvor: str, subject: str | None = None, case: str | None = None) -> dict[str, Any]:
    """Run serially after one strict preflight; never delete partial roots."""
    all_records = load_final5_registry(registry)
    root = Path(final_root)
    result = deepcopy(build_batch_plan(registry, root, device=device, records=all_records))
    invalid = [
        f"{key} {stage} path={value.get('artifact_path', '<unknown>')} reason={value.get('reason', '<none>')}"
        for key, item in result["cases"].items()
        for stage, value in item["stages"].items()
        if value["state"] == "BLOCKED_EXISTING_INVALID"
    ]
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
                dependency_reason = _runtime_dependency_reason(stage, record, result, records_by_key)
                if dependency_reason is not None:
                    stage_plan.update({"state": "BLOCKED_DEPENDENCY", "reason": dependency_reason})
                    continue
                log = _state_root(root) / "logs" / record.subject / record.case_id / f"{stage}.log"
                if stage in SUBPROCESS_STAGES:
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
                    post = _validate_stage_output(stage, record, root, records_by_key)
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
        values = list(stages.values()); state = "COMPLETE" if values and all(value["state"] == "SKIP_VALID" for value in values) else "FAILED" if any(value["state"] == "FAILED" for value in values) else "BLOCKED" if any(value["state"].startswith("BLOCKED") for value in values) else "READY"
        counts[state] = counts.get(state, 0) + 1
        print(f"{subject:<7} {case:<28} {stages.get('proposed_reconstruction', {}).get('state', '-'): <4} {stages.get('fit_first_reconstruction', {}).get('state', '-'): <4} {stages.get('proposed_endpoint', {}).get('state', '-'): <4} {stages.get('fit_first_endpoint', {}).get('state', '-'): <4} {stages.get('final_native_grid_metrics', {}).get('state', '-'): <6} {state}")
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
