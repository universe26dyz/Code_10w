"""Immutable fixed-L100 exporters and contracts for matched sparse outputs.

This module is deliberately server-run only for volume sampling.  It never
trains, reconstructs, or refits a model; its role is to query already frozen
checkpoints and formalize existing Proposed exports with unambiguous endpoint
semantics.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .contracts import sha256, write_immutable_json
from .fit_first_mask_audit import (
    NORMALIZATION_SCALES,
    _denormalize,
    _run,
    checkpoint_mask_export_command,
    sample_volume_runtime_command,
)
from .fit_first_route_contract import FitFirstFormalRoute, resolve_fit_first_route


ENDPOINTS: dict[str, dict[str, Any]] = {
    "central": {
        "endpoint": "central",
        "fit_first_sampling_mode": "k1",
        "query_mask_semantics": "frozen_l100_checkpoint_mask_only",
        "proposed_source_subdirectory": ".",
        "export_kind": "central_no_map_psf_quantitative_export",
        "output_query_psf": {
            "domain": "quantitative_map",
            "description": "central / no map-domain PSF",
            "output_resolution_mm": 1.0,
            "output_psf_factor": 0.0,
            "n_inference_samples": 1,
        },
    },
    "psf128": {
        "endpoint": "psf128",
        "fit_first_sampling_mode": "k128",
        "query_mask_semantics": "frozen_l100_checkpoint_mask_only",
        "proposed_source_subdirectory": "formal_export_psf128",
        "export_kind": "post_hoc_map_domain_quantitative_export",
        "output_query_psf": {
            "domain": "quantitative_map",
            "description": "post-hoc map-domain quantitative PSF128 export",
            "output_resolution_mm": 1.0,
            "output_psf_factor": 1.0,
            "n_inference_samples": 128,
        },
    },
}


def require_new_output(output: str | Path) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    return output


def endpoint_contract(endpoint: str) -> dict[str, Any]:
    try:
        return copy.deepcopy(ENDPOINTS[endpoint])
    except KeyError as error:
        raise ValueError(f"unknown matched-output endpoint: {endpoint}") from error


def require_matched_endpoints(proposed_endpoint: str, fit_first_endpoint: str) -> dict[str, Any]:
    """Refuse comparisons whose map-query semantics do not match."""
    proposed, fit_first = endpoint_contract(proposed_endpoint), endpoint_contract(fit_first_endpoint)
    if proposed["endpoint"] != fit_first["endpoint"]:
        raise ValueError(
            "endpoint mismatch: final comparison requires Proposed and Fit-first "
            f"to use the same endpoint, got {proposed_endpoint} vs {fit_first_endpoint}"
        )
    return proposed


def primary_decomposition_contract(l100_full_voxels: int, sparse_not_l100_boundary_discrepancy_voxels: int) -> dict[str, Any]:
    """Describe diagnostic decomposition without replacing the L100 primary domain."""
    discrepancy = int(sparse_not_l100_boundary_discrepancy_voxels)
    return {
        "primary_region": "l100_full",
        "primary_region_definition": "frozen L100 native-domain support",
        "l100_full_voxels": int(l100_full_voxels),
        "diagnostic_boundary_discrepancy_voxels": discrepancy,
        "decomposition_exact": discrepancy == 0,
        "diagnostic_regions": ("recovered_only", "sparse_region"),
    }


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _record(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def verified_fit_first_effective_query_mask(endpoint_root: str | Path, parameter: str) -> tuple[Path, dict[str, str]]:
    """Resolve the formal endpoint's immutable effective query mask fail-closed."""
    root = Path(endpoint_root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "phase4r_fixed_l100_fit_first_export_bundle/v2":
        raise ValueError("Fit-first formal endpoint schema mismatch")
    record = manifest.get("parameters", {}).get(parameter, {}).get("effective_query_mask")
    expected = root / parameter / "query_effective_mask.nii.gz"
    if not isinstance(record, Mapping) or Path(str(record.get("path", ""))).resolve() != expected.resolve():
        raise ValueError(f"{parameter}: formal endpoint effective query mask provenance mismatch")
    if not expected.is_file() or record.get("sha256") != sha256(expected):
        raise ValueError(f"{parameter}: formal endpoint effective query mask SHA256 mismatch")
    return expected, _record(expected)


def _training_psf_from_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve declared NeSVoR training semantics without inferring equivalence."""
    candidates = (payload.get("nesvor_training_n_samples"), payload.get("n_samples"), payload.get("training_n_samples"))
    samples = next((value for value in candidates if value is not None), None)
    return {
        "method": "Fit-first native 2D fitting + NeSVoR",
        "nesvor_training_n_samples": samples,
        "semantics": "from sparse manifest when declared; not equated to output/query PSF samples",
    }


def fixed_l100_export_manifest(
    *, subject: str, condition: str, endpoint: str, sparse_manifest: str | Path,
    r0_native_maps_root: str | Path, sparse_checkpoint: str | Path, l100_mask: str | Path,
    training_route: str, training_psf: Mapping[str, Any], runtime: Mapping[str, Any] | None = None,
    l100_route: FitFirstFormalRoute | None = None, sparse_route: FitFirstFormalRoute | None = None,
) -> dict[str, Any]:
    """Build the public provenance contract for one Fit-first endpoint export."""
    contract = endpoint_contract(endpoint)
    result = {
        "schema": "phase4r_fixed_l100_fit_first_export/v2",
        "subject": subject,
        "condition": condition,
        "method": "Fit-first",
        "training_route": training_route,
        "training_psf": dict(training_psf),
        "export_query_psf": contract["output_query_psf"],
        "output_query_psf": contract["output_query_psf"],
        "endpoint": contract["endpoint"],
        "query_mask_source": _record(l100_mask),
        "query_mask_semantics": contract["query_mask_semantics"],
        "sparse_checkpoint": _record(sparse_checkpoint),
        "sparse_manifest": _record(sparse_manifest),
        "r0_native_cache": {"path": str(Path(r0_native_maps_root).resolve()), "provenance": "immutable verified R0 native-map cache"},
        "sampling_runtime": dict(runtime or {}),
        "code_git_sha": _git_sha(),
        "no_training": True,
        "no_refitting": True,
        "no_reconstruction": True,
    }
    if l100_route is not None and sparse_route is not None:
        result["fit_first_routes"] = {"l100": l100_route.provenance(), "sparse": sparse_route.provenance()}
    return result


def _proposed_training_metadata(source_root: Path) -> dict[str, Any]:
    for name in ("experiment_manifest.json", "q002_route_manifest.json"):
        path = source_root / name
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            route = payload.get("route", {})
            sampling = route.get("sampling", {}) if isinstance(route, dict) else {}
            training_samples = payload.get("training_psf_samples", sampling.get("shared_training_psf_samples"))
            return {
                "training_route": payload.get("experiment_id", payload.get("route", "existing Proposed reconstruction")),
                "training_psf": {
                    "training_psf_samples": training_samples,
                    "semantics": payload.get("training_psf_semantics", "recorded in source manifest"),
                },
                "source_manifest": _record(path),
            }
    return {"training_route": "existing Proposed reconstruction", "training_psf": {"training_psf_samples": None, "semantics": "source manifest unavailable"}}


def formalize_proposed_export(*, subject: str, condition: str, source_root: str | Path, endpoint: str, output: str | Path,
                              training_metadata_override: Mapping[str, Any] | None = None) -> Path:
    """Copy immutable existing Proposed fields into one endpoint-labelled formal root."""
    source_root, output = Path(source_root), require_new_output(output)
    contract = endpoint_contract(endpoint)
    source = source_root / contract["proposed_source_subdirectory"]
    volumes = {parameter: source / f"{parameter.upper()}_3D.nii.gz" for parameter in ("t1", "t2")}
    for path in volumes.values():
        if not path.is_file():
            raise FileNotFoundError(f"missing Proposed {endpoint} export: {path}")
    checkpoint = source_root / "model.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing Proposed source checkpoint: {checkpoint}")
    if endpoint == "psf128":
        psf_manifest = source / "formal_export_manifest.json"
        if not psf_manifest.is_file():
            raise FileNotFoundError(f"missing Proposed PSF128 manifest: {psf_manifest}")
        if json.loads(psf_manifest.read_text(encoding="utf-8")).get("n_inference_samples") != 128:
            raise ValueError("Proposed PSF128 source does not declare n_inference_samples=128")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        for parameter, source_path in volumes.items():
            shutil.copy2(source_path, staging / f"{parameter.upper()}_3D.nii.gz")
        metadata = _proposed_training_metadata(source_root)
        if training_metadata_override is not None:
            metadata.update(copy.deepcopy(dict(training_metadata_override)))
        manifest = {
            "schema": "phase4r_formal_proposed_export/v1",
            "subject": subject,
            "condition": condition,
            "method": "Proposed",
            **metadata,
            "endpoint": contract["endpoint"],
            "export_kind": contract["export_kind"],
            "export_query_psf": contract["output_query_psf"],
            "export_psf": contract["output_query_psf"],
            "output_query_psf": contract["output_query_psf"],
            "source_checkpoint": _record(checkpoint),
            "source_volumes": {parameter: _record(path) for parameter, path in volumes.items()},
            "code_git_sha": _git_sha(),
            "no_training": True,
            "no_refitting": True,
            "no_reconstruction": True,
        }
        write_immutable_json(staging / "manifest.json", manifest)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def export_fixed_l100_fit_first(*, subject: str, condition: str, l100_fit_first_root: str | Path,
                                sparse_fit_first_root: str | Path, sparse_manifest: str | Path,
                                r0_native_maps_root: str | Path, nesvor_executable: str | Path,
                                endpoint: str, output: str | Path, device: str) -> Path:
    """Query sparse frozen Fit-first checkpoints on L100 masks, without reconstruction."""
    l100_root, sparse_root = Path(l100_fit_first_root), Path(sparse_fit_first_root)
    sparse_manifest, r0_cache, nesvor = Path(sparse_manifest), Path(r0_native_maps_root), Path(nesvor_executable)
    output = require_new_output(output)
    contract = endpoint_contract(endpoint)
    if not sparse_manifest.is_file() or not r0_cache.is_dir() or not nesvor.is_file():
        raise FileNotFoundError("fixed-L100 export requires sparse manifest, R0 native cache, and NeSVoR executable")
    sparse_payload = json.loads(sparse_manifest.read_text(encoding="utf-8"))
    if sparse_payload.get("subject") not in (None, subject):
        raise ValueError("subject does not match sparse manifest")
    nesvor_python = nesvor.with_name("python")
    if not nesvor_python.is_file():
        raise FileNotFoundError(f"NeSVoR Python adjacent to executable is unavailable: {nesvor_python}")
    l100_route, sparse_route = resolve_fit_first_route(l100_root), resolve_fit_first_route(sparse_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        manifests: dict[str, Any] = {}
        for parameter, scale in NORMALIZATION_SCALES.items():
            l100_checkpoint = l100_route.models[parameter]
            sparse_checkpoint = sparse_route.models[parameter]
            for path in (l100_checkpoint, sparse_checkpoint):
                if not path.is_file():
                    raise FileNotFoundError(path)
            target = staging / parameter
            target.mkdir()
            l100_mask, sparse_mask = target / "l100_checkpoint_mask.nii.gz", target / "sparse_checkpoint_mask.nii.gz"
            _run(checkpoint_mask_export_command(nesvor_python, l100_checkpoint, l100_mask))
            _run(checkpoint_mask_export_command(nesvor_python, sparse_checkpoint, sparse_mask))
            normalized, effective_mask, runtime_path = target / "volume_normalized.nii.gz", target / "query_effective_mask.nii.gz", target / "runtime.json"
            _run(sample_volume_runtime_command(nesvor_python, sparse_checkpoint, l100_mask, normalized, effective_mask, runtime_path, device=device, mode=contract["fit_first_sampling_mode"]))
            volume_ms = target / "volume_ms.nii.gz"
            _denormalize(normalized, volume_ms, scale)
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            manifest = fixed_l100_export_manifest(
                subject=subject, condition=condition, endpoint=endpoint, sparse_manifest=sparse_manifest,
                r0_native_maps_root=r0_cache, sparse_checkpoint=sparse_checkpoint, l100_mask=l100_mask,
                training_route="Fit-first sparse checkpoint query", training_psf=_training_psf_from_manifest(sparse_payload), runtime=runtime,
                l100_route=l100_route, sparse_route=sparse_route,
            )
            manifest.update({
                "parameter": parameter,
                "normalization_scale_ms": scale,
                "l100_checkpoint": _record(l100_checkpoint),
                "sparse_checkpoint_mask": _record(sparse_mask),
                "effective_query_mask": _record(effective_mask),
                "output_volume_ms": _record(volume_ms),
            })
            manifests[parameter] = manifest
            write_immutable_json(target / "manifest.json", manifest)
        write_immutable_json(staging / "manifest.json", {
            "schema": "phase4r_fixed_l100_fit_first_export_bundle/v2", "subject": subject, "condition": condition,
            "method": "Fit-first", "endpoint": endpoint, "parameters": manifests,
            "no_training": True, "no_refitting": True, "no_reconstruction": True,
        })
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit-first")
    for name in ("subject", "condition", "l100-fit-first-root", "sparse-fit-first-root", "sparse-manifest", "r0-native-maps-root", "nesvor-executable", "output", "device"):
        fit.add_argument("--" + name, required=True)
    fit.add_argument("--endpoint", choices=tuple(ENDPOINTS), required=True)
    proposed = subparsers.add_parser("proposed")
    for name in ("subject", "condition", "source-root", "output"):
        proposed.add_argument("--" + name, required=True)
    proposed.add_argument("--endpoint", choices=tuple(ENDPOINTS), required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "fit-first":
        args["l100_fit_first_root"] = args.pop("l100_fit_first_root")
        args["sparse_fit_first_root"] = args.pop("sparse_fit_first_root")
        args["r0_native_maps_root"] = args.pop("r0_native_maps_root")
        args["nesvor_executable"] = args.pop("nesvor_executable")
        print(export_fixed_l100_fit_first(**args))
    else:
        args["source_root"] = args.pop("source_root")
        print(formalize_proposed_export(**args))


if __name__ == "__main__":
    main()
