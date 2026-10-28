"""Publish r3 Proposed predictions on the learned final pose of sparse inputs.

This is an additive downstream adapter for the migrated CYJ r3 layout.  It
does not run reconstruction or change a formal endpoint.  In particular, it
does not require the historical absolute ``prepared_root`` path retained in a
profile manifest: the copied local observations are admitted only when their
SHA256 values equal the manifest's immutable source records.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..phase2_clean.contracts import sha256, write_immutable_json
from ..phase2_clean.direct_native_endpoint_audit import _checkpoint_manifest, _load_model, _sparse_contract, direct_native_world_grid


SCHEMA = "cyj11_r3_direct_final_pose_native_bundle/v1"
STACKS = ("sax", "2ch", "4ch")


def _record(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _profile_mapping(item: Mapping[str, Any], stack: str) -> list[tuple[int, int]]:
    records = item.get("derived_to_original_group_idx")
    if not isinstance(records, list):
        raise ValueError(f"{stack}: sparse profile lacks derived-to-original group mapping")
    pairs: list[tuple[int, int]] = []
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("derived_group_idx"), int) or not isinstance(record.get("original_group_idx"), int):
            raise ValueError(f"{stack}: malformed derived-to-original group mapping")
        pairs.append((int(record["derived_group_idx"]), int(record["original_group_idx"])))
    pairs.sort()
    if [derived for derived, _original in pairs] != list(range(len(pairs))):
        raise ValueError(f"{stack}: derived group IDs are not contiguous")
    return pairs


def r3_sparse_prepared_contract(*, prepared_root: str | Path, sparse_manifest: str | Path,
                                subject: str, case: str) -> tuple[dict[str, Any], list[Path], dict[str, list[tuple[int, int]]]]:
    """Bind migrated sparse observations to a declared r3 manifest by digest."""
    root, manifest_path = Path(prepared_root), Path(sparse_manifest)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("subject") != subject or payload.get("status") != "ELIGIBLE":
        raise ValueError("r3 sparse manifest subject/status mismatch")
    schema, stacks = payload.get("schema_version"), payload.get("stacks")
    if not isinstance(stacks, Mapping) or set(stacks) != set(STACKS):
        raise ValueError("r3 sparse manifest lacks exact SAX/2CH/4CH stack records")

    paths: list[Path] = []
    mappings: dict[str, list[tuple[int, int]]] = {}
    if schema == "phase4r_sparse_profile_v2":
        if payload.get("case_id") != case:
            raise ValueError("r3 sparse profile case mismatch")
        active = [stack for stack in STACKS if isinstance(stacks[stack], Mapping) and stacks[stack].get("training_stack_present") is True]
        if not active:
            raise ValueError("r3 sparse profile has no active training stack")
        for stack in active:
            item = stacks[stack]
            assert isinstance(item, Mapping)
            record = item.get("derived_observation")
            path = root / stack / "observations.npz"
            if not isinstance(record, Mapping) or not isinstance(record.get("sha256"), str) or not path.is_file() or sha256(path) != record["sha256"]:
                raise ValueError(f"{stack}: migrated sparse observation SHA256 mismatch")
            paths.append(path)
            mappings[stack] = _profile_mapping(item, stack)
    elif schema == "phase4r_sparse_v1":
        condition, phase = payload.get("condition"), payload.get("phase")
        if not isinstance(condition, str) or not isinstance(phase, int) or case != f"{condition}_fold{phase}":
            raise ValueError("r3 legacy sparse case/condition/phase mismatch")
        for stack in STACKS:
            item, path = stacks[stack], root / stack / "observations.npz"
            if not isinstance(item, Mapping):
                raise ValueError(f"{stack}: malformed legacy sparse stack record")
            # L100 uses the source observation unchanged; L50/L25 use the
            # sparse NPZ whose group IDs were deliberately renumbered.  The
            # manifest mapping, not physical integer order, is authoritative.
            record = item.get("derived_observation", item.get("source_observation"))
            if not isinstance(record, Mapping) or not isinstance(record.get("sha256"), str) or not path.is_file() or sha256(path) != record["sha256"]:
                raise ValueError(f"{stack}: migrated legacy sparse observation SHA256 mismatch")
            with np.load(path, allow_pickle=False) as observations:
                groups = sorted(int(value) for value in np.unique(observations["group_idx"]))
            paths.append(path)
            if "derived_observation" in item:
                if "derived_to_original_group_idx" in item:
                    mappings[stack] = _profile_mapping(item, stack)
                else:
                    retained = item.get("retained_original_group_ids")
                    if not isinstance(retained, list) or not all(isinstance(value, int) for value in retained):
                        raise ValueError(f"{stack}: legacy derived observation lacks retained original group IDs")
                    if len(retained) != len(groups):
                        raise ValueError(f"{stack}: retained original group count disagrees with derived observation")
                    mappings[stack] = list(enumerate(int(value) for value in retained))
                if groups != [derived for derived, _original in mappings[stack]]:
                    raise ValueError(f"{stack}: legacy sparse observation group IDs disagree with manifest mapping")
            else:
                mappings[stack] = list(enumerate(groups))
    else:
        raise ValueError("unsupported r3 sparse manifest schema")
    return payload, paths, mappings


def _verify_checkpoint_r3_identity(checkpoint: Path, profile: Mapping[str, Any], sparse_path: Path,
                                   *, subject: str, case: str) -> None:
    manifest = _checkpoint_manifest(checkpoint)
    if manifest.get("subject_id") != subject:
        raise ValueError("r3 checkpoint subject mismatch")
    if profile.get("schema_version") == "phase4r_sparse_profile_v2":
        recorded = manifest.get("sparse_manifest")
        if manifest.get("case_id") != case or not isinstance(recorded, Mapping) or recorded.get("sha256") != sha256(sparse_path):
            raise ValueError("r3 checkpoint sparse-profile provenance mismatch")


def build_cyj11_r3_direct_final_pose_bundle(*, checkpoint: str | Path, prepared_root: str | Path,
                                             sparse_manifest: str | Path, subject: str, case: str,
                                             output: str | Path, device: str = "cuda:0") -> Path:
    """Query each observed native plane through the current Proposed final pose."""
    checkpoint, sparse_path, output = Path(checkpoint), Path(sparse_manifest), Path(output)
    if not checkpoint.is_file() or not sparse_path.is_file():
        raise FileNotFoundError("r3 Proposed final-pose bundle requires checkpoint and sparse manifest")
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    profile, paths, mappings = r3_sparse_prepared_contract(
        prepared_root=prepared_root, sparse_manifest=sparse_path, subject=subject, case=case)
    _verify_checkpoint_r3_identity(checkpoint, profile, sparse_path, subject=subject, case=case)
    sparse_contract = _sparse_contract(sparse_path)
    # This import carries the reconstruction runtime; keeping it here lets the
    # manifest-only contract validation run in a light CPU test environment.
    from ..phase2_clean.direct_native_bundles import _predict_central
    model, space, torch = _load_model(checkpoint, paths, device, sparse_contract=sparse_contract)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        planes: list[dict[str, Any]] = []
        offset = 0
        for path in paths:
            stack, mapping = path.parent.name, mappings[path.parent.name]
            with np.load(path, allow_pickle=False) as observations:
                groups = np.asarray(observations["group_idx"], int)
                if sorted(np.unique(groups).tolist()) != [derived for derived, _original in mapping]:
                    raise ValueError(f"{stack}: observation group IDs disagree with sparse manifest")
                for pose_index, (derived_group, original_group) in enumerate(mapping):
                    rows = np.flatnonzero(groups == derived_group)
                    if not len(rows):
                        raise ValueError(f"{stack}: sparse observation is missing group {derived_group}")
                    row = int(rows[0])
                    shape = tuple(int(value) for value in np.asarray(observations["images"])[row].shape)
                    spacing = np.asarray(observations["pixel_spacing_rc_mm"])[row]
                    world = direct_native_world_grid(model, space, group_idx=offset + pose_index,
                                                     shape=shape, spacing_rc_mm=spacing, torch_module=torch)
                    target = staging / "planes" / stack / f"group{original_group:03d}.npz"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(target, **_predict_central(model, world, torch),
                                        world_train_ras=world.detach().cpu().numpy().astype(np.float32))
                    planes.append({"stack": stack, "derived_group_idx": derived_group,
                                   "original_group_idx": original_group, "global_pose_group_idx": offset + pose_index,
                                   "shape_rc": list(shape), "pixel_spacing_rc_mm": np.asarray(spacing, float).tolist(),
                                   "path": str(target.relative_to(staging)), "sha256": sha256(target)})
            offset += len(mapping)
        write_immutable_json(staging / "manifest.json", {
            "schema": SCHEMA, "subject": subject, "case_id": case, "endpoint": "central",
            "checkpoint": _record(checkpoint), "sparse_manifest": _record(sparse_path),
            "prepared_observations": {path.parent.name: _record(path) for path in paths},
            "coordinate_contract": "migrated sparse native local grid -> current checkpoint final learned rigid pose -> INR(world_train)",
            "coverage": "observed sparse planes only", "planes": planes,
            "no_training": True, "no_reconstruction": True, "no_formal_metric": True,
        })
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "prepared-root", "sparse-manifest", "subject", "case", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda:0")
    print(build_cyj11_r3_direct_final_pose_bundle(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
