"""Read-only final-pose Proposed predictions for observed sparse native planes.

This is deliberately not an endpoint-volume reprojection.  The source-local
grid is transformed with the sparse reconstruction's final learned rigid pose,
using the exact active observation ordering recorded by the sparse manifest.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from quality_experiments.Q002_B6_roi_same_anchor_mse.contracts import profiled_observation_paths

from .contracts import sha256, write_immutable_json
from .direct_native_bundles import _predict_central
from .direct_native_endpoint_audit import _load_model, direct_native_world_grid


STACKS = ("sax", "2ch", "4ch")
SCHEMA = "direct_sparse_native_prediction_bundle/v1"


def _record(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _profile(path: Path, subject: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "phase4r_sparse_profile_v2" or payload.get("subject") != subject:
        raise ValueError("sparse manifest subject/schema mismatch")
    if payload.get("status") != "ELIGIBLE":
        raise ValueError("sparse manifest is not eligible")
    return payload


def _derived_to_original(profile: dict[str, Any], stack: str) -> list[int]:
    item = profile.get("stacks", {}).get(stack)
    records = item.get("derived_to_original_group_idx") if isinstance(item, dict) else None
    if not isinstance(records, list):
        raise ValueError(f"{stack}: sparse manifest lacks derived-to-original group mapping")
    pairs = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("derived_group_idx"), int) or not isinstance(record.get("original_group_idx"), int):
            raise ValueError(f"{stack}: malformed derived-to-original group mapping")
        pairs.append((int(record["derived_group_idx"]), int(record["original_group_idx"])))
    pairs.sort()
    if [value[0] for value in pairs] != list(range(len(pairs))):
        raise ValueError(f"{stack}: derived group IDs are not contiguous")
    return [value[1] for value in pairs]


def build_sparse_direct_native_prediction_bundle(*, checkpoint: str | Path, prepared_root: str | Path,
                                                 sparse_manifest: str | Path, subject: str,
                                                 endpoint: str, output: str | Path,
                                                 device: str = "cuda:0") -> Path:
    """Publish final-pose maps only for planes actually observed by a sparse run."""
    checkpoint, prepared_root, sparse_path, output = (Path(value) for value in (checkpoint, prepared_root, sparse_manifest, output))
    if endpoint != "central":
        raise ValueError("sparse direct-native bundle supports the formal central endpoint only")
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    if not checkpoint.is_file() or not sparse_path.is_file():
        raise FileNotFoundError("sparse direct-native bundle requires checkpoint and sparse manifest")
    profile = _profile(sparse_path, subject)
    paths = profiled_observation_paths(prepared_root, subject, sparse_path)
    active = [path.parent.name for path in paths]
    if active != [stack for stack in STACKS if profile["stacks"][stack].get("training_stack_present") is True]:
        raise ValueError("sparse active observation order disagrees with manifest")
    model, space, torch = _load_model(checkpoint, paths, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        records: list[dict[str, Any]] = []
        offset = 0
        for stack, path in zip(active, paths):
            original_ids = _derived_to_original(profile, stack)
            with np.load(path, allow_pickle=False) as observations:
                groups = np.asarray(observations["group_idx"], int)
                expected = list(range(len(original_ids)))
                if sorted(np.unique(groups).tolist()) != expected:
                    raise ValueError(f"{stack}: observed derived groups disagree with manifest")
                for local_group, original_group in enumerate(original_ids):
                    rows = np.flatnonzero(groups == local_group)
                    if not len(rows):
                        raise ValueError(f"{stack}: missing observed group {local_group}")
                    first = int(rows[0])
                    shape = tuple(int(value) for value in np.asarray(observations["images"])[first].shape)
                    spacing = np.asarray(observations["pixel_spacing_rc_mm"])[first]
                    world = direct_native_world_grid(model, space, group_idx=offset + local_group, shape=shape,
                                                     spacing_rc_mm=spacing, torch_module=torch)
                    prediction = _predict_central(model, world, torch)
                    target = staging / "planes" / stack / f"group{original_group:03d}.npz"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(target, **prediction,
                                        world_train_ras=world.detach().cpu().numpy().astype(np.float32))
                    records.append({"stack": stack, "derived_group_idx": local_group,
                                    "original_group_idx": original_group, "global_pose_group_idx": offset + local_group,
                                    "shape_rc": list(shape), "pixel_spacing_rc_mm": np.asarray(spacing, float).tolist(),
                                    "path": str(target.relative_to(staging)), "sha256": sha256(target)})
            offset += len(original_ids)
        write_immutable_json(staging / "manifest.json", {
            "schema": SCHEMA, "subject": subject, "case_id": profile["case_id"], "endpoint": endpoint,
            "checkpoint": _record(checkpoint), "sparse_manifest": _record(sparse_path),
            "prepared_observations": {stack: _record(path) for stack, path in zip(active, paths)},
            "coordinate_contract": "observed sparse native local grid -> space.local_to_train -> final learned rigid pose -> INR(world_train)",
            "coverage": "observed planes only; held-out planes are intentionally absent and must use original-native endpoint reprojection",
            "planes": records, "no_training": True, "no_reconstruction": True, "no_formal_export_modified": True,
        })
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--sparse-manifest", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--endpoint", choices=("central",), default="central")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    print(build_sparse_direct_native_prediction_bundle(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
