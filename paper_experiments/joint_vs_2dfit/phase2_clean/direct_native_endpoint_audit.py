"""Read-only direct-native versus central-volume audit for canonical Q004.

The direct path deliberately recreates the historical
``export_native_plane_reprojections`` coordinate sequence: centred local grid,
``space.local_to_train``, then the checkpoint's learned rigid transform.  The
central-volume path samples its NIfTI at those exact world coordinates; it
never constructs a second grid from an R0 native affine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import sha256, write_immutable_json
from .evaluation import _metrics, _world_sample
from .final_cyj_reporting import _load_all_verified_native_cache_groups


STACKS = ("sax", "2ch", "4ch")
PARAMETERS = {"t1": "t1_ms", "t2": "t2_ms"}
FINAL5_SUBJECTS = ("CYJ", "DYL_2", "DYZ", "HHZ", "HJL")


def historical_local_grid(height: int, width: int, spacing_rc_mm: np.ndarray, torch_module: Any, device: Any) -> Any:
    """Exact local pixel-centre layout used by historical native export."""
    rows, cols = np.mgrid[0:height, 0:width]
    return torch_module.as_tensor(np.column_stack(((cols.ravel() - (width - 1) / 2.0) * float(spacing_rc_mm[1]), (rows.ravel() - (height - 1) / 2.0) * float(spacing_rc_mm[0]), np.zeros(height * width))), dtype=torch_module.float32, device=device)


def direct_native_world_grid(model: Any, space: Any, *, group_idx: int, shape: tuple[int, int], spacing_rc_mm: np.ndarray, torch_module: Any) -> Any:
    """World grid from the final learned Q004 pose, not from a native affine."""
    device = next(model.parameters()).device
    local = historical_local_grid(shape[0], shape[1], spacing_rc_mm, torch_module, device)
    train = space.local_to_train(local)
    groups = torch_module.full((train.shape[0],), int(group_idx), dtype=torch_module.long, device=device)
    return model.rigid_psf.transform_local_to_ras(train, groups).reshape(shape[0], shape[1], 3)


def exact_shared_support(reference: np.ndarray, source_support: np.ndarray, l100_native: np.ndarray, direct: np.ndarray, volume: np.ndarray, direct_valid: np.ndarray, volume_valid: np.ndarray) -> np.ndarray:
    """One paired support; only method computation differs between columns."""
    arrays = (reference, source_support, l100_native, direct, volume, direct_valid, volume_valid)
    if len({np.asarray(item).shape for item in arrays}) != 1:
        raise ValueError("direct-native audit arrays must share one plane shape")
    return (np.asarray(source_support, bool) & np.asarray(l100_native, bool) & np.asarray(direct_valid, bool) & np.asarray(volume_valid, bool)
            & np.isfinite(reference) & np.isfinite(direct) & np.isfinite(volume))


def _world_homogeneous(world_xyz: np.ndarray) -> np.ndarray:
    world_xyz = np.asarray(world_xyz, float)
    return np.concatenate((np.moveaxis(world_xyz, -1, 0), np.ones((1, *world_xyz.shape[:2]))), axis=0)


def _binary_world_sample(mask: np.ndarray, affine: np.ndarray, world_xyz: np.ndarray) -> np.ndarray:
    values, inside = _world_sample(np.asarray(mask, np.uint8), affine, _world_homogeneous(world_xyz), order=0)
    return np.asarray(inside, bool) & np.isfinite(values) & (values > .5)


def train_world_to_physical_ras(space: Any, world_train: np.ndarray, torch_module: Any) -> np.ndarray:
    """Convert the historical direct-INR training coordinates to physical RAS-mm."""
    train = torch_module.as_tensor(world_train, dtype=torch_module.float32)
    return space.train_ras_to_physical(train).detach().cpu().numpy()


def sample_exported_volume_at_train_world(volume: np.ndarray, affine_ras_mm: np.ndarray, world_train: np.ndarray, space: Any, torch_module: Any, *, order: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Sample a physical-RAS NIfTI at native points represented in training space."""
    physical_ras = train_world_to_physical_ras(space, world_train, torch_module)
    return _world_sample(volume, affine_ras_mm, _world_homogeneous(physical_ras), order=order)


def _model_prediction(model: Any, world_xyz: np.ndarray, parameter_key: str, torch_module: Any) -> np.ndarray:
    device = next(model.parameters()).device
    world = torch_module.as_tensor(world_xyz.reshape(-1, 3), dtype=torch_module.float32, device=device)
    parts: list[np.ndarray] = []
    with torch_module.no_grad():
        for points in world.split(65536):
            parts.append(model.inr(points)[parameter_key].detach().cpu().numpy())
    return np.concatenate(parts).reshape(world_xyz.shape[:2])


def _prepared_inputs_sha256(paths: list[Path]) -> str:
    """Match the immutable training-manifest digest, without trusting paths."""
    return hashlib.sha256("".join(sha256(path) for path in paths).encode("ascii")).hexdigest()


def _checkpoint_manifest(checkpoint: Path) -> dict[str, Any]:
    manifest_path = checkpoint.parent / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("direct-native audit requires checkpoint experiment_manifest.json for prepared-input identity")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("checkpoint experiment manifest must be an object")
    return manifest


def _stack_hashes(value: Any, *, label: str, active_stacks: tuple[str, ...]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint experiment manifest {label} must be a path-to-SHA256 mapping")
    result: dict[str, str] = {}
    for path_text, digest in value.items():
        stack = Path(str(path_text)).parent.name.lower()
        if stack not in active_stacks or stack in result or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"checkpoint experiment manifest {label} has invalid stack/SHA256 provenance")
        result[stack] = digest
    if tuple(result) != active_stacks:
        raise ValueError(f"checkpoint experiment manifest {label} does not match canonical active stack identity")
    return result


def _sparse_contract(path: Path) -> dict[str, Any]:
    """Read only the sparse identity and manifest-declared active stack subset."""
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "ELIGIBLE":
        raise ValueError("direct-native audit requires an eligible sparse manifest")
    schema = payload.get("schema_version")
    if schema == "phase4r_sparse_v1":
        subject, condition, phase, generator = payload.get("subject"), payload.get("condition"), payload.get("phase"), payload.get("git_sha")
        if not isinstance(subject, str) or condition != "L100" or phase != 0 or not isinstance(generator, str) or len(generator) != 40:
            raise ValueError("direct-native audit accepts phase4r_sparse_v1 only for L100_fold0")
        stacks = payload.get("stacks")
        if not isinstance(stacks, Mapping) or set(stacks) != set(STACKS):
            raise ValueError("sparse manifest lacks exact sax/2ch/4ch stack records")
        return {"path": path.resolve(), "sha256": sha256(path), "subject": subject,
                "case": "L100", "generator_git_sha": generator, "active_stacks": STACKS,
                "schema_version": schema}
    if schema != "phase4r_sparse_profile_v2":
        raise ValueError("direct-native audit requires phase4r_sparse_profile_v2 or L100 phase4r_sparse_v1")
    subject, case, generator = payload.get("subject"), payload.get("case_id"), payload.get("git_sha")
    if not isinstance(subject, str) or not isinstance(case, str) or not isinstance(generator, str) or len(generator) != 40:
        raise ValueError("sparse manifest lacks subject/case/generator provenance")
    stacks = payload.get("stacks")
    if not isinstance(stacks, Mapping) or set(stacks) != set(STACKS):
        raise ValueError("sparse manifest lacks exact sax/2ch/4ch stack records")
    active = tuple(stack for stack in STACKS if isinstance(stacks[stack], Mapping) and stacks[stack].get("training_stack_present") is True)
    if not active or (case == "L100" and active != STACKS):
        raise ValueError("sparse manifest active stacks are incompatible with the requested case")
    return {"path": path.resolve(), "sha256": sha256(path), "subject": subject, "case": case,
            "generator_git_sha": generator, "active_stacks": active, "schema_version": schema}


def _validate_checkpoint_sparse_provenance(checkpoint: Path, sparse: Mapping[str, Any]) -> None:
    """Bind the audit's supplied sparse profile to Q002's recorded profile identity."""
    manifest = _checkpoint_manifest(checkpoint)
    if manifest.get("subject_id") != sparse["subject"]:
        raise ValueError("checkpoint sparse subject/case provenance mismatch")
    if sparse["schema_version"] == "phase4r_sparse_v1":
        return
    recorded = manifest.get("sparse_manifest")
    if not isinstance(recorded, Mapping):
        raise ValueError("checkpoint experiment manifest lacks sparse manifest provenance")
    if manifest.get("case_id") != sparse["case"]:
        raise ValueError("checkpoint sparse subject/case provenance mismatch")
    if recorded.get("sha256") != sparse["sha256"]:
        raise ValueError("checkpoint sparse manifest SHA256 mismatch")
    if recorded.get("generator_git_sha") != sparse["generator_git_sha"]:
        raise ValueError("checkpoint sparse generator provenance mismatch")


def _validate_prepared_input_identity(checkpoint: Path, prepared_paths: list[Path],
                                      active_stacks: tuple[str, ...] = STACKS) -> None:
    """Bind local prepared inputs to old or Q002 manifest records by stack hash."""
    if not active_stacks or any(stack not in STACKS for stack in active_stacks) or tuple(stack for stack in STACKS if stack in active_stacks) != active_stacks:
        raise ValueError("active stacks must be a non-empty canonical sax/2ch/4ch subset")
    actual_stacks = tuple(path.parent.name.lower() for path in prepared_paths)
    if actual_stacks != active_stacks:
        raise ValueError("prepared inputs do not match canonical active stack identity")
    manifest = _checkpoint_manifest(checkpoint)
    actual = {stack: sha256(path) for stack, path in zip(active_stacks, prepared_paths)}
    sources: list[dict[str, str]] = []
    for field in ("prepared_inputs_sha256", "training_input_sha256"):
        if field in manifest:
            sources.append(_stack_hashes(manifest[field], label=field, active_stacks=active_stacks))
    declared, aggregate = manifest.get("prepared_inputs"), manifest.get("prepared_manifest_sha256")
    if declared is not None or aggregate is not None:
        if not isinstance(declared, list) or not isinstance(aggregate, str):
            raise ValueError("checkpoint experiment manifest legacy prepared-input provenance is incomplete")
        declared_stacks = tuple(Path(str(path)).parent.name.lower() for path in declared)
        if declared_stacks != active_stacks:
            raise ValueError("checkpoint legacy prepared-input stack identity does not match active stacks")
        if _prepared_inputs_sha256(prepared_paths) != aggregate:
            raise ValueError("checkpoint prepared-input SHA256 mismatch; local inputs are not the original training inputs")
        if sources and hashlib.sha256("".join(sources[0][stack] for stack in active_stacks).encode("ascii")).hexdigest() != aggregate:
            raise ValueError("checkpoint legacy and per-input prepared SHA256 provenance disagree")
    if not sources and declared is None:
        raise ValueError("checkpoint experiment manifest lacks verifiable prepared-input SHA256 provenance")
    for expected in sources:
        if expected != actual:
            raise ValueError("checkpoint prepared-input SHA256 mismatch; local inputs are not the original training inputs")
    if len(sources) == 2 and sources[0] != sources[1]:
        raise ValueError("checkpoint prepared_inputs_sha256 and training_input_sha256 provenance disagree")


def _checkpoint_initial_pose(payload: Mapping[str, Any], torch_module: Any, device: Any) -> Any:
    """Recover the exact training pose used to derive the hash-grid bbox."""
    training_space = payload.get("training_space")
    candidates = (
        payload.get("group_axisangle_post_stack_init_physical"),
        payload.get("group_axisangle_init_physical"),
        training_space.get("group_axisangle_init_physical") if isinstance(training_space, Mapping) else None,
    )
    pose = next((value for value in candidates if value is not None), None)
    if pose is None:
        raise ValueError("checkpoint lacks post-stack and legacy initial physical group poses")
    pose = torch_module.as_tensor(pose, dtype=torch_module.float32, device=device)
    if pose.ndim != 2 or pose.shape[1] != 6:
        raise ValueError("checkpoint initial physical group pose must have shape [groups, 6]")
    return pose


def _validate_checkpoint_model_shapes(model: Any, checkpoint_state: Any) -> None:
    """Fail before strict checkpoint loading if reconstruction topology differs."""
    if not isinstance(checkpoint_state, Mapping):
        raise ValueError("checkpoint model_state must be a mapping")
    current = model.state_dict()
    missing, unexpected = set(current).difference(checkpoint_state), set(checkpoint_state).difference(current)
    if missing or unexpected:
        raise ValueError(f"checkpoint model-state keys differ before strict load; missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    mismatches = [(key, tuple(current[key].shape), tuple(checkpoint_state[key].shape))
                  for key in current if tuple(current[key].shape) != tuple(checkpoint_state[key].shape)]
    if mismatches:
        encoding = [item for item in mismatches if item[0] in {
            "inr.encoding.params", "forward_model.quantitative_inr.encoding.params"}]
        if encoding:
            detail = "; ".join(f"{key}: checkpoint={saved}, rebuilt={rebuilt}" for key, rebuilt, saved in encoding)
            raise ValueError(f"checkpoint encoding parameter shape mismatch before strict load ({detail})")
        detail = "; ".join(f"{key}: checkpoint={saved}, rebuilt={rebuilt}" for key, rebuilt, saved in mismatches[:8])
        raise ValueError(f"checkpoint model parameter shape mismatch before strict load ({detail})")


def _validate_rebuilt_training_space(rebuilt: Any, saved: Any, torch_module: Any) -> None:
    """Require the rebuilt bbox/geometry to match the checkpoint coordinate contract."""
    if not isinstance(saved, Mapping):
        raise ValueError("checkpoint training_space must be a mapping")
    current = rebuilt.state_dict()
    keys = ("center_ras_mm", "spatial_scaling", "physical_bbox_ras_mm", "bbox_train",
            "group_resolution_xyz_mm", "group_axisangle_init_physical", "group_axisangle_init_train")
    missing = set(keys).difference(saved) | set(keys).difference(current)
    if missing:
        raise ValueError(f"checkpoint/rebuilt training_space lacks: {sorted(missing)}")
    for key in keys:
        expected = torch_module.as_tensor(saved[key]).detach().cpu()
        actual = torch_module.as_tensor(current[key]).detach().cpu()
        equal = expected.shape == actual.shape and torch_module.allclose(expected, actual, rtol=0, atol=1e-6)
        if not equal:
            kind = "bounding-box" if key in {"physical_bbox_ras_mm", "bbox_train"} else "geometry"
            raise ValueError(f"checkpoint/rebuilt training-space {kind} mismatch for {key}; prepared inputs or initial pose differ")


def _load_model(checkpoint: Path, prepared_paths: list[Path], device_text: str,
                *, sparse_contract: Mapping[str, Any] | None = None) -> tuple[Any, Any, Any]:
    import torch
    from mlp.modules.module_07_objective_training.mlp_trainer import build_mlp_training_model
    from reconstruction_core.orchestration import load_checkpoint
    from trad.modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset
    from trad.modules.module_07_objective_training.training_space import TrainingSpace

    device = torch.device(device_text)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must be a mapping")
    active_stacks = tuple(sparse_contract["active_stacks"]) if sparse_contract is not None else STACKS
    _validate_prepared_input_identity(checkpoint, prepared_paths, active_stacks)
    if sparse_contract is not None:
        _validate_checkpoint_sparse_provenance(checkpoint, sparse_contract)
    dataset = QuantPointDataset(prepared_paths, device=device)
    config = payload["resolved_config"]
    protocol = config["decoder"].get("protocol_yaml", "trad/configs/protocol_hhz_v1.yaml")
    initial_pose = _checkpoint_initial_pose(payload, torch, device)
    model, rebuilt_space, _, _ = build_mlp_training_model(
        dataset, config, protocol, device, initial_group_axisangle_physical=initial_pose)
    _validate_rebuilt_training_space(rebuilt_space, payload.get("training_space"), torch)
    _validate_checkpoint_model_shapes(model, payload.get("model_state"))
    load_checkpoint(checkpoint, model, device)
    model.eval()
    return model, TrainingSpace.from_state_dict(payload["training_space"]), torch


def _paths(prepared_root: Path, *, active_stacks: tuple[str, ...] = STACKS) -> list[Path]:
    candidates = [[prepared_root / stack / "observations.npz" for stack in active_stacks]]
    candidates.extend([prepared_root / subject / stack / "observations.npz" for stack in active_stacks] for subject in FINAL5_SUBJECTS)
    matches = [items for items in candidates if all(item.is_file() for item in items)]
    if len(matches) != 1:
        if not matches:
            raise FileNotFoundError("direct-native audit requires manifest-active cropped observations.npz at the supplied root or one explicit Final5 subject directory")
        raise ValueError("direct-native audit prepared root is ambiguous; provide one subject-specific prepared directory")
    paths = matches[0]
    if "full_fov" in str(prepared_root).lower(): raise ValueError("direct-native audit rejects full-FOV inputs")
    return paths


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _plot(path: Path, reference: np.ndarray, direct: np.ndarray, volume: np.ndarray, support: np.ndarray, parameter: str) -> None:
    import matplotlib.pyplot as plt
    limits = (0, 2500) if parameter == "t1" else (0, 200)
    figure, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    panels = (("Native", reference), ("Direct model.inr(world)", direct), ("Central NIfTI sampled at same world", volume), ("|direct - volume|", np.abs(direct - volume)))
    for axis, (title, values) in zip(axes, panels):
        axis.imshow(np.ma.masked_invalid(np.where(support, values, np.nan)), cmap="magma" if title.startswith("|") else "viridis", vmin=0, vmax=limits[1], interpolation="none")
        axis.set_title(title, fontsize=8); axis.set_xticks([]); axis.set_yticks([])
    figure.suptitle(f"{parameter.upper()} — exact shared L100 native support", fontsize=10); figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)


def run_direct_native_endpoint_audit(*, checkpoint: str | Path, prepared_root: str | Path, r0_native_maps_root: str | Path, sparse_manifest: str | Path, central_t1: str | Path, central_t2: str | Path, l100_t1: str | Path, l100_t2: str | Path, output: str | Path, device: str = "cuda:0") -> Path:
    """Create a fresh audit directory; no training or formal export is modified."""
    import nibabel as nib

    checkpoint, prepared_root, output = Path(checkpoint), Path(prepared_root), Path(output)
    required = (checkpoint, Path(sparse_manifest), Path(central_t1), Path(central_t2), Path(l100_t1), Path(l100_t2))
    if not all(path.is_file() for path in required): raise FileNotFoundError("direct-native audit requires every explicit checkpoint/map/mask input")
    if output.exists(): raise FileExistsError(f"refusing overwrite: {output}")
    sparse_contract = _sparse_contract(Path(sparse_manifest))
    prepared_paths = _paths(prepared_root, active_stacks=sparse_contract["active_stacks"])
    model, space, torch = _load_model(checkpoint, prepared_paths, device, sparse_contract=sparse_contract)
    central = {"t1": nib.load(str(central_t1)), "t2": nib.load(str(central_t2))}
    l100 = {"t1": nib.load(str(l100_t1)), "t2": nib.load(str(l100_t2))}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        plane_records: list[dict[str, Any]] = []; group_offset = 0
        for stack, prepared_path in zip(sparse_contract["active_stacks"], prepared_paths):
            cached = _load_all_verified_native_cache_groups(r0_native_maps_root, sparse_manifest, stack)
            with np.load(prepared_path, allow_pickle=False) as observations:
                group_ids = [int(value) for value in np.unique(observations["group_idx"])]
                if group_ids != [int(value) for value in cached["group_idx"]]: raise ValueError(f"{stack}: prepared and R0 group identities differ")
                for local_group in group_ids:
                    row = int(np.flatnonzero(observations["group_idx"] == local_group)[0])
                    shape = tuple(int(value) for value in observations["images"][row].shape)
                    spacing = np.asarray(observations["pixel_spacing_rc_mm"][row], float)
                    world = direct_native_world_grid(model, space, group_idx=group_offset + local_group, shape=shape, spacing_rc_mm=spacing, torch_module=torch).detach().cpu().numpy()
                    cache_index = [int(value) for value in cached["group_idx"]].index(local_group)
                    source_support = np.asarray(cached["source_support"][cache_index], bool)
                    if source_support.shape != shape: raise ValueError(f"{stack} group {local_group}: source support shape mismatch")
                    for parameter, field in PARAMETERS.items():
                        reference = np.asarray(cached[field][cache_index], float)
                        direct = _model_prediction(model, world, field, torch)
                        volume, volume_valid = sample_exported_volume_at_train_world(central[parameter].get_fdata(), central[parameter].affine, world, space, torch, order=1)
                        l100_native = _binary_world_sample(l100[parameter].get_fdata() > .5, l100[parameter].affine, train_world_to_physical_ras(space, world, torch))
                        support = exact_shared_support(reference, source_support, l100_native, direct, volume, np.ones(shape, bool), volume_valid)
                        target = staging / "planes" / parameter / stack / f"group{local_group:03d}.npz"; target.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(target, reference=reference, direct_native=direct, central_volume=volume, source_support=source_support, l100_native=l100_native, shared_support=support, world_train_ras=world)
                        figure = staging / "figures" / parameter / stack / f"group{local_group:03d}.png"; figure.parent.mkdir(parents=True, exist_ok=True); _plot(figure, reference, direct, volume, support, parameter)
                        plane_records.append({"parameter": parameter, "stack": stack, "group_idx": local_group, "global_pose_group_idx": group_offset + local_group, "output": str(target.relative_to(staging)), "figure": str(figure.relative_to(staging)), "shared_support_voxels": int(support.sum()), "direct_native_vs_reference": _metrics(reference, direct, support) if support.any() else None, "central_volume_vs_reference": _metrics(reference, volume, support) if support.any() else None, "direct_native_vs_central_volume": _metrics(direct, volume, support) if support.any() else None})
            group_offset += len(group_ids)
        write_immutable_json(staging / "manifest.json", {"schema": "q004_direct_native_endpoint_audit/v2", "checkpoint": _record(checkpoint), "prepared_observations": {_stack: _record(path) for _stack, path in zip(sparse_contract["active_stacks"], prepared_paths)}, "sparse_manifest": _record(Path(sparse_manifest)), "active_stacks": list(sparse_contract["active_stacks"]), "central_volumes": {name: _record(path) for name, path in (("t1", Path(central_t1)), ("t2", Path(central_t2)))}, "frozen_l100_masks": {name: _record(path) for name, path in (("t1", Path(l100_t1)), ("t2", Path(l100_t2)))}, "coordinate_contract": "historical native local grid -> space.local_to_train -> final learned pose gives training-space INR coordinates; direct=model.inr(training_world), while NIfTI/mask sampling first applies space.train_ras_to_physical(training_world) then samples physical-RAS NIfTI", "direct_native_is_not_training": True, "central_volume_is_not_reconstruction": True, "no_training": True, "no_reconstruction": True, "no_formal_export_modified": True, "planes": plane_records})
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True); raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "prepared-root", "r0-native-maps-root", "sparse-manifest", "central-t1", "central-t2", "l100-t1", "l100-t2", "output"): parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", default="cuda:0")
    print(run_direct_native_endpoint_audit(**{key.replace("-", "_"): value for key, value in vars(parser.parse_args()).items()}))


if __name__ == "__main__": main()
