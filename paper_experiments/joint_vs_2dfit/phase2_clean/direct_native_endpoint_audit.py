"""Read-only direct-native versus central-volume audit for canonical Q004.

The direct path deliberately recreates the historical
``export_native_plane_reprojections`` coordinate sequence: centred local grid,
``space.local_to_train``, then the checkpoint's learned rigid transform.  The
central-volume path samples its NIfTI at those exact world coordinates; it
never constructs a second grid from an R0 native affine.
"""
from __future__ import annotations

import argparse
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


def _load_model(checkpoint: Path, prepared_paths: list[Path], device_text: str) -> tuple[Any, Any, Any]:
    import torch
    from mlp.modules.module_07_objective_training.mlp_trainer import build_mlp_training_model
    from reconstruction_core.orchestration import load_checkpoint
    from trad.modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset

    device = torch.device(device_text)
    dataset = QuantPointDataset(prepared_paths, device=device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = payload["resolved_config"]
    protocol = config["decoder"].get("protocol_yaml", "trad/configs/protocol_hhz_v1.yaml")
    model, space, _, _ = build_mlp_training_model(dataset, config, protocol, device)
    load_checkpoint(checkpoint, model, device)
    model.eval()
    return model, space, torch


def _paths(prepared_root: Path) -> list[Path]:
    candidates = [[prepared_root / stack / "observations.npz" for stack in STACKS], [prepared_root / "CYJ" / stack / "observations.npz" for stack in STACKS]]
    paths = next((items for items in candidates if all(item.is_file() for item in items)), None)
    if paths is None: raise FileNotFoundError("direct-native audit requires cropped CYJ sax/2ch/4ch observations.npz")
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
    prepared_paths = _paths(prepared_root)
    model, space, torch = _load_model(checkpoint, prepared_paths, device)
    central = {"t1": nib.load(str(central_t1)), "t2": nib.load(str(central_t2))}
    l100 = {"t1": nib.load(str(l100_t1)), "t2": nib.load(str(l100_t2))}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        plane_records: list[dict[str, Any]] = []; group_offset = 0
        for stack, prepared_path in zip(STACKS, prepared_paths):
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
        write_immutable_json(staging / "manifest.json", {"schema": "q004_direct_native_endpoint_audit/v2", "checkpoint": _record(checkpoint), "prepared_observations": {_stack: _record(path) for _stack, path in zip(STACKS, prepared_paths)}, "central_volumes": {name: _record(path) for name, path in (("t1", Path(central_t1)), ("t2", Path(central_t2)))}, "frozen_l100_masks": {name: _record(path) for name, path in (("t1", Path(l100_t1)), ("t2", Path(l100_t2)))}, "coordinate_contract": "historical native local grid -> space.local_to_train -> final learned pose gives training-space INR coordinates; direct=model.inr(training_world), while NIfTI/mask sampling first applies space.train_ras_to_physical(training_world) then samples physical-RAS NIfTI", "direct_native_is_not_training": True, "central_volume_is_not_reconstruction": True, "no_training": True, "no_reconstruction": True, "no_formal_export_modified": True, "planes": plane_records})
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
