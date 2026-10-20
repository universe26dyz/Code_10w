"""Immutable direct-native central and PSF128 predictions on the canonical L100 lattice."""
from __future__ import annotations

import argparse
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from trad.modules.module_08_inference_export.export_quantitative import (
    FORMAL_PSF128_OUTPUT_PSF_FACTOR,
    FORMAL_PSF128_OUTPUT_RESOLUTION_MM,
    FORMAL_PSF128_SEED,
    FORMAL_PSF128_SAMPLES,
)
from trad.third_party.nesvor.nesvor.utils import resolution2sigma

from .contracts import sha256, write_immutable_json
from .direct_native_endpoint_audit import STACKS, _load_model, _paths, _record, direct_native_world_grid


@dataclass(frozen=True)
class NativeLatticePlane:
    stack: str
    prepared_path: Path
    original_group_idx: int
    global_pose_group_idx: int
    shape_rc: tuple[int, int]
    spacing_rc_mm: np.ndarray
    global_query_start: int

    @property
    def global_query_indices(self) -> np.ndarray:
        return np.arange(self.global_query_start, self.global_query_start + int(np.prod(self.shape_rc)), dtype=np.int64).reshape(self.shape_rc)


def canonical_l100_native_lattice(prepared_paths: Sequence[str | Path]) -> list[NativeLatticePlane]:
    """Return the complete, request-independent canonical L100 source lattice."""
    if len(prepared_paths) != len(STACKS):
        raise ValueError("canonical L100 lattice requires sax, 2ch, and 4ch prepared inputs")
    lattice: list[NativeLatticePlane] = []
    query_start, group_offset = 0, 0
    for stack, prepared_text in zip(STACKS, prepared_paths):
        prepared = Path(prepared_text)
        with np.load(prepared, allow_pickle=False) as data:
            groups = sorted(int(value) for value in np.unique(data["group_idx"]))
            for local_position, group in enumerate(groups):
                rows = np.flatnonzero(np.asarray(data["group_idx"], int) == group)
                if not len(rows):
                    raise ValueError(f"{prepared}: missing declared group {group}")
                images = np.asarray(data["images"])[rows]
                spacing = np.asarray(data["pixel_spacing_rc_mm"])[rows]
                if len({tuple(item.shape) for item in images}) != 1 or len({tuple(item) for item in spacing}) != 1:
                    raise ValueError(f"{prepared}: group {group} has inconsistent native geometry across weights")
                shape = tuple(int(value) for value in images[0].shape)
                lattice.append(NativeLatticePlane(stack, prepared, group, group_offset + local_position, shape, np.asarray(spacing[0], float), query_start))
                query_start += int(np.prod(shape))
            group_offset += len(groups)
    return lattice


def draw_psf128_physical_offsets(rng: np.random.Generator, query_count: int) -> np.ndarray:
    """Draw the formal K=128 physical-RAS offsets for consecutive global queries."""
    if query_count < 0:
        raise ValueError("query_count must be non-negative")
    sigma_mm = float(resolution2sigma(FORMAL_PSF128_OUTPUT_RESOLUTION_MM * FORMAL_PSF128_OUTPUT_PSF_FACTOR, isotropic=True))
    return rng.standard_normal((query_count, FORMAL_PSF128_SAMPLES, 3), dtype=np.float32) * sigma_mm


def _predict_central(model: Any, world_train: Any, torch_module: Any) -> dict[str, np.ndarray]:
    output = {"t1_ms": [], "t2_ms": []}
    with torch_module.no_grad():
        for points in world_train.reshape(-1, 3).split(65536):
            fields = model.inr(points)
            for field in output:
                output[field].append(fields[field].detach().cpu().numpy())
    return {field: np.concatenate(values).reshape(tuple(world_train.shape[:2])).astype(np.float32) for field, values in output.items()}


def _predict_psf128(model: Any, space: Any, world_train: Any, torch_module: Any, rng: np.random.Generator, *, query_batch_size: int) -> dict[str, np.ndarray]:
    if query_batch_size <= 0:
        raise ValueError("query_batch_size must be positive")
    centers_physical = space.train_ras_to_physical(world_train.reshape(-1, 3)).detach()
    output = {"t1_ms": [], "t2_ms": []}
    with torch_module.no_grad():
        for start in range(0, centers_physical.shape[0], query_batch_size):
            centers = centers_physical[start:start + query_batch_size]
            offsets = torch_module.from_numpy(draw_psf128_physical_offsets(rng, int(centers.shape[0]))).to(device=centers.device, dtype=centers.dtype)
            fields = model.inr(space.physical_ras_to_train(centers[:, None, :] + offsets))
            for field in output:
                output[field].append(fields[field].mean(-1).detach().cpu().numpy())
    return {field: np.concatenate(values).reshape(tuple(world_train.shape[:2])).astype(np.float32) for field, values in output.items()}


def build_direct_native_prediction_bundle(*, checkpoint: str | Path, prepared_root: str | Path, endpoint: str, output: str | Path, device: str = "cuda:0", query_batch_size: int = 8192) -> Path:
    """Publish one complete canonical-L100 direct-native endpoint bundle without training."""
    checkpoint, prepared_root, output = Path(checkpoint), Path(prepared_root), Path(output)
    if endpoint not in {"central", "psf128"}:
        raise ValueError("endpoint must be central or psf128")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    prepared_paths = _paths(prepared_root)
    lattice = canonical_l100_native_lattice(prepared_paths)
    model, space, torch = _load_model(checkpoint, prepared_paths, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    rng = np.random.default_rng(FORMAL_PSF128_SEED) if endpoint == "psf128" else None
    try:
        records: list[dict[str, Any]] = []
        for plane in lattice:
            world_train = direct_native_world_grid(model, space, group_idx=plane.global_pose_group_idx, shape=plane.shape_rc, spacing_rc_mm=plane.spacing_rc_mm, torch_module=torch)
            predictions = _predict_central(model, world_train, torch) if endpoint == "central" else _predict_psf128(model, space, world_train, torch, rng, query_batch_size=query_batch_size)
            target = staging / "planes" / plane.stack / f"group{plane.original_group_idx:03d}.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **predictions, global_query_index=plane.global_query_indices, world_train_ras=world_train.detach().cpu().numpy().astype(np.float32))
            records.append({"stack": plane.stack, "original_group_idx": plane.original_group_idx, "global_pose_group_idx": plane.global_pose_group_idx, "shape_rc": list(plane.shape_rc), "pixel_spacing_rc_mm": plane.spacing_rc_mm.tolist(), "global_query_start": plane.global_query_start, "global_query_stop_exclusive": plane.global_query_start + int(np.prod(plane.shape_rc)), "path": str(target.relative_to(staging)), "sha256": sha256(target)})
        manifest: dict[str, Any] = {"schema": "direct_native_prediction_bundle/v1", "endpoint": endpoint, "checkpoint": _record(checkpoint), "prepared_observations": {stack: _record(path) for stack, path in zip(STACKS, prepared_paths)}, "coordinate_contract": "historical local grid -> space.local_to_train -> final learned pose gives training-space INR coordinates", "canonical_l100_lattice": "sax -> 2ch -> 4ch; original_group_idx ascending within stack; pixel C-order", "query_identity": "global_query_index is fixed by the complete canonical L100 lattice and never by a requested sparse subset", "planes": records, "no_training": True, "no_reconstruction": True, "no_formal_export_modified": True}
        if endpoint == "psf128":
            manifest["psf128_contract"] = {"seed": FORMAL_PSF128_SEED, "numpy_rng": "numpy.random.Generator(PCG64) via numpy.random.default_rng", "n_inference_samples": FORMAL_PSF128_SAMPLES, "output_resolution_mm": FORMAL_PSF128_OUTPUT_RESOLUTION_MM, "output_psf_factor": FORMAL_PSF128_OUTPUT_PSF_FACTOR, "sigma_physical_mm": float(resolution2sigma(FORMAL_PSF128_OUTPUT_RESOLUTION_MM * FORMAL_PSF128_OUTPUT_PSF_FACTOR, isotropic=True)), "operator": "physical-RAS isotropic Gaussian offsets -> physical_ras_to_train -> model.inr -> mean T1/T2 in map domain", "offset_reuse": "one RNG stream in canonical global_query_index order; Q002/Q004 and T1/T2 use the same physical offsets; sparse conditions select from this complete bundle"}
        write_immutable_json(staging / "manifest.json", manifest)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--endpoint", required=True, choices=("central", "psf128")); parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--query-batch-size", type=int, default=8192)
    print(build_direct_native_prediction_bundle(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
