"""Read-only D1 input loaders and output/provenance helpers."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from trad.evaluation.scmr.decoder_pair import PARAMETERS, STACKS, load_decoder_pair, validate_verified_reference
from trad.evaluation.scmr.reference_2d import load_verified_reference, sha256


@dataclass(frozen=True)
class MapPlane:
    values: np.ndarray
    support: np.ndarray
    source: Path


@dataclass(frozen=True)
class D1Inputs:
    subject_id: str
    arrays: dict[str, dict[str, dict[str, MapPlane]]]
    native_manifest: Path
    provenance: dict[str, Any]


def ensure_empty_output(output: str | Path) -> Path:
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def prepare_output_layout(output: str | Path, command: list[str]) -> Path:
    """Create the required self-contained D1 result layout after overwrite validation."""

    output = ensure_empty_output(output)
    for relative in ("metrics", "figures/representative", "figures/all_slices", "figures/myocardium", "artifacts", "logs", "commands"):
        (output / relative).mkdir(parents=True, exist_ok=True)
    (output / "commands" / "command.txt").write_text(" ".join(shlex.quote(value) for value in command) + "\n", encoding="utf-8")
    (output / "logs" / "run.log").write_text("D1 runner started; this file is replaced with the full server tee log when run_D1_CYJ.sh is used.\n", encoding="utf-8")
    return output


def validate_stack_shapes(stacks: dict[str, np.ndarray]) -> None:
    """Validate each stack independently; heterogeneous in-plane shapes are allowed."""

    if tuple(stacks) != STACKS:
        raise ValueError(f"Expected exactly stack keys {STACKS}.")
    for stack, values in stacks.items():
        if np.asarray(values).ndim != 3:
            raise ValueError(f"{stack} must be a [group,row,column] array.")


def _load_central_plane(run_root: Path, stack: str, parameter: str, native: np.ndarray) -> MapPlane:
    path = run_root / f"t1_t2_native_plane_{stack}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"D1 central-plane archive is missing: {path}")
    field = "t1_ms" if parameter == "T1" else "t2_ms"
    with np.load(path, allow_pickle=False) as archive:
        required = {field, "masks", "group_idx", "weight_idx"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing required keys {sorted(missing)}")
        values = np.asarray(archive[field], dtype=np.float32)
        masks = np.asarray(archive["masks"], dtype=bool)
        groups = np.asarray(archive["group_idx"], dtype=np.int64)
        weights = np.asarray(archive["weight_idx"], dtype=np.int64)
    if values.ndim != 3 or values.shape != masks.shape or groups.shape != (values.shape[0],) or weights.shape != (values.shape[0],):
        raise ValueError(f"{path} has invalid D1 central-plane schema.")
    expected = np.arange(native.shape[0], dtype=np.int64)
    if not np.array_equal(np.unique(groups), expected):
        raise ValueError(f"{path} group_idx must contain the complete zero-based native group set.")
    selected_values, selected_support = [], []
    for group in expected:
        rows = np.flatnonzero(groups == group)
        if rows.size != 10:
            raise ValueError(f"{path} group {group} must contain exactly 10 observation rows.")
        if not np.array_equal(np.sort(weights[rows]), np.arange(10, dtype=np.int64)):
            raise ValueError(f"{path} group {group} must contain exactly one row for every weight 0..9.")
        weight_zero = rows[weights[rows] == 0]
        if weight_zero.size != 1:
            raise ValueError(f"{path} group {group} must contain a unique weight-0 central-plane row.")
        selected_values.append(values[int(weight_zero[0])])
        selected_support.append(np.all(masks[rows], axis=0))
    central_values, central_support = np.stack(selected_values), np.stack(selected_support)
    if central_values.shape != native.shape:
        raise ValueError(f"{path} {parameter}/{stack} shape={central_values.shape}, expected native shape={native.shape}; no resize is permitted.")
    return MapPlane(values=central_values, support=central_support & np.isfinite(central_values), source=path)


def load_d1_inputs(
    *, subject_id: str, trad_run: str | Path, mlp_run: str | Path, trad_eval: str | Path, mlp_eval: str | Path,
    native_reference_root: str | Path, preprocessed_root: str | Path,
) -> D1Inputs:
    """Load only pre-existing baseline artifacts and validate their shared provenance."""

    reference = load_verified_reference(native_reference_root, subject_id, preprocessed_root)
    pair = load_decoder_pair(subject_id, trad_eval, mlp_eval)
    native_hash = sha256(reference.manifest_path)
    validate_verified_reference(pair, reference, native_hash)
    arrays: dict[str, dict[str, dict[str, MapPlane]]] = {}
    for parameter, native_field in (("T1", "t1_ms"), ("T2", "t2_ms")):
        arrays[parameter] = {}
        for stack in STACKS:
            native = np.asarray(getattr(reference.stacks[stack], native_field), dtype=np.float32)
            native_valid = np.asarray(reference.stacks[stack].valid_mask, dtype=bool) & np.isfinite(native)
            bloch_psf = pair.trad.arrays[parameter][stack]
            mlp_psf = pair.mlp.arrays[parameter][stack]
            for label, item in (("Bloch PSF", bloch_psf), ("FrozenMLP PSF", mlp_psf)):
                if item["prediction"].shape != native.shape or item["support"].shape != native.shape:
                    raise ValueError(f"{parameter}/{stack} {label} does not match native reference shape; no resize is permitted.")
            arrays[parameter][stack] = {
                "native": MapPlane(native, native_valid, reference.stacks[stack].path),
                "central_bloch": _load_central_plane(Path(trad_run), stack, parameter, native),
                "psf_bloch": MapPlane(np.asarray(bloch_psf["prediction"], dtype=np.float32), np.asarray(bloch_psf["support"], dtype=bool), pair.trad.root / f"{parameter}_native_map_domain_psf_comparison.npz"),
                "central_mlp": _load_central_plane(Path(mlp_run), stack, parameter, native),
                "psf_mlp": MapPlane(np.asarray(mlp_psf["prediction"], dtype=np.float32), np.asarray(mlp_psf["support"], dtype=bool), pair.mlp.root / f"{parameter}_native_map_domain_psf_comparison.npz"),
            }
    provenance = {
        "native_reference_manifest": str(reference.manifest_path),
        "native_reference_manifest_sha256": native_hash,
        "trad_psf_manifest": str(pair.trad.manifest_path),
        "mlp_psf_manifest": str(pair.mlp.manifest_path),
        "trad_psf_manifest_sha256": sha256(pair.trad.manifest_path),
        "mlp_psf_manifest_sha256": sha256(pair.mlp.manifest_path),
        "central_plane_sources": {"Bloch": str(Path(trad_run).resolve()), "FrozenMLP": str(Path(mlp_run).resolve())},
    }
    return D1Inputs(subject_id=subject_id, arrays=arrays, native_manifest=reference.manifest_path, provenance=provenance)


def file_sha256(path: str | Path) -> str:
    return sha256(Path(path))


def git_commit(root: str | Path) -> str:
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
