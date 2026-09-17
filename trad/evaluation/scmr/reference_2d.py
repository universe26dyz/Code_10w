"""Verified native 2-D dictionary-map bundle loading.

The bundle deliberately has a small, explicit contract.  It prevents an old
2-D-fit-first result from being silently compared with a different MIND/MP-PCA
preprocessing run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


STACKS = ("sax", "2ch", "4ch")
REQUIRED_SEMANTICS = "MP-PCA(MIND_mag_reg)"


@dataclass(frozen=True)
class NativeStack:
    t1_ms: np.ndarray
    t2_ms: np.ndarray
    valid_mask: np.ndarray
    group_idx: np.ndarray
    path: Path


@dataclass(frozen=True)
class NativeReference:
    root: Path
    manifest_path: Path
    manifest: dict
    stacks: dict[str, NativeStack]
    figure1_stacks: dict[str, Path]
    file_hashes: dict[str, str]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verified_reference(root: str | Path, subject_id: str, preprocessed_root: str | Path) -> NativeReference:
    root = Path(root).expanduser().resolve()
    manifest_path = root / "native_reference_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "No verified native 2-D reference bundle: expected "
            f"{manifest_path}. Do not substitute older 2D-fit-first maps."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("subject_id") != subject_id:
        raise ValueError(f"Native reference subject_id={manifest.get('subject_id')!r}, expected {subject_id!r}.")
    if manifest.get("preprocessing_semantics") != REQUIRED_SEMANTICS:
        raise ValueError(
            "Native reference must declare preprocessing_semantics="
            f"{REQUIRED_SEMANTICS!r}; got {manifest.get('preprocessing_semantics')!r}."
        )
    if manifest.get("map_units") != "ms":
        raise ValueError("Native reference must declare map_units='ms'; normalized map exports are not accepted.")
    source_hashes = manifest.get("preprocessed_mat_sha256", {})
    maps = manifest.get("maps", {})
    preprocessed_root = Path(preprocessed_root).expanduser().resolve()
    stacks: dict[str, NativeStack] = {}
    hashes: dict[str, str] = {str(manifest_path): sha256(manifest_path)}
    for stack in STACKS:
        current_mat = preprocessed_root / subject_id / stack / "preprocessed.mat"
        if not current_mat.is_file():
            raise FileNotFoundError(f"Current preprocessed MAT is missing: {current_mat}")
        expected_hash = source_hashes.get(stack)
        actual_hash = sha256(current_mat)
        if expected_hash != actual_hash:
            raise ValueError(
                f"Native reference provenance fails for {stack}: manifest hash does not match "
                f"current {current_mat}."
            )
        archive = root / str(maps.get(stack, ""))
        if not archive.is_file():
            raise FileNotFoundError(f"Native reference archive for {stack} is missing: {archive}")
        with np.load(archive, allow_pickle=False) as data:
            required = {"t1_ms", "t2_ms", "group_idx"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"{archive} lacks required keys: {sorted(missing)}")
            t1, t2 = np.asarray(data["t1_ms"], dtype=np.float32), np.asarray(data["t2_ms"], dtype=np.float32)
            groups = np.asarray(data["group_idx"], dtype=np.int64)
            valid = np.asarray(data["valid_mask"], dtype=bool) if "valid_mask" in data else np.isfinite(t1) & np.isfinite(t2) & (t1 != 0) & (t2 != 0)
        if t1.ndim != 3 or t1.shape != t2.shape or valid.shape != t1.shape:
            raise ValueError(f"{archive} must contain matching [group,row,col] T1/T2/valid_mask arrays.")
        if groups.shape != (t1.shape[0],) or not np.array_equal(groups, np.arange(t1.shape[0])):
            raise ValueError(f"{archive} group_idx must be contiguous zero-based and match the map depth.")
        stacks[stack] = NativeStack(t1, t2, valid, groups, archive)
        hashes[str(current_mat)] = actual_hash
        hashes[str(archive)] = sha256(archive)
    figure1 = manifest.get("figure1_native_stacks", {})
    figure1_stacks = {parameter: (root / str(figure1.get(parameter, ""))).resolve() for parameter in ("t1", "t2")}
    for parameter, path in figure1_stacks.items():
        if not path.is_file():
            raise FileNotFoundError(f"Figure 1 native {parameter.upper()} stack is missing: {path}")
        hashes[str(path)] = sha256(path)
    return NativeReference(root, manifest_path, manifest, stacks, figure1_stacks, hashes)
