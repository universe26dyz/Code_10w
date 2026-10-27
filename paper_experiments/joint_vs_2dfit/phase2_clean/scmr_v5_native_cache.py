"""Build an immutable SCMR-v5 native-map cache from current cropped inputs.

The mapper is MATLAB-only because it calls the retained SCMR-v5 fitting
function directly.  This module packages its results using the exact geometry
and group order of the current ``observations.npz`` files.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import sha256, write_immutable_json
from .fit_first import SCMR_V5_DICTIONARY_CONTRACT, SCMR_V5_NATIVE_FIT_METHOD

STACKS = ("sax", "2ch", "4ch")


def _matlab_quote(path: str | Path) -> str:
    return str(path).replace("'", "''")


def _prepared_subject_root(prepared_root: str | Path, subject: str) -> Path:
    root = Path(prepared_root).resolve()
    return root if all((root / stack / "observations.npz").is_file() for stack in STACKS) else root / subject


def _require_empty_target(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite native-map cache: {path}")
    path.mkdir(parents=True)


def run_matlab_mapping(*, subject: str, preprocessed_root: str | Path, mapping_root: str | Path, matlab_executable: str = "matlab") -> None:
    """Call the exact SCMR-v5 MATLAB matcher once for each current stack."""
    source_root, target = Path(preprocessed_root).resolve(), Path(mapping_root).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite MATLAB mapping root: {target}")
    target.mkdir(parents=True)
    matlab_dir = Path(__file__).resolve().parents[3] / "trad" / "evaluation" / "scmr" / "matlab"
    for stack in STACKS:
        source = source_root / subject / stack / "preprocessed.mat"
        if not source.is_file():
            raise FileNotFoundError(f"missing current preprocessed input: {source}")
        destination = target / f"{stack}_mapping.mat"
        command = (
            f"addpath('{_matlab_quote(matlab_dir)}'); "
            f"build_scmr_v5_native_maps('{_matlab_quote(source)}','{_matlab_quote(destination)}');"
        )
        subprocess.run([matlab_executable, "-batch", command], check=True)
        if not destination.is_file():
            raise RuntimeError(f"MATLAB completed without output: {destination}")


def _load_mapping(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from scipy.io import loadmat

    payload = loadmat(path, squeeze_me=False, struct_as_record=False)
    required = {"t1_ms", "t2_ms", "b1", "valid_mask", "group_idx"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"{path} lacks fields: {sorted(missing)}")
    t1 = np.asarray(payload["t1_ms"], dtype=np.float32)
    t2 = np.asarray(payload["t2_ms"], dtype=np.float32)
    b1 = np.asarray(payload["b1"], dtype=np.float32)
    valid = np.asarray(payload["valid_mask"], dtype=bool)
    groups = np.asarray(payload["group_idx"], dtype=np.int64).reshape(-1)
    if t1.ndim != 3 or not (t1.shape == t2.shape == b1.shape == valid.shape):
        raise ValueError(f"{path}: maps must share [group,row,col] shape")
    if not np.array_equal(groups, np.arange(t1.shape[0], dtype=np.int64)):
        raise ValueError(f"{path}: group_idx must be contiguous and zero based")
    invalid = ~valid
    for values in (t1, t2, b1):
        values[invalid] = np.nan
    if not np.any(valid) or np.any(valid & (~np.isfinite(t1) | ~np.isfinite(t2) | ~np.isfinite(b1) | (t1 <= 0) | (t2 <= 0) | (b1 <= 0))):
        raise ValueError(f"{path}: valid SCMR-v5 map support is invalid")
    return t1, t2, b1, valid, groups


def _load_observations(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {"images", "masks", "group_idx", "weight_idx", "affine_lps_rc", "pixel_spacing_rc_mm", "slice_thickness_mm"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} lacks fields: {sorted(missing)}")
        return tuple(np.asarray(data[name]) for name in ("images", "group_idx", "weight_idx", "affine_lps_rc", "pixel_spacing_rc_mm", "slice_thickness_mm"))  # type: ignore[return-value]


def package_native_cache(*, subject: str, prepared_root: str | Path, preprocessed_root: str | Path, mapping_root: str | Path, output: str | Path) -> Path:
    """Package validated v5 maps into the cache consumed by Fit-first/AHA."""
    target = Path(output).resolve()
    _require_empty_target(target)
    prepared = _prepared_subject_root(prepared_root, subject)
    preprocessed = Path(preprocessed_root).resolve()
    mapping = Path(mapping_root).resolve()
    try:
        for stack in STACKS:
            observations_path = prepared / stack / "observations.npz"
            source_mat = preprocessed / subject / stack / "preprocessed.mat"
            t1, t2, b1, valid, groups = _load_mapping(mapping / f"{stack}_mapping.mat")
            images, group_idx, weight_idx, affine, spacing, thickness = _load_observations(observations_path)
            observed_groups = np.unique(group_idx).astype(np.int64)
            if t1.shape != (len(observed_groups), *images.shape[1:]) or not np.array_equal(groups, observed_groups):
                raise ValueError(f"{stack}: SCMR-v5 maps do not match current prepared shape/group order")
            rows = [np.flatnonzero(group_idx == group) for group in groups]
            if any(indices.size != 10 or not np.array_equal(np.sort(weight_idx[indices]), np.arange(10)) for indices in rows):
                raise ValueError(f"{stack}: each prepared group must contain weights 0..9")
            first = np.asarray([indices[0] for indices in rows], dtype=np.int64)
            stack_target = target / stack
            stack_target.mkdir()
            np.savez_compressed(stack_target / "native_maps.npz", t1_ms=t1, t2_ms=t2, b1=b1, group_idx=groups)
            np.savez_compressed(stack_target / "geometry.npz", affine_lps_rc=affine[first], pixel_spacing_rc_mm=spacing[first], slice_thickness_mm=thickness[first])
            write_immutable_json(stack_target / "manifest.json", {
                "method": SCMR_V5_NATIVE_FIT_METHOD,
                "dictionary_contract": SCMR_V5_DICTIONARY_CONTRACT,
                "subject_id": subject,
                "source_observations": {"path": str(observations_path.resolve()), "sha256": sha256(observations_path)},
                "source_preprocessed": {"path": str(source_mat.resolve()), "sha256": sha256(source_mat)},
                "scmr_v5_mapping": {"path": str((mapping / f"{stack}_mapping.mat").resolve()), "sha256": sha256(mapping / f"{stack}_mapping.mat")},
                "group_ids": [int(value) for value in groups],
                "valid_pixel_count": int(valid.sum()),
                "native_maps_sha256": sha256(stack_target / "native_maps.npz"),
                "geometry_sha256": sha256(stack_target / "geometry.npz"),
                "invalid_pixel_policy": "invalid SCMR-v5 QC matches are NaN and exported as NeSVoR background",
            })
        write_immutable_json(target / "native_cache_manifest.json", {
            "schema": "scmr_v5_native_cache/v1", "subject_id": subject,
            "dictionary_contract": SCMR_V5_DICTIONARY_CONTRACT,
            "stacks": list(STACKS), "immutable": True,
        })
    except BaseException:
        raise
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--preprocessed-root", required=True)
    parser.add_argument("--mapping-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--matlab-executable", default="matlab")
    parser.add_argument("--skip-matlab", action="store_true")
    args = parser.parse_args()
    if not args.skip_matlab:
        run_matlab_mapping(subject=args.subject, preprocessed_root=args.preprocessed_root, mapping_root=args.mapping_root, matlab_executable=args.matlab_executable)
    result = package_native_cache(subject=args.subject, prepared_root=args.prepared_root, preprocessed_root=args.preprocessed_root, mapping_root=args.mapping_root, output=args.output)
    print(result)


if __name__ == "__main__":
    main()
