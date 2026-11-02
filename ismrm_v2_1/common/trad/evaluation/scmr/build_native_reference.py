"""Build a verified native 2-D dictionary-map bundle for SCMR evaluation.

MATLAB performs the matching with the vendored original MultiMap functions.
This module never reimplements dictionary physics: it only launches that
wrapper, verifies its outputs against prepared observations, and packages the
maps with their current-preprocessing provenance.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .reference_2d import REQUIRED_SEMANTICS, STACKS, sha256


CYJ_GROUP_COUNTS = {"sax": 14, "2ch": 15, "4ch": 12}
NONFINITE_FRACTION_LIMIT = 1e-3


def _write_json(value: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_mapping(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise RuntimeError("scipy is required to package the MATLAB mapping output.") from exc
    if not path.is_file():
        raise FileNotFoundError(f"MATLAB mapping output is missing: {path}")
    data = loadmat(path, squeeze_me=False, struct_as_record=False)
    required = {"t1_ms", "t2_ms", "valid_mask", "group_idx"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"{path} lacks required mapping fields: {sorted(missing)}")
    t1, t2 = np.asarray(data["t1_ms"], dtype=np.float32), np.asarray(data["t2_ms"], dtype=np.float32)
    valid = np.asarray(data["valid_mask"], dtype=bool)
    groups = np.asarray(data["group_idx"], dtype=np.int64).reshape(-1)
    if t1.ndim != 3 or t1.shape != t2.shape or valid.shape != t1.shape:
        raise ValueError(f"{path} T1/T2/valid_mask must have the same [group,row,col] shape.")
    if groups.shape != (t1.shape[0],) or not np.array_equal(groups, np.arange(t1.shape[0])):
        raise ValueError(f"{path} group_idx must be contiguous zero-based and match map order.")
    for parameter, values in (("T1", t1), ("T2", t2)):
        fraction = float(np.mean(~np.isfinite(values)))
        if fraction > NONFINITE_FRACTION_LIMIT:
            raise ValueError(f"{path} {parameter} non-finite fraction {fraction:.6f} exceeds {NONFINITE_FRACTION_LIMIT}.")
    if np.any(valid & (~np.isfinite(t1) | ~np.isfinite(t2) | (t1 <= 0) | (t2 <= 0))):
        raise ValueError(f"{path} valid_mask includes non-finite or non-positive T1/T2 values.")
    if not np.any(valid):
        raise ValueError(f"{path} valid_mask is empty.")
    return t1, t2, valid, groups


def _load_prepared(path: Path, stack: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared observations are missing: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"images", "masks", "group_idx", "weight_idx", "affine_lps_rc"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} lacks required prepared fields: {sorted(missing)}")
        images = np.asarray(data["images"])
        masks = np.asarray(data["masks"], dtype=bool)
        groups = np.asarray(data["group_idx"], dtype=np.int64)
        weights = np.asarray(data["weight_idx"], dtype=np.int64)
        affine = np.asarray(data["affine_lps_rc"], dtype=np.float64)
    if images.ndim != 3 or masks.shape != images.shape or groups.shape != (images.shape[0],) or weights.shape != groups.shape:
        raise ValueError(f"{path} images/masks/group_idx/weight_idx shapes are inconsistent.")
    if affine.shape != (images.shape[0], 4, 4):
        raise ValueError(f"{path} affine_lps_rc must be [observation,4,4].")
    expected_groups = np.arange(int(groups.max()) + 1) if groups.size else np.empty(0, dtype=np.int64)
    if not np.array_equal(np.unique(groups), expected_groups):
        raise ValueError(f"{path} group_idx is not contiguous zero-based.")
    for group in expected_groups:
        indices = np.flatnonzero(groups == group)
        if indices.size != 10 or not np.array_equal(np.sort(weights[indices]), np.arange(10)):
            raise ValueError(f"{path} group {group} does not contain exactly weights 0..9.")
        if not np.allclose(affine[indices], affine[indices[0]], rtol=0, atol=2e-4):
            raise ValueError(f"{path} group {group} weights do not share one HB1 cropped affine.")
    if stack == "sax" and not np.array_equal(expected_groups, np.arange(expected_groups.size)):
        raise ValueError(f"{path} SAX group ordering cannot be confirmed.")
    return groups, weights, affine, tuple(int(value) for value in images.shape[1:])


def _sax_volume_affine_ras(groups: np.ndarray, weights: np.ndarray, affine_lps_rc: np.ndarray) -> np.ndarray:
    selections = []
    for group in np.unique(groups):
        index = np.flatnonzero((groups == group) & (weights == 0))
        if index.size != 1:
            raise ValueError(f"SAX group {group} lacks one unique weight-0 affine.")
        selections.append(int(index[0]))
    matrices = affine_lps_rc[np.asarray(selections, dtype=np.int64)]
    base = matrices[0].copy()
    translations = matrices[:, :3, 3]
    if len(matrices) > 1:
        increments = np.diff(translations, axis=0)
        slice_increment = increments.mean(axis=0)
        if not np.allclose(increments, slice_increment, rtol=0, atol=2e-4):
            raise ValueError("SAX group order does not have regular physical slice positions.")
        thickness = float(np.linalg.norm(base[:3, 2]))
        if not np.isclose(np.linalg.norm(slice_increment), thickness, rtol=1e-4, atol=2e-3):
            raise ValueError("SAX group-to-group physical spacing does not match native slice thickness.")
        base[:3, 2] = slice_increment
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    return lps_to_ras @ base


def _write_sax_nifti(t1: np.ndarray, t2: np.ndarray, groups: np.ndarray, weights: np.ndarray, affine_lps_rc: np.ndarray, output: Path) -> dict[str, str]:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("nibabel is required to write native SAX NIfTI stacks.") from exc
    affine_ras = _sax_volume_affine_ras(groups, weights, affine_lps_rc)
    paths: dict[str, str] = {}
    for parameter, values, filename in (("T1", t1, "sax_T1_stack_ms.nii.gz"), ("T2", t2, "sax_T2_stack_ms.nii.gz")):
        image = nib.Nifti1Image(np.moveaxis(values, 0, 2).astype(np.float32, copy=False), affine_ras)
        image.header.set_xyzt_units("mm")
        image.header["descrip"] = b"native 8-mm SAX dictionary map; physical RAS-mm; ms"
        target = output / filename
        nib.save(image, target)
        paths[parameter.lower()] = target.name
    return paths


def package_native_reference(subject_id: str, prepared_root: str | Path, preprocessed_root: str | Path, mapping_root: str | Path, output_root: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Validate MATLAB maps and create the reference bundle consumed by Figure 1/2."""

    prepared_root, preprocessed_root = Path(prepared_root).resolve(), Path(preprocessed_root).resolve()
    mapping_root, output = Path(mapping_root).resolve(), Path(output_root).resolve()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty native reference bundle: {output}")
    output.mkdir(parents=True, exist_ok=True)
    map_entries: dict[str, str] = {}
    preprocessed_hashes: dict[str, str] = {}
    shape_report: dict[str, list[int]] = {}
    sax_geometry: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    for stack in STACKS:
        source_mat = preprocessed_root / subject_id / stack / "preprocessed.mat"
        if not source_mat.is_file():
            raise FileNotFoundError(f"Exact current preprocessed source MAT is missing: {source_mat}")
        t1, t2, valid, mapping_groups = _load_mapping(mapping_root / f"{stack}_mapping.mat")
        groups, weights, affine, image_shape = _load_prepared(prepared_root / subject_id / stack / "observations.npz", stack)
        if t1.shape != (len(np.unique(groups)), *image_shape):
            raise ValueError(f"{stack} mapping shape {t1.shape} does not match prepared groups/images {(len(np.unique(groups)), *image_shape)}.")
        if not np.array_equal(mapping_groups, np.unique(groups)):
            raise ValueError(f"{stack} MATLAB mapping group_idx does not match observations.npz group ordering.")
        if subject_id == "CYJ" and len(mapping_groups) != CYJ_GROUP_COUNTS[stack]:
            raise ValueError(f"CYJ {stack} requires {CYJ_GROUP_COUNTS[stack]} groups, got {len(mapping_groups)}.")
        target = output / f"{stack}.npz"
        np.savez_compressed(target, t1_ms=t1, t2_ms=t2, valid_mask=valid, group_idx=mapping_groups)
        map_entries[stack] = target.name
        preprocessed_hashes[stack] = sha256(source_mat)
        shape_report[stack] = list(t1.shape)
        if stack == "sax":
            sax_geometry = (t1, t2, groups, weights, affine)
    if sax_geometry is None:
        raise RuntimeError("SAX mapping was not packaged.")
    t1, t2, groups, weights, affine = sax_geometry
    figure1_stacks = _write_sax_nifti(t1, t2, groups, weights, affine, output)
    manifest = {
        "subject_id": subject_id,
        "preprocessing_semantics": REQUIRED_SEMANTICS,
        "map_units": "ms",
        "preprocessed_mat_sha256": preprocessed_hashes,
        "maps": map_entries,
        "figure1_native_stacks": figure1_stacks,
        "group_ordering_verified": True,
        "group_ordering_definition": "MATLAB iterates exact Mag_crop dimension-4 group order; packager verifies group_idx and weights 0..9 against current observations.npz.",
        "shape_group_row_col": shape_report,
        "mapping_output_sha256": {stack: sha256(output / filename) for stack, filename in map_entries.items()},
        "source_geometry": "current prepared observations affine_lps_rc; converted from LPS to RAS only for SAX NIfTI",
        "native_slice_geometry": "native group-to-group spacing verified against prepared 8-mm slice thickness; no SVR or through-plane interpolation",
    }
    _write_json(manifest, output / "native_reference_manifest.json")
    return manifest


def _matlab_quote(value: str | Path) -> str:
    return str(value).replace("'", "''")


def run_matlab_dictionary_mapping(subject_id: str, preprocessed_root: str | Path, mapping_root: str | Path, *, matlab_executable: str = "matlab") -> None:
    """Run the MATLAB-only original dictionary matcher once for every stack."""

    preprocessed_root, mapping_root = Path(preprocessed_root).resolve(), Path(mapping_root).resolve()
    if mapping_root.exists() and any(mapping_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty MATLAB mapping directory: {mapping_root}")
    mapping_root.mkdir(parents=True, exist_ok=True)
    matlab_dir = Path(__file__).resolve().parent / "matlab"
    for stack in STACKS:
        source = preprocessed_root / subject_id / stack / "preprocessed.mat"
        target = mapping_root / f"{stack}_mapping.mat"
        command = f"addpath('{_matlab_quote(matlab_dir)}'); build_native_reference_maps('{_matlab_quote(source)}','{_matlab_quote(target)}');"
        subprocess.run([matlab_executable, "-batch", command], check=True)
        if not target.is_file():
            raise RuntimeError(f"MATLAB completed without expected mapping output: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--preprocessed-root", required=True)
    parser.add_argument("--mapping-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-matlab", action="store_true")
    parser.add_argument("--matlab-executable", default="matlab")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        if not args.skip_matlab:
            run_matlab_dictionary_mapping(args.subject_id, args.preprocessed_root, args.mapping_root, matlab_executable=args.matlab_executable)
        manifest = package_native_reference(args.subject_id, args.prepared_root, args.preprocessed_root, args.mapping_root, args.output, overwrite=args.overwrite)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
