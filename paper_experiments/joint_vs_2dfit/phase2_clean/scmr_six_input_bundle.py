"""Publish historical SCMR six-input planes on the current native lattice.

The producer is deliberately fail-closed: maps and masks may only move by a
proved signed-permutation/integer-voxel relation; no interpolation is present.
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
from .final_cyj_reporting import _load_all_verified_native_cache_groups

STACKS = ("sax", "2ch", "4ch")
PARAMETERS = ("t1", "t2")
HISTORICAL_FIELDS = ("original_ms", "registered_ms", "synthetic_ms", "training_mask", "synthetic_support", "fitting_valid_mask")
OUTPUT_FIELDS = ("original", "registered", "synthetic", "training_mask", "synthetic_support", "fitting_valid_mask")
EXPECTED_CYJ_L100_GROUPS = {"sax": 14, "2ch": 15, "4ch": 12}


def _record(path: str | Path) -> dict[str, str]:
    item = Path(path)
    if not item.is_file():
        raise FileNotFoundError(item)
    return {"path": str(item.resolve()), "sha256": sha256(item)}


def _array_and_affine(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    import nibabel as nib

    image = nib.load(str(path)); values = np.asarray(image.get_fdata())
    while values.ndim > 2 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"{path}: expected one 2-D/singleton-z NIfTI, got {values.shape}")
    return values, np.asarray(image.affine, float)


def exact_grid_relation(source_affine: np.ndarray, source_shape: tuple[int, int], target_affine: np.ndarray, target_shape: tuple[int, int], *, tolerance_mm: float = 1e-4) -> dict[str, Any]:
    """Prove target [row,col] is a discrete reindex of source [x,y]."""
    matrix = np.linalg.inv(np.asarray(source_affine, float)) @ np.asarray(target_affine, float)
    if not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-8, rtol=0):
        raise ValueError("SCMR_SIX_INPUT_GEOMETRY_NOT_EXACTLY_REINDEXABLE: non-affine homogeneous row")
    linear, offset = matrix[:2, :2], matrix[:2, 3]
    rounded_linear, rounded_offset = np.rint(linear), np.rint(offset)
    if (not np.allclose(linear, rounded_linear, atol=1e-8, rtol=0) or not np.allclose(offset, rounded_offset, atol=1e-6, rtol=0)
            or not np.allclose(matrix[2, :2], 0, atol=1e-8, rtol=0) or not np.isclose(matrix[2, 3], round(matrix[2, 3]), atol=1e-6, rtol=0)):
        raise ValueError("SCMR_SIX_INPUT_GEOMETRY_NOT_EXACTLY_REINDEXABLE: non-integer/through-plane relation")
    permutation = np.asarray(rounded_linear, int)
    if not (np.all(np.sum(np.abs(permutation), axis=0) == 1) and np.all(np.sum(np.abs(permutation), axis=1) == 1)):
        raise ValueError("SCMR_SIX_INPUT_GEOMETRY_NOT_EXACTLY_REINDEXABLE: scale, shear, or non-right-angle rotation")
    rows, cols = np.indices(target_shape, dtype=int)
    source_indices = permutation @ np.stack((rows.ravel(), cols.ravel())) + np.asarray(rounded_offset, int)[:, None]
    if (source_indices[0].min() < 0 or source_indices[1].min() < 0 or source_indices[0].max() >= source_shape[0] or source_indices[1].max() >= source_shape[1]
            or np.unique(source_indices.T, axis=0).shape[0] != rows.size):
        raise ValueError("SCMR_SIX_INPUT_GEOMETRY_NOT_EXACTLY_REINDEXABLE: target lattice is not one-to-one in source bounds")
    target_vox = np.vstack((rows.ravel(), cols.ravel(), np.zeros(rows.size), np.ones(rows.size)))
    source_vox = np.vstack((source_indices, np.full(rows.size, int(round(matrix[2, 3]))), np.ones(rows.size)))
    error = float(np.max(np.linalg.norm((np.asarray(target_affine) @ target_vox)[:3] - (np.asarray(source_affine) @ source_vox)[:3], axis=0)))
    if error > tolerance_mm:
        raise ValueError("SCMR_SIX_INPUT_GEOMETRY_NOT_EXACTLY_REINDEXABLE: physical-coordinate mismatch")
    relation = "IDENTICAL_GRID" if np.array_equal(permutation, np.eye(2, dtype=int)) and np.array_equal(rounded_offset, (0, 0)) and tuple(source_shape) == tuple(target_shape) else "EXACT_REORIENTABLE_GRID"
    return {"relation_type": relation, "permutation": permutation.tolist(), "flips": [bool(permutation[row, np.nonzero(permutation[row])[0][0]] < 0) for row in range(2)], "integer_offset": np.asarray(rounded_offset, int).tolist(), "source_z_index": int(round(matrix[2, 3])), "max_physical_coordinate_error_mm": error, "no_interpolation": True, "source_indices": source_indices}


def apply_exact_relation(values: np.ndarray, relation: Mapping[str, Any], target_shape: tuple[int, int]) -> np.ndarray:
    """Copy source values onto target lattice using only proved integer indices."""
    source = np.asarray(values)
    indices = np.asarray(relation["source_indices"], int)
    return source[indices[0], indices[1]].reshape(target_shape)


def _group_map(path: str | Path) -> dict[tuple[str, str, int], int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "scmr_historical_group_mapping/v1":
        raise ValueError("group map must be scmr_historical_group_mapping/v1")
    result: dict[tuple[str, str, int], int] = {}
    for item in payload.get("mappings", []):
        key = (str(item["parameter"]).lower(), str(item["stack"]), int(item["historical_native_slice_id"]))
        if key in result:
            raise ValueError(f"duplicate historical group mapping: {key}")
        result[key] = int(item["original_group_idx"])
    return result


def _l100_records(root: str | Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    root = Path(root); manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema not in {"phase4r_l100_native_support/v1", "q004_direct_native_endpoint_audit/v2"}:
        raise ValueError("phase4r L100 source must be phase4r_l100_native_support/v1 or q004_direct_native_endpoint_audit/v2")
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for item in payload.get("planes", []):
        group_key = "original_group_idx" if schema == "phase4r_l100_native_support/v1" else "group_idx"
        key = (str(item["parameter"]).lower(), str(item["stack"]), int(item[group_key]))
        if key in result:
            raise ValueError(f"duplicate L100 support identity: {key}")
        record = dict(item); path = Path(record["path"] if schema == "phase4r_l100_native_support/v1" else record["output"])
        record["path"] = str((path if path.is_absolute() else root / path).resolve())
        record["source_field"] = "phase4r_l100" if schema == "phase4r_l100_native_support/v1" else "l100_native"
        record["mapping_provenance"] = record.get("mapping_provenance", {"source_schema": schema, "source_group_identity": {"parameter": key[0], "stack": key[1], group_key: key[2]}, "coordinate_contract": payload.get("coordinate_contract")})
        result[key] = record
    return result


def _historical_records(path: str | Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    for item in payload.get("slices", []):
        key = (str(item["parameter"]).lower(), str(item["stack"]), int(item["native_slice_id"]))
        if key in records:
            raise ValueError(f"duplicate historical common-mask record: {key}")
        missing = [name for name in HISTORICAL_FIELDS if not item.get(name)]
        if missing:
            raise ValueError(f"historical {key} lacks explicit fields: {missing}")
        records[key] = item
    return records


def discover_scmr_six_input_sources(*, historical_common_mask_manifest: str | Path, group_map_manifest: str | Path, r0_native_maps_root: str | Path, sparse_manifest: str | Path, phase4r_l100_root: str | Path) -> dict[str, Any]:
    """Read-only coverage report. Missing inputs are reported, never invented."""
    historical, group_map, l100 = _historical_records(historical_common_mask_manifest), _group_map(group_map_manifest), _l100_records(phase4r_l100_root)
    current = {stack: _load_all_verified_native_cache_groups(r0_native_maps_root, sparse_manifest, stack) for stack in STACKS}
    found: list[dict[str, Any]] = []; missing: list[dict[str, Any]] = []
    coverage = {parameter: {stack: {"required": int(len(current[stack]["group_idx"])), "found": 0, "missing": 0} for stack in STACKS} for parameter in PARAMETERS}
    for parameter in PARAMETERS:
        for stack in STACKS:
            for group in (int(value) for value in current[stack]["group_idx"]):
                candidates = [key for key, value in group_map.items() if key[:2] == (parameter, stack) and value == group]
                if len(candidates) != 1:
                    missing.append({"parameter": parameter, "stack": stack, "original_group_idx": group, "missing": "explicit_historical_group_mapping"}); coverage[parameter][stack]["missing"] += 1; continue
                historical_key = candidates[0]
                record = historical.get(historical_key); l100_record = l100.get((parameter, stack, group))
                absent = [] if record is not None else ["historical_six_input_record"]
                if l100_record is None: absent.append("phase4r_l100_native_support")
                elif not Path(l100_record["path"]).is_file(): absent.append("phase4r_l100_native_support_file")
                if record is not None:
                    absent.extend(name for name in HISTORICAL_FIELDS if not Path(record[name]).is_file())
                if absent:
                    missing.append({"parameter": parameter, "stack": stack, "original_group_idx": group, "missing": absent}); coverage[parameter][stack]["missing"] += 1
                else:
                    found.append({"parameter": parameter, "stack": stack, "original_group_idx": group, "historical_native_slice_id": historical_key[2]}); coverage[parameter][stack]["found"] += 1
    return {"schema": "scmr_six_input_discovery/v1", "found": found, "missing": missing, "coverage": coverage, "complete": not missing}


def build_scmr_six_input_bundle(*, historical_common_mask_manifest: str | Path, group_map_manifest: str | Path, r0_native_maps_root: str | Path, sparse_manifest: str | Path, phase4r_l100_root: str | Path, output: str | Path) -> Path:
    """Build the complete immutable CYJ L100 six-input source bundle."""
    output = Path(output)
    if output.exists(): raise FileExistsError(f"refusing overwrite: {output}")
    discovery = discover_scmr_six_input_sources(historical_common_mask_manifest=historical_common_mask_manifest, group_map_manifest=group_map_manifest, r0_native_maps_root=r0_native_maps_root, sparse_manifest=sparse_manifest, phase4r_l100_root=phase4r_l100_root)
    if not discovery["complete"]: raise ValueError(f"SCMR six-input discovery incomplete: {discovery['missing']}")
    historical, group_map, l100 = _historical_records(historical_common_mask_manifest), _group_map(group_map_manifest), _l100_records(phase4r_l100_root)
    current = {stack: _load_all_verified_native_cache_groups(r0_native_maps_root, sparse_manifest, stack) for stack in STACKS}
    sparse = json.loads(Path(sparse_manifest).read_text(encoding="utf-8"))
    if sparse.get("subject") != "CYJ" or sparse.get("condition") != "L100": raise ValueError("formal six-input producer is CYJ/L100 only")
    if {stack: len(current[stack]["group_idx"]) for stack in STACKS} != EXPECTED_CYJ_L100_GROUPS: raise ValueError("CYJ L100 group completeness is not 14/15/12")
    output.parent.mkdir(parents=True, exist_ok=True); staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        planes: list[dict[str, Any]] = []
        for parameter in PARAMETERS:
            for stack in STACKS:
                for index, group_value in enumerate(current[stack]["group_idx"]):
                    group = int(group_value); target_shape = tuple(int(value) for value in np.asarray(current[stack][f"{parameter}_ms"][index]).shape)
                    target_affine = np.diag([-1., -1., 1., 1.]) @ np.asarray(current[stack]["affine_lps_rc"][index], float)
                    historical_key = next(key for key, value in group_map.items() if key[:2] == (parameter, stack) and value == group)
                    record = historical[historical_key]
                    mapped: dict[str, np.ndarray] = {}; relations: dict[str, Any] = {}; sources: dict[str, Any] = {}
                    for output_name, historical_name in zip(OUTPUT_FIELDS, HISTORICAL_FIELDS):
                        values, affine = _array_and_affine(record[historical_name]); relation = exact_grid_relation(affine, tuple(values.shape), target_affine, target_shape)
                        mapped[output_name] = apply_exact_relation(values, relation, target_shape)
                        relation_record = {key: value for key, value in relation.items() if key != "source_indices"}
                        relation_record.update({"source_voxel_count": int(values.size), "output_voxel_count": int(mapped[output_name].size)})
                        if output_name in {"training_mask", "synthetic_support", "fitting_valid_mask"}:
                            relation_record.update({"source_true_voxels": int(np.asarray(values, bool).sum()), "output_true_voxels": int(np.asarray(mapped[output_name], bool).sum())})
                        relations[output_name] = relation_record
                        sources[output_name] = {**_record(record[historical_name]), "shape_rc": list(values.shape)}
                    l100_record = l100[(parameter, stack, group)]; l100_path = Path(l100_record["path"])
                    if not l100_path.is_file() or (l100_record.get("sha256") is not None and sha256(l100_path) != l100_record["sha256"]): raise ValueError(f"L100 provenance mismatch: {l100_path}")
                    with np.load(l100_path, allow_pickle=False) as archive:
                        if l100_record["source_field"] not in archive.files: raise ValueError(f"{l100_path}: missing {l100_record['source_field']}")
                        l100_mask = np.asarray(archive[l100_record["source_field"]], bool)
                    if l100_mask.shape != target_shape: raise ValueError(f"{l100_path}: L100 native support is not on current exact target lattice")
                    target = staging / "planes" / parameter / stack / f"group{group:03d}.npz"; target.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(target, **mapped, phase4r_l100=l100_mask, source_support=np.asarray(current[stack]["source_support"][index], bool), reference=np.asarray(current[stack][f"{parameter}_ms"][index], np.float32))
                    planes.append({"parameter": parameter, "stack": stack, "original_group_idx": group, "historical_native_slice_id": historical_key[2], "shape_rc": list(target_shape), "historical_sources": sources, "current_r0_source": current[stack]["provenance"], "l100_source": {"path": str(l100_path.resolve()), "sha256": sha256(l100_path), "source_field": l100_record["source_field"], "mask_voxels": int(l100_mask.sum()), "mapping_provenance": l100_record["mapping_provenance"]}, "exact_grid_relations": relations, "source_support_voxels": int(np.asarray(current[stack]["source_support"][index], bool).sum()), "source_support_output_voxels": int(np.asarray(current[stack]["source_support"][index], bool).sum()), "output": {"path": str(target.relative_to(staging)), "sha256": sha256(target)}, "no_interpolation": True})
        if len(planes) != 82: raise ValueError(f"expected 82 CYJ L100 parameter-plane files, got {len(planes)}")
        write_immutable_json(staging / "manifest.json", {"schema": "scmr_six_input_native_bundle/v1", "subject": "CYJ", "condition": "L100_fold0", "historical_common_mask_manifest": _record(historical_common_mask_manifest), "group_mapping_manifest": _record(group_map_manifest), "phase4r_l100_manifest": _record(Path(phase4r_l100_root) / "manifest.json"), "sparse_manifest": _record(sparse_manifest), "planes": planes, "no_training": True, "no_reconstruction": True, "no_resampling": True, "no_interpolation": True})
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True); raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("historical_common_mask_manifest", "group_map_manifest", "r0_native_maps_root", "sparse_manifest", "phase4r_l100_root"): parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--output"); parser.add_argument("--mode", choices=("discover", "build"), default="build")
    args = vars(parser.parse_args())
    if args.pop("mode") == "discover":
        args.pop("output"); print(json.dumps(discover_scmr_six_input_sources(**args), indent=2, sort_keys=True)); return
    if not args.get("output"): raise ValueError("--output is required for build")
    print(build_scmr_six_input_bundle(**args))


if __name__ == "__main__": main()
