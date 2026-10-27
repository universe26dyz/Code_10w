"""Publish CYJ1.1-r3 Fit-first native predictions on current final NeSVoR poses.

The r3 migration preserves final-slice provenance in ``nesvor/*/result.json``
but does not pretend to be a Final5 route.  This downstream-only adapter binds
the current r3 input manifests to those source final slices explicitly.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..phase2_clean.contracts import sha256, write_immutable_json
from ..phase2_clean.evaluation import _world_sample


SCHEMA = "cyj11_r3_fit_first_final_pose_native_bundle/v1"
STACKS = ("sax", "2ch", "4ch")
PARAMETERS = ("t1", "t2")


def _record(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def r3_effective_query_mask(endpoint_root: str | Path, parameter: str) -> tuple[Path, dict[str, str]]:
    """Resolve a migrated r3 endpoint mask without applying Final5's v2 contract.

    r3 central endpoints were copied from their immutable source export.  Their
    v1 manifest deliberately retains source paths, so the local copy is bound
    by its recorded digest rather than by string-equality of the old path.
    """
    root = Path(endpoint_root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "phase4r_fixed_l100_fit_first_export_bundle/v1":
        raise ValueError("r3 Fit-first endpoint schema mismatch")
    item = manifest.get("parameters", {}).get(parameter)
    record = item.get("effective_query_mask") if isinstance(item, dict) else None
    expected = root / parameter / "query_effective_mask.nii.gz"
    if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
        raise ValueError(f"{parameter}: r3 endpoint lacks effective query-mask provenance")
    if not expected.is_file() or record["sha256"] != sha256(expected):
        raise ValueError(f"{parameter}: r3 endpoint effective query-mask SHA256 mismatch")
    return expected, _record(expected)


def r3_endpoint_volume(endpoint_root: str | Path, parameter: str) -> tuple[Path, dict[str, str]]:
    """Resolve the local copy of an r3 endpoint volume against its source digest."""
    root = Path(endpoint_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    item = manifest.get("parameters", {}).get(parameter)
    record = item.get("output_volume_ms") if isinstance(item, dict) else None
    expected = root / parameter / "volume_ms.nii.gz"
    if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
        raise ValueError(f"{parameter}: r3 endpoint lacks volume provenance")
    if not expected.is_file() or record["sha256"] != sha256(expected):
        raise ValueError(f"{parameter}: r3 endpoint volume_ms SHA256 mismatch")
    return expected, _record(expected)


def _cache_stack_root(native_root: Path, stack: str) -> Path:
    candidates = (native_root / "native_maps" / stack, native_root / stack)
    required = ("native_maps.npz", "geometry.npz", "manifest.json")
    return next((path for path in candidates if all((path / name).is_file() for name in required)), candidates[0])


def _r0_shape(native_root: Path, stack: str, group: int, parameter: str) -> tuple[int, int]:
    with np.load(_cache_stack_root(native_root, stack) / "native_maps.npz", allow_pickle=False) as maps:
        groups = [int(value) for value in maps["group_idx"]]
        if groups.count(group) != 1:
            raise ValueError(f"{stack}: native-cache group {group} is unavailable")
        return tuple(map(int, np.asarray(maps[f"{parameter}_ms"][groups.index(group)]).shape))


def _result_args(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    args = payload.get("args", payload)
    if not isinstance(args, dict):
        raise ValueError(f"{path}: result arguments are invalid")
    if args.get("registration") != "stack" or args.get("no_transformation_optimization") is not False:
        raise ValueError(f"{path}: does not prove current one-stage final-pose optimisation")
    if args.get("no_slice_scale") is not True:
        raise ValueError(f"{path}: does not prove fixed slice scale")
    return args


def r3_group_records(baseline_root: str | Path, parameter: str) -> list[dict[str, Any]]:
    """Read the r3-declared group order and the corresponding source final slices."""
    root = Path(baseline_root)
    if parameter not in PARAMETERS:
        raise ValueError("unknown quantitative parameter")
    result_path = root / "nesvor" / parameter / "result.json"
    args = _result_args(result_path)
    input_stacks = args.get("input_stacks")
    if not isinstance(input_stacks, list) or not input_stacks:
        raise ValueError(f"{result_path}: input_stacks is required")
    stack_order = [Path(str(path)).parent.parent.name.lower() for path in input_stacks]
    if not stack_order or any(stack not in STACKS for stack in stack_order) or len(set(stack_order)) != len(stack_order):
        raise ValueError(f"{result_path}: input_stacks does not encode a unique SAX/2CH/4CH order")
    final_root = Path(str(args.get("output_slices", "")))
    if not final_root.is_dir():
        raise FileNotFoundError(f"{result_path}: declared current final-slice directory is unavailable: {final_root}")
    ordered: list[tuple[str, int]] = []
    for stack in stack_order:
        manifest_path = root / "nesvor_inputs" / stack / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        groups = manifest.get("group_idx")
        if not isinstance(groups, list) or not groups or any(isinstance(value, bool) or not isinstance(value, int) for value in groups):
            raise ValueError(f"{manifest_path}: group_idx must be a non-empty integer list")
        if len(groups) != len(set(groups)):
            raise ValueError(f"{manifest_path}: group_idx is ambiguous")
        ordered.extend((stack, int(group)) for group in groups)
    slices = sorted(((int(path.name.removesuffix(".nii.gz")), path)
                     for path in final_root.glob("*.nii.gz") if path.name.removesuffix(".nii.gz").isdigit()),
                    key=lambda item: item[0])
    if [index for index, _path in slices] != list(range(len(ordered))):
        raise ValueError(f"{final_root}: final slice IDs must be exactly 0..{len(ordered) - 1}")
    return [{"stack": stack, "original_group_idx": group, "output_slice_index": index,
             "final_output_slice": _record(path)}
            for index, ((stack, group), (_slice_id, path)) in enumerate(zip(ordered, slices))]


def build_cyj11_r3_fit_first_final_pose_bundle(*, baseline_root: str | Path, native_maps_root: str | Path,
                                                endpoint_source_root: str | Path, subject: str, case: str,
                                                output: str | Path) -> Path:
    """Sample r3 endpoint maps at its own immutable parameter-specific final poses."""
    import nibabel as nib
    from ..phase2_clean.fit_first_native_geometry import final_world_grid_original_order, negative_determinant_reindex

    baseline, native_root, endpoint_root, output = (Path(value) for value in (baseline_root, native_maps_root, endpoint_source_root, output))
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    endpoint_manifest_path = endpoint_root / "manifest.json"
    endpoint_manifest = json.loads(endpoint_manifest_path.read_text(encoding="utf-8"))
    if (endpoint_manifest.get("subject"), endpoint_manifest.get("condition"), endpoint_manifest.get("endpoint")) != (subject, case, "central"):
        raise ValueError("r3 Fit-first endpoint subject/case/endpoint mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        planes: list[dict[str, Any]] = []
        for parameter in PARAMETERS:
            volume_path, volume_record = r3_endpoint_volume(endpoint_root, parameter)
            volume_image = nib.load(str(volume_path))
            volume = np.asarray(volume_image.get_fdata(), float)
            mask_path, mask_record = r3_effective_query_mask(endpoint_root, parameter)
            mask_image = nib.load(str(mask_path)); mask = np.asarray(mask_image.get_fdata(), float)
            if mask.shape != volume.shape or not np.allclose(mask_image.affine, volume_image.affine, atol=1e-6, rtol=0):
                raise ValueError(f"{parameter}: endpoint effective query-mask geometry mismatch")
            for group_record in r3_group_records(baseline, parameter):
                stack, group = group_record["stack"], int(group_record["original_group_idx"])
                shape = _r0_shape(native_root, stack, group, parameter)
                input_stack = baseline / "nesvor_inputs" / stack / parameter / "stack.nii.gz"
                flip = negative_determinant_reindex(input_stack, shape)
                final_path = Path(group_record["final_output_slice"]["path"])
                world = final_world_grid_original_order(final_path, shape, flip)
                prediction, geometry = _world_sample(volume, volume_image.affine, world, order=1)
                mask_values, mask_geometry = _world_sample(mask, mask_image.affine, world, order=0)
                query = np.asarray(mask_geometry, bool) & np.isfinite(mask_values) & (np.asarray(mask_values) > .5)
                valid = np.asarray(geometry, bool) & query & np.isfinite(prediction)
                target = staging / "planes" / parameter / stack / f"group{group:03d}.npz"
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target, prediction_ms=np.asarray(prediction, np.float32),
                                    volume_geometry_valid=np.asarray(geometry, bool),
                                    query_effective_mask=np.asarray(query, bool), valid=np.asarray(valid, bool),
                                    world_ras_mm=np.asarray(world[:3], np.float32))
                planes.append({"parameter": parameter, "stack": stack, "original_group_idx": group,
                               "output_slice_index": group_record["output_slice_index"],
                               "final_output_slice": group_record["final_output_slice"],
                               "input_stack": _record(input_stack), "reindex_flip": flip,
                               "path": str(target.relative_to(staging)), "sha256": sha256(target),
                               "volume_ms": volume_record, "effective_query_mask": mask_record})
        write_immutable_json(staging / "manifest.json", {
            "schema": SCHEMA, "subject": subject, "case": case, "endpoint": "central",
            "baseline_root": str(baseline.resolve()), "endpoint_source_manifest": _record(endpoint_manifest_path),
            "coordinate_contract": "r3 group order from current nesvor_inputs manifests -> final NeSVoR output-slice physical RAS -> central volume_ms linear sample -> original native [row,col] ordering",
            "pose_authority": "current r3 NeSVoR output-slice affine; source final slices may remain at the r3 recorded immutable path",
            "planes": planes, "no_training": True, "no_reconstruction": True, "no_formal_metric": True,
        })
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-root", "native-maps-root", "endpoint-source-root", "subject", "case", "output"):
        parser.add_argument("--" + name, required=True)
    args = vars(parser.parse_args())
    print(build_cyj11_r3_fit_first_final_pose_bundle(**args))


if __name__ == "__main__":
    main()
