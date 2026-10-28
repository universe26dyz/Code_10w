"""Export current Fit-first raw-volume predictions on NeSVoR's final slice poses.

This is a downstream sampler for a completed reconstruction.  It deliberately
does not reconstruct, refit, or manufacture a fixed-L100 endpoint mask: raw
NeSVoR volume coverage is represented only by interpolation geometry and
finite sampled values.
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
from .cyj11_r3_fit_first_final_pose_bundle import _cache_stack_root, _r0_shape, _record, r3_group_records


SCHEMA = "fit_first_raw_final_pose_native_bundle/v1"
PARAMETERS = ("t1", "t2")
SUPPORT_POLICY = "volume_geometry_and_finite_prediction_only"
STACKS = ("sax", "2ch", "4ch")


def _input_manifest(root: Path, *, subject: str, case: str) -> Path:
    path = root / "input_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("method") != "fit_first" or payload.get("subject_id") != subject:
        raise ValueError("Fit-first input manifest subject/method mismatch")
    recorded_case = payload.get("case_id")
    if recorded_case is not None and recorded_case != case:
        raise ValueError("Fit-first input manifest case mismatch")
    return path


def _sparse_group_mapping(baseline: Path) -> dict[str, dict[int, int]]:
    """Resolve NeSVoR-derived group IDs back to native-cache group IDs.

    The NeSVoR input manifests retain the derived NPZ group numbering.  The
    sparse manifest is the identity authority for cases where those IDs were
    renumbered during sparse extraction.
    """
    input_payload = json.loads((baseline / "input_manifest.json").read_text(encoding="utf-8"))
    record = input_payload.get("sparse_manifest")
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError("Fit-first input manifest lacks sparse-manifest provenance")
    sparse_path = Path(record["path"])
    if not sparse_path.is_file():
        raise FileNotFoundError(sparse_path)
    sparse_payload = json.loads(sparse_path.read_text(encoding="utf-8"))
    stacks = sparse_payload.get("stacks")
    if not isinstance(stacks, dict):
        raise ValueError("sparse manifest lacks stack records")
    result: dict[str, dict[int, int]] = {}
    for stack in STACKS:
        item = stacks.get(stack)
        if not isinstance(item, dict):
            raise ValueError(f"sparse manifest lacks {stack} record")
        records = item.get("derived_to_original_group_idx")
        if isinstance(records, list):
            mapping: dict[int, int] = {}
            for entry in records:
                if not isinstance(entry, dict) or not isinstance(entry.get("derived_group_idx"), int) or not isinstance(entry.get("original_group_idx"), int):
                    raise ValueError(f"{stack}: malformed derived-to-original mapping")
                mapping[int(entry["derived_group_idx"])] = int(entry["original_group_idx"])
            result[stack] = mapping
            continue
        if "derived_observation" in item:
            retained = item.get("retained_original_group_ids")
            if not isinstance(retained, list) or not all(isinstance(value, int) for value in retained):
                raise ValueError(f"{stack}: derived sparse record lacks retained original group IDs")
            result[stack] = {index: int(group) for index, group in enumerate(retained)}
        else:
            result[stack] = {}
    return result


def build_current_fit_first_final_pose_bundle(*, baseline_root: str | Path, native_maps_root: str | Path,
                                              subject: str, case: str, output: str | Path) -> Path:
    """Sample current denormalised NeSVoR volumes at parameter-specific final poses."""
    import nibabel as nib
    from ..phase2_clean.fit_first_native_geometry import final_world_grid_original_order, negative_determinant_reindex

    baseline, native_root, output = (Path(value) for value in (baseline_root, native_maps_root, output))
    input_manifest = _input_manifest(baseline, subject=subject, case=case)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    group_mapping = _sparse_group_mapping(baseline)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        planes: list[dict[str, Any]] = []
        for parameter in PARAMETERS:
            volume_path = baseline / "nesvor" / parameter / "volume_ms.nii.gz"
            if not volume_path.is_file():
                raise FileNotFoundError(volume_path)
            image = nib.load(str(volume_path))
            volume = np.asarray(image.get_fdata(), float)
            for group_record in r3_group_records(baseline, parameter):
                stack = group_record["stack"]
                derived_group = int(group_record["original_group_idx"])
                group = group_mapping[stack].get(derived_group, derived_group)
                shape = _r0_shape(native_root, stack, group, parameter)
                input_stack = baseline / "nesvor_inputs" / stack / parameter / "stack.nii.gz"
                if not input_stack.is_file():
                    raise FileNotFoundError(input_stack)
                world = final_world_grid_original_order(
                    Path(group_record["final_output_slice"]["path"]), shape,
                    negative_determinant_reindex(input_stack, shape),
                )
                prediction, geometry = _world_sample(volume, image.affine, world, order=1)
                valid = np.asarray(geometry, bool) & np.isfinite(prediction)
                target = staging / "planes" / parameter / stack / f"group{group:03d}.npz"
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    target,
                    prediction_ms=np.asarray(prediction, np.float32),
                    volume_geometry_valid=np.asarray(geometry, bool),
                    valid=np.asarray(valid, bool),
                    world_ras_mm=np.asarray(world[:3], np.float32),
                )
                planes.append({
                    "parameter": parameter, "stack": stack, "original_group_idx": group,
                    "output_slice_index": group_record["output_slice_index"],
                    "final_output_slice": group_record["final_output_slice"],
                    "input_stack": _record(input_stack),
                    "path": str(target.relative_to(staging)), "sha256": sha256(target),
                    "volume_ms": _record(volume_path),
                })
        write_immutable_json(staging / "manifest.json", {
            "schema": SCHEMA, "subject": subject, "case": case, "endpoint": "central",
            "fit_first_input_manifest": _record(input_manifest),
            "coordinate_contract": "current NeSVoR input stack -> parameter-specific final output-slice physical RAS -> current denormalised volume_ms linear sample -> original native [row,col] ordering",
            "pose_authority": "current NeSVoR final output-slice affine",
            "support_policy": SUPPORT_POLICY,
            "valid_definition": "volume_geometry_valid AND finite(prediction_ms)",
            "planes": planes,
            "no_training": True, "no_reconstruction": True, "no_refitting": True,
        })
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-root", "native-maps-root", "subject", "case", "output"):
        parser.add_argument("--" + name, required=True)
    args = vars(parser.parse_args())
    print(build_current_fit_first_final_pose_bundle(
        baseline_root=args["baseline_root"], native_maps_root=args["native_maps_root"],
        subject=args["subject"], case=args["case"], output=args["output"],
    ))


if __name__ == "__main__":
    main()
