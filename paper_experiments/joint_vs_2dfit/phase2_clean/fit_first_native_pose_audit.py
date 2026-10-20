"""Fail-closed provenance audit for Fit-first final-NeSVoR native planes.

This does not reconstruct, train, render, or modify metrics. It proves the
geometry bridge needed before a final-pose volume endpoint is allowed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import sha256, write_immutable_json
from .fit_first_native_geometry import (
    initial_world_grid_original_order,
    load_final_output_slices,
    load_initial_stack_slices,
    negative_determinant_reindex,
    pose_delta,
)

STACKS = ("sax", "2ch", "4ch")
PARAMETERS = ("t1", "t2")
_SCALE = {"t1": 2500.0, "t2": 200.0}
_GEOMETRY_EQUIVALENCE_TOL_MM = 1e-3
_VALUE_CORRESPONDENCE_TOL = 1e-5


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _slice_paths(root: Path) -> list[Path]:
    paths = sorted((root / "slices").glob("[0-9]*.nii.gz"), key=lambda path: int(path.name.split(".", 1)[0]))
    observed = [int(path.name.split(".", 1)[0]) for path in paths]
    if not paths or observed != list(range(len(paths))):
        raise ValueError(f"{root}: --output-slices must be contiguous zero-based NIfTI IDs")
    return paths


def _ordered_groups(baseline_root: Path) -> list[tuple[str, int, int]]:
    records: list[tuple[str, int, int]] = []
    output_index = 0
    for stack in STACKS:
        path = baseline_root / "nesvor_inputs" / stack / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"{stack}: missing exported NeSVoR input manifest")
        groups = [int(value) for value in json.loads(path.read_text(encoding="utf-8")).get("group_idx", [])]
        if not groups or len(groups) != len(set(groups)):
            raise ValueError(f"{stack}: immutable export group order is absent or ambiguous")
        for group in groups:
            records.append((stack, group, output_index))
            output_index += 1
    return records


def _r0(root: Path, stack: str, group: int, parameter: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(root / stack / "geometry.npz", allow_pickle=False) as geometry, np.load(root / stack / "native_maps.npz", allow_pickle=False) as maps:
        groups = [int(value) for value in maps["group_idx"]]
        if groups.count(group) != 1:
            raise ValueError(f"{stack}: R0 lacks exactly one original group {group}")
        index = groups.index(group)
        values = np.asarray(maps[f"{parameter}_ms"][index], float)
        affine_lps = np.asarray(geometry["affine_lps_rc"][index], float)
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine_lps, values


def _r0_world(affine: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows, cols = np.indices(shape, dtype=float)
    voxels = np.stack((rows, cols, np.zeros_like(rows), np.ones_like(rows)), axis=0)
    return (affine @ voxels.reshape(4, -1)).reshape(4, *shape)


def _normalised(values_ms: np.ndarray, parameter: str) -> np.ndarray:
    values = np.asarray(values_ms, np.float32) / _SCALE[parameter]
    return np.where(np.isfinite(values), values, 0.0)


def _initial_values_original_order(initial: Any, shape: tuple[int, int], flip_row: bool) -> np.ndarray:
    values = initial.image.squeeze(0).detach().cpu().numpy()  # internal [col,row]
    rows, cols = np.indices(shape)
    return values[cols, shape[0] - 1 - rows if flip_row else rows]


def _final_values_original_order(path: Path, shape: tuple[int, int], flip_row: bool) -> np.ndarray:
    import nibabel as nib

    values = np.asarray(nib.load(str(path)).get_fdata(), float)
    if values.shape != (*shape, 1):
        raise ValueError(f"{path}: final output-slice shape {values.shape} does not preserve source {(*shape, 1)}")
    rows, cols = np.indices(shape)
    return values[shape[0] - 1 - rows if flip_row else rows, cols, 0]


def _max_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def _result_proves_pose_optimisation(path: Path) -> None:
    result = json.loads(path.read_text(encoding="utf-8"))
    args = result.get("args", result)
    registration = args.get("registration", result.get("registration"))
    disabled = args.get("no_transformation_optimization", result.get("no_transformation_optimization"))
    if registration != "stack" or disabled is not False:
        raise ValueError(f"{path}: does not prove --registration stack with transformation optimisation enabled")


def audit_fit_first_native_poses(*, baseline_root: str | Path, r0_native_maps_root: str | Path, output: str | Path) -> Path:
    """Audit representation changes separately from true final rigid motion."""
    import nibabel as nib

    baseline_root, r0_root, output = Path(baseline_root), Path(r0_native_maps_root), Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    ordered = _ordered_groups(baseline_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        parameters: dict[str, Any] = {}
        gate_failures: list[str] = []
        finals_by_parameter: dict[str, list[Any]] = {}
        for parameter in PARAMETERS:
            root = baseline_root / "nesvor" / parameter
            result_path, model_path = root / "result.json", root / "model.pt"
            if not result_path.is_file() or not model_path.is_file():
                raise FileNotFoundError(f"{parameter}: result.json/model.pt required for provenance")
            _result_proves_pose_optimisation(result_path)
            final_paths, finals = _slice_paths(root), load_final_output_slices(root / "slices")
            if len(final_paths) != len(ordered) or len(finals) != len(ordered):
                raise ValueError(f"{parameter}: output-slices do not match complete original group lattice")
            initials = {stack: load_initial_stack_slices(baseline_root / "nesvor_inputs" / stack / parameter / "stack.nii.gz") for stack in STACKS}
            local_group_order = {stack: [group for name, group, _ in ordered if name == stack] for stack in STACKS}
            descriptors: dict[str, dict[str, Any]] = {}
            for stack in STACKS:
                first_group = local_group_order[stack][0]
                _, values = _r0(r0_root, stack, first_group, parameter)
                descriptors[stack] = negative_determinant_reindex(baseline_root / "nesvor_inputs" / stack / parameter / "stack.nii.gz", tuple(values.shape))
            groups: list[dict[str, Any]] = []
            for stack, group, output_index in ordered:
                r0_affine, r0_values = _r0(r0_root, stack, group, parameter)
                shape = tuple(int(value) for value in r0_values.shape)
                initial = initials[stack][local_group_order[stack].index(group)]
                final, final_path, descriptor = finals[output_index], final_paths[output_index], descriptors[stack]
                expected = _normalised(r0_values, parameter)
                initial_world = initial_world_grid_original_order(initial, shape, descriptor)
                r0_world = _r0_world(r0_affine, shape)[:3].transpose(1, 2, 0)
                expected_spacing = [float(np.linalg.norm(r0_affine[:3, axis])) for axis in (0, 1)]
                initial_value_error = _max_error(_initial_values_original_order(initial, shape, descriptor["flip_original_row"]), expected)
                final_value_error = _max_error(_final_values_original_order(final_path, shape, descriptor["flip_original_row"]), expected)
                geometry_error = _max_error(initial_world, r0_world)
                corners = ((0, 0), (0, shape[1] - 1), (shape[0] - 1, 0), (shape[0] - 1, shape[1] - 1))
                corner_error = float(max(np.linalg.norm(initial_world[row, col] - r0_world[row, col]) for row, col in corners))
                center = ((shape[0] - 1) // 2, (shape[1] - 1) // 2)
                center_error = float(np.linalg.norm(initial_world[center] - r0_world[center]))
                spacing_error = float(max(abs(initial.resolution_x - expected_spacing[0]), abs(initial.resolution_y - expected_spacing[1])))
                passed = (geometry_error <= _GEOMETRY_EQUIVALENCE_TOL_MM and spacing_error <= _GEOMETRY_EQUIVALENCE_TOL_MM
                          and initial_value_error <= _VALUE_CORRESPONDENCE_TOL and final_value_error <= _VALUE_CORRESPONDENCE_TOL)
                if not passed:
                    gate_failures.append(f"{parameter}/{stack}/group{group:03d}")
                final_affine = np.asarray(nib.load(str(final_path)).affine, float)
                raw_equivalent = tuple(nib.load(str(final_path)).shape) == (*shape, 1) and bool(np.allclose(r0_affine, final_affine, rtol=0, atol=1e-5))
                groups.append({
                    "stack": stack, "original_group_idx": group, "output_slice_index": output_index,
                    "authoritative_final_geometry": "nesvor_output_slices_final_optimized_affine", "final_output_slice": _record(final_path), "final_affine_ras_mm": final_affine.tolist(),
                    "r0_vs_initial": {"shape_rc": list(shape), "expected_spacing_rc_mm": expected_spacing, "initial_spacing_rc_mm": [float(initial.resolution_x), float(initial.resolution_y)], "max_world_error_mm": geometry_error, "corner_max_error_mm": corner_error, "center_error_mm": center_error, "spacing_error_mm": spacing_error, "max_normalized_value_error": initial_value_error, "status": "PASS_REPRESENTATION_EQUIVALENT" if geometry_error <= _GEOMETRY_EQUIVALENCE_TOL_MM and spacing_error <= _GEOMETRY_EQUIVALENCE_TOL_MM and initial_value_error <= _VALUE_CORRESPONDENCE_TOL else "RECONSTRUCTION_INPUT_GEOMETRY_MISMATCH"},
                    "initial_vs_final": pose_delta(initial, final), "output_slice_value_validation": {"max_normalized_value_error": final_value_error, "value_tolerance": _VALUE_CORRESPONDENCE_TOL, "status": "PASS" if final_value_error <= _VALUE_CORRESPONDENCE_TOL else "FAIL"}, "reindex_flip": descriptor,
                    "raw_affine_diagnostic": {"r0_vs_final_mathematically_equivalent": raw_equivalent, "raw_affine_axis_convention_difference_not_motion": not raw_equivalent},
                })
            parameters[parameter] = {"result": _record(result_path), "model": _record(model_path), "model_pose_role": "model.pt is checkpoint provenance only; final per-group pose is the corresponding --output-slices transformation", "groups": groups}
            finals_by_parameter[parameter] = finals
        for index, (t1, t2) in enumerate(zip(parameters["t1"]["groups"], parameters["t2"]["groups"])):
            delta = pose_delta(finals_by_parameter["t1"][index], finals_by_parameter["t2"][index])
            t1["t1_final_vs_t2_final"] = delta; t2["t1_final_vs_t2_final"] = delta
        status = "RECONSTRUCTION_INPUT_GEOMETRY_VALID" if not gate_failures else "RECONSTRUCTION_INPUT_GEOMETRY_MISMATCH"
        write_immutable_json(staging / "manifest.json", {
            "schema": "fit_first_native_pose_audit/v3", "baseline_root": str(baseline_root.resolve()), "r0_native_maps_root": str(r0_root.resolve()), "input_group_order": "sax -> 2ch -> 4ch; immutable exported group_idx order",
            "reconstruction_input_geometry_gate": {"status": status, "geometry_equivalence_tolerance_mm": _GEOMETRY_EQUIVALENCE_TOL_MM, "value_correspondence_tolerance": _VALUE_CORRESPONDENCE_TOL, "failures": gate_failures, "pass_action": "final-pose endpoint bundle is permitted; no retraining required", "fail_action": "fail closed: do not build formal endpoint and do not change reconstruction/training"},
            "fit_first_endpoint_policy": "only final NeSVoR output-slice pose may define Fit-first native prediction; original R0 affine is reference geometry only", "scmr_comparison": "historical registered_ms/synthetic_ms uses parameter-specific registered slice geometry; this endpoint has the same final-pose semantics", "parameters": parameters, "no_training": True, "no_reconstruction": True, "no_metrics_modified": True,
        })
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True); parser.add_argument("--r0-native-maps-root", required=True); parser.add_argument("--output", required=True)
    print(audit_fit_first_native_poses(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
