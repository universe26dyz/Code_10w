"""Read-only provenance audit for Fit-first final NeSVoR native-plane geometry."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import sha256, write_immutable_json


STACKS = ("sax", "2ch", "4ch")
PARAMETERS = ("t1", "t2")


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _final_slice_paths(root: Path) -> list[Path]:
    paths = sorted((root / "slices").glob("[0-9]*.nii.gz"), key=lambda path: int(path.name.split(".", 1)[0]))
    if not paths:
        raise FileNotFoundError(f"missing NeSVoR --output-slices under {root / 'slices'}")
    observed = [int(path.name.split(".", 1)[0]) for path in paths]
    if observed != list(range(len(paths))):
        raise ValueError(f"NeSVoR output-slices must be contiguous zero-based IDs, got {observed}")
    return paths


def _original_groups(baseline_root: Path) -> list[tuple[str, int, int]]:
    """Recover the stable NeSVoR input-stack order from exported group manifests."""
    result: list[tuple[str, int, int]] = []
    output_index = 0
    for stack in STACKS:
        input_manifest = baseline_root / "nesvor_inputs" / stack / "manifest.json"
        native_maps = baseline_root / "native_maps" / stack / "native_maps.npz"
        if not input_manifest.is_file() or not native_maps.is_file():
            raise FileNotFoundError(f"{stack}: missing immutable NeSVoR input/group provenance")
        declared = [int(value) for value in json.loads(input_manifest.read_text(encoding="utf-8")).get("group_idx", [])]
        with np.load(native_maps, allow_pickle=False) as maps:
            actual = [int(value) for value in maps["group_idx"]]
        if declared != actual or len(declared) != len(set(declared)):
            raise ValueError(f"{stack}: NeSVoR input group order is not proven by native-map provenance")
        for group in declared:
            result.append((stack, group, output_index)); output_index += 1
    return result


def _r0_geometry(r0_root: Path, stack: str, group: int) -> tuple[np.ndarray, tuple[int, int, int]]:
    geometry_path, maps_path = r0_root / stack / "geometry.npz", r0_root / stack / "native_maps.npz"
    with np.load(geometry_path, allow_pickle=False) as geometry, np.load(maps_path, allow_pickle=False) as maps:
        groups = [int(value) for value in maps["group_idx"]]
        if groups.count(group) != 1:
            raise ValueError(f"{stack}: R0 geometry lacks original group {group}")
        index = groups.index(group)
        affine_lps_rc = np.asarray(geometry["affine_lps_rc"][index], float)
        shape_rc = tuple(int(value) for value in np.asarray(maps["t1_ms"])[index].shape)
    affine_ras_rc = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine_lps_rc
    return affine_ras_rc, (*shape_rc, 1)


def _equivalent_r0_and_final(r0_affine_ras: np.ndarray, r0_shape: tuple[int, int, int], final_affine_ras: np.ndarray, final_shape: tuple[int, ...]) -> bool:
    """Deliberately strict: a fallback is allowed only for identical voxel geometry."""
    return tuple(final_shape) == tuple(r0_shape) and bool(np.allclose(r0_affine_ras, final_affine_ras, rtol=0.0, atol=1e-5))


def audit_fit_first_native_poses(*, baseline_root: str | Path, r0_native_maps_root: str | Path, output: str | Path) -> Path:
    """Freeze final optimized NeSVoR geometry; never compute or overwrite metrics."""
    import nibabel as nib

    baseline_root, r0_native_maps_root, output = Path(baseline_root), Path(r0_native_maps_root), Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    ordered_groups = _original_groups(baseline_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        parameters: dict[str, Any] = {}
        for parameter in PARAMETERS:
            root = baseline_root / "nesvor" / parameter
            result_path, model_path = root / "result.json", root / "model.pt"
            if not result_path.is_file() or not model_path.is_file():
                raise FileNotFoundError(f"{parameter}: missing NeSVoR result.json/model.pt")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("registration") != "stack" or result.get("no_transformation_optimization") is not False:
                raise ValueError(f"{parameter}: result.json does not prove final stack transformation optimization")
            slices = _final_slice_paths(root)
            if len(slices) != len(ordered_groups):
                raise ValueError(f"{parameter}: output-slices count does not match immutable input group lattice")
            groups: list[dict[str, Any]] = []
            for stack, group, output_index in ordered_groups:
                final_slice = slices[output_index]
                image = nib.load(str(final_slice))
                r0_affine, r0_shape = _r0_geometry(r0_native_maps_root, stack, group)
                equivalent = _equivalent_r0_and_final(r0_affine, r0_shape, np.asarray(image.affine, float), tuple(image.shape))
                groups.append({"stack": stack, "original_group_idx": group, "output_slice_index": output_index, "authoritative_geometry": "nesvor_output_slices_final_optimized_affine", "final_output_slice": _record(final_slice), "final_shape_xyz": list(image.shape), "final_affine_ras_mm": np.asarray(image.affine, float).tolist(), "r0_shape_rc1": list(r0_shape), "r0_affine_ras_mm": r0_affine.tolist(), "r0_affine_mathematically_equivalent": equivalent, "fallback_rule": "R0/native affine is prohibited unless this record is mathematically equivalent"})
            parameters[parameter] = {"result": _record(result_path), "model": _record(model_path), "model_pose_role": "model.pt is checkpoint provenance; per-group final geometry is authoritatively encoded by --output-slices NIfTI affines", "groups": groups}
        write_immutable_json(staging / "manifest.json", {"schema": "fit_first_native_pose_audit/v1", "baseline_root": str(baseline_root.resolve()), "r0_native_maps_root": str(r0_native_maps_root.resolve()), "input_group_order": "nesvor_inputs stack order sax -> 2ch -> 4ch; group_idx order frozen in each immutable export manifest", "fit_first_endpoint_policy": "must_use_final_nesvor_output_slices_unless_equivalence_is_proven", "scmr_comparison": "historical registered_ms and synthetic_ms are both expressed on the parameter-specific registered/final-pose slice geometry; this audit freezes the same final-pose source for Fit-first native-plane endpoints", "parameters": parameters, "no_training": True, "no_reconstruction": True, "no_metrics_modified": True, "no_formal_export_modified": True})
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
