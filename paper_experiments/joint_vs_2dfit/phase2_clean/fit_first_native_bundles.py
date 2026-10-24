"""Immutable Fit-first final-pose native-plane predictions from NeSVoR volumes."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import sha256, write_immutable_json
from .evaluation import _world_sample
from .fit_first_native_geometry import final_world_grid_original_order
from .fit_first_route_contract import INTENSITY_AUDIT_SCHEMA, resolve_fit_first_route
from .matched_output import verified_fit_first_effective_query_mask


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _r0_shape(root: Path, stack: str, group: int, parameter: str) -> tuple[int, int]:
    with np.load(root / stack / "native_maps.npz", allow_pickle=False) as maps:
        groups = [int(value) for value in maps["group_idx"]]
        if groups.count(group) != 1:
            raise ValueError(f"{stack}: R0 group {group} unavailable")
        return tuple(int(value) for value in np.asarray(maps[f"{parameter}_ms"][groups.index(group)]).shape)


def build_fit_first_native_prediction_bundle(*, audit_root: str | Path | None = None, baseline_root: str | Path, r0_native_maps_root: str | Path, endpoint: str, endpoint_source_root: str | Path, output: str | Path) -> Path:
    """Sample each parameter's volume on its proven final NeSVoR pose.

    ``--output-slices`` validates pose/value correspondence only: predictions
    are always sampled from the corresponding denormalised ``volume_ms``.
    """
    import nibabel as nib

    baseline_root, r0_root, endpoint_root, output = (Path(value) for value in (baseline_root, r0_native_maps_root, endpoint_source_root, output))
    audit_path_root = None if audit_root is None else Path(audit_root)
    route = resolve_fit_first_route(baseline_root, legacy_audit_root=audit_path_root)
    if route.route_kind == "legacy_fit_first":
        if audit_path_root is None:
            raise ValueError("legacy Fit-first requires an explicit audit_root")
        audit_root = audit_path_root
        audit_path = audit_root / "manifest.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("schema") != "fit_first_native_pose_audit/v3":
            raise ValueError("audit_root must contain fit_first_native_pose_audit/v3")
        if audit.get("reconstruction_input_geometry_gate", {}).get("status") != "RECONSTRUCTION_INPUT_GEOMETRY_VALID":
            raise ValueError("RECONSTRUCTION_INPUT_GEOMETRY_MISMATCH: refusing final-pose bundle")
    else:
        audit_root = baseline_root
        audit_path, audit = route.pose_audit_path, route.pose_audit
        if audit_path is None or audit is None or audit.get("schema") != INTENSITY_AUDIT_SCHEMA:
            raise ValueError("intensity-preserving final-pose audit is unavailable")
    endpoint_manifest_path = endpoint_root / "manifest.json"
    if endpoint not in ("central", "psf128") or not endpoint_manifest_path.is_file():
        raise ValueError("Fit-first final-pose bundle requires a formal central or psf128 endpoint source")
    endpoint_manifest = json.loads(endpoint_manifest_path.read_text(encoding="utf-8"))
    if endpoint_manifest.get("schema") != "phase4r_fixed_l100_fit_first_export_bundle/v2" or endpoint_manifest.get("endpoint") != endpoint:
        raise ValueError("Fit-first endpoint source schema/endpoint mismatch")
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        planes: list[dict[str, Any]] = []
        for parameter in ("t1", "t2"):
            volume_path = endpoint_root / parameter / "volume_ms.nii.gz"
            if not volume_path.is_file():
                raise FileNotFoundError(f"{parameter}: missing denormalised NeSVoR volume {volume_path}")
            image = nib.load(str(volume_path))
            volume = image.get_fdata()
            effective_mask_path, effective_mask_record = verified_fit_first_effective_query_mask(endpoint_root, parameter)
            effective_mask_image = nib.load(str(effective_mask_path))
            effective_mask = effective_mask_image.get_fdata()
            if effective_mask.shape != volume.shape or not np.allclose(effective_mask_image.affine, image.affine, atol=1e-6, rtol=0):
                raise ValueError(f"{parameter}: formal effective query mask geometry differs from endpoint volume")
            if not np.all(np.isin(effective_mask, (0.0, 1.0))):
                raise ValueError(f"{parameter}: formal effective query mask is not binary")
            if route.route_kind == "legacy_fit_first":
                group_records = audit["parameters"][parameter]["groups"]
            else:
                inputs = json.loads((baseline_root / "registered_quantitative_inputs" / "manifest.json").read_text(encoding="utf-8"))
                group_records = []
                for input_record in inputs.get("parameters", {}).get(parameter, []):
                    index, stack = int(input_record["index"]), input_record["stack"]
                    final_path = route.final_slices[parameter] / f"{index}.nii.gz"
                    if not final_path.is_file():
                        raise FileNotFoundError(f"{parameter}/{stack}: Stage-2 final slice is missing")
                    group_records.append({"stack": stack, "original_group_idx": int(input_record["standardized_group_idx"]), "standardized_group_idx": int(input_record["standardized_group_idx"]), "original_prestandardization_group_idx": int(input_record["original_prestandardization_group_idx"]), "output_slice_index": index, "final_output_slice": _record(final_path), "reindex_flip": {"flip_original_row": bool(np.linalg.det(np.asarray(nib.load(str(final_path)).affine, float)[:3, :3]) < 0)}})
            for group_record in group_records:
                stack, group = group_record["stack"], int(group_record["original_group_idx"])
                shape = _r0_shape(r0_root, stack, group, parameter)
                final_path = Path(group_record["final_output_slice"]["path"])
                world = final_world_grid_original_order(final_path, shape, group_record["reindex_flip"])
                prediction, volume_geometry_valid = _world_sample(volume, image.affine, world, order=1)
                effective_mask_values, effective_mask_geometry_valid = _world_sample(effective_mask, effective_mask_image.affine, world, order=0)
                query_effective_mask = np.asarray(effective_mask_geometry_valid, bool) & np.isfinite(effective_mask_values) & (np.asarray(effective_mask_values) > .5)
                valid = np.asarray(volume_geometry_valid, bool) & query_effective_mask & np.isfinite(prediction)
                target = staging / "planes" / parameter / stack / f"group{group:03d}.npz"
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target, prediction_ms=np.asarray(prediction, np.float32), volume_geometry_valid=np.asarray(volume_geometry_valid, bool), query_effective_mask=np.asarray(query_effective_mask, bool), valid=np.asarray(valid, bool), world_ras_mm=np.asarray(world[:3], np.float32))
                plane_record = {"parameter": parameter, "stack": stack, "original_group_idx": group, "shape_rc": list(shape), "path": str(target.relative_to(staging)), "sha256": sha256(target), "volume_ms": _record(volume_path), "formal_effective_query_mask": effective_mask_record, "effective_mask_native_mapping": "nearest-neighbor order=0 on original native plane", "final_output_slice": group_record["final_output_slice"], "output_slice_index": group_record["output_slice_index"], "reindex_flip": group_record["reindex_flip"]}
                if route.route_kind == "intensity_preserving_fit_first":
                    plane_record.update({"standardized_group_idx": group_record["standardized_group_idx"], "original_prestandardization_group_idx": group_record["original_prestandardization_group_idx"]})
                planes.append(plane_record)
        schema = "fit_first_native_prediction_bundle/v2" if route.route_kind == "legacy_fit_first" else "fit_first_native_prediction_bundle/v4"
        write_immutable_json(staging / "manifest.json", {"schema": schema, "route": route.provenance(), "audit_manifest": _record(audit_path), "reconstruction_input_geometry_gate": "RECONSTRUCTION_INPUT_GEOMETRY_VALID" if route.route_kind == "legacy_fit_first" else "REGISTERED_QUANTITATIVE_INPUTS_VALID", "endpoint": endpoint, "endpoint_source_manifest": _record(endpoint_manifest_path), "coordinate_contract": "original native [row,col] lattice -> proven NeSVoR reindex/negative-determinant flip -> parameter-specific final output-slice physical RAS -> endpoint-specific volume_ms linear sample -> original [row,col] ordering", "effective_mask_contract": "formal endpoint effective query mask SHA-verified and mapped to each native plane with nearest-neighbour order=0; valid = volume_geometry_valid AND query_effective_mask AND finite(prediction)", "output_slices_role": "geometry only; Stage-1 registration pixels are never quantitative predictions; prediction values are sampled from formal Stage-2 endpoint volume_ms", "parameter_pose_policy": "T1 and T2 use independently optimised/frozen parameter-specific final poses", "planes": planes, "no_training": True, "no_reconstruction": True, "no_metrics_modified": True})
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root"); parser.add_argument("--baseline-root", required=True); parser.add_argument("--r0-native-maps-root", required=True); parser.add_argument("--endpoint", choices=("central", "psf128"), required=True); parser.add_argument("--endpoint-source-root", required=True); parser.add_argument("--output", required=True)
    print(build_fit_first_native_prediction_bundle(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
