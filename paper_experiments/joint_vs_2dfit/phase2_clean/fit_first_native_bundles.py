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


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _r0_shape(root: Path, stack: str, group: int, parameter: str) -> tuple[int, int]:
    with np.load(root / stack / "native_maps.npz", allow_pickle=False) as maps:
        groups = [int(value) for value in maps["group_idx"]]
        if groups.count(group) != 1:
            raise ValueError(f"{stack}: R0 group {group} unavailable")
        return tuple(int(value) for value in np.asarray(maps[f"{parameter}_ms"][groups.index(group)]).shape)


def build_fit_first_native_prediction_bundle(*, audit_root: str | Path, baseline_root: str | Path, r0_native_maps_root: str | Path, endpoint: str, endpoint_source_root: str | Path, output: str | Path) -> Path:
    """Sample each parameter's volume on its proven final NeSVoR pose.

    ``--output-slices`` validates pose/value correspondence only: predictions
    are always sampled from the corresponding denormalised ``volume_ms``.
    """
    import nibabel as nib

    audit_root, baseline_root, r0_root, endpoint_root, output = (Path(value) for value in (audit_root, baseline_root, r0_native_maps_root, endpoint_source_root, output))
    audit_path = audit_root / "manifest.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema") != "fit_first_native_pose_audit/v3":
        raise ValueError("audit_root must contain fit_first_native_pose_audit/v3")
    if audit.get("reconstruction_input_geometry_gate", {}).get("status") != "RECONSTRUCTION_INPUT_GEOMETRY_VALID":
        raise ValueError("RECONSTRUCTION_INPUT_GEOMETRY_MISMATCH: refusing final-pose bundle")
    endpoint_manifest_path = endpoint_root / "manifest.json"
    if endpoint not in ("central", "psf128") or not endpoint_manifest_path.is_file():
        raise ValueError("Fit-first final-pose bundle requires a formal central or psf128 endpoint source")
    endpoint_manifest = json.loads(endpoint_manifest_path.read_text(encoding="utf-8"))
    if endpoint_manifest.get("schema") != "phase4r_fixed_l100_fit_first_export_bundle/v1" or endpoint_manifest.get("endpoint") != endpoint:
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
            for group_record in audit["parameters"][parameter]["groups"]:
                stack, group = group_record["stack"], int(group_record["original_group_idx"])
                shape = _r0_shape(r0_root, stack, group, parameter)
                final_path = Path(group_record["final_output_slice"]["path"])
                world = final_world_grid_original_order(final_path, shape, group_record["reindex_flip"])
                prediction, valid = _world_sample(volume, image.affine, world, order=1)
                target = staging / "planes" / parameter / stack / f"group{group:03d}.npz"
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target, prediction_ms=np.asarray(prediction, np.float32), valid=np.asarray(valid, bool), world_ras_mm=np.asarray(world[:3], np.float32))
                planes.append({"parameter": parameter, "stack": stack, "original_group_idx": group, "shape_rc": list(shape), "path": str(target.relative_to(staging)), "sha256": sha256(target), "volume_ms": _record(volume_path), "final_output_slice": group_record["final_output_slice"], "output_slice_index": group_record["output_slice_index"], "reindex_flip": group_record["reindex_flip"]})
        write_immutable_json(staging / "manifest.json", {"schema": "fit_first_native_prediction_bundle/v2", "audit_manifest": _record(audit_path), "reconstruction_input_geometry_gate": "RECONSTRUCTION_INPUT_GEOMETRY_VALID", "endpoint": endpoint, "endpoint_source_manifest": _record(endpoint_manifest_path), "coordinate_contract": "original native [row,col] lattice -> proven NeSVoR reindex/negative-determinant flip -> parameter-specific final output-slice physical RAS -> endpoint-specific volume_ms linear sample -> original [row,col] ordering", "output_slices_role": "geometry and value-correspondence validation only; prediction values are sampled from formal endpoint volume_ms", "parameter_pose_policy": "T1 and T2 use their independently optimised final poses", "planes": planes, "no_training": True, "no_reconstruction": True, "no_metrics_modified": True})
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", required=True); parser.add_argument("--baseline-root", required=True); parser.add_argument("--r0-native-maps-root", required=True); parser.add_argument("--endpoint", choices=("central", "psf128"), required=True); parser.add_argument("--endpoint-source-root", required=True); parser.add_argument("--output", required=True)
    print(build_fit_first_native_prediction_bundle(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
