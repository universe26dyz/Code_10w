"""Immutable common-support metrics for final-pose Fit-first and direct INR bundles."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import sha256, write_immutable_json
from .formal_l100_sparse_manifest import validate_formal_l100_sparse_manifest
from .evaluation import _metrics
from .final_cyj_reporting import PARAMETERS, _load_all_verified_native_cache_groups
from .current_native_central_component_mask import load_validated_current_native_mask_plane

PRIMARY_MASK = "primary_native_mask"
MASKS = (PRIMARY_MASK, "phase4r_l100")
COMMON_SUPPORT_EQUATION = "primary_native_mask AND finite(reference) AND proposed_valid AND fit_first_volume_geometry_valid AND fit_first_query_effective_mask AND finite(proposed, fit_first)"


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def build_final_pose_native_metrics(*, direct_bundle_root: str | Path, fit_first_bundle_root: str | Path, r0_native_maps_root: str | Path, sparse_manifest: str | Path, mask_root: str | Path, output: str | Path) -> Path:
    """Evaluate both immutable endpoints on identical source/mask/valid support."""
    direct_root, fit_root, mask_root, output = (Path(value) for value in (direct_bundle_root, fit_first_bundle_root, mask_root, output))
    direct_manifest = json.loads((direct_root / "manifest.json").read_text(encoding="utf-8"))
    fit_manifest = json.loads((fit_root / "manifest.json").read_text(encoding="utf-8"))
    sparse_identity = validate_formal_l100_sparse_manifest(sparse_manifest)
    if direct_manifest.get("schema") != "direct_native_prediction_bundle/v1":
        raise ValueError("direct bundle schema mismatch")
    if fit_manifest.get("schema") not in ("fit_first_native_prediction_bundle/v2", "fit_first_native_prediction_bundle/v4"):
        raise ValueError("Fit-first bundle schema mismatch")
    if direct_manifest.get("endpoint") != fit_manifest.get("endpoint"):
        raise ValueError("endpoint mismatch: direct and Fit-first final-pose bundles must match")
    subject = sparse_identity["subject"]
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        cache = {stack: _load_all_verified_native_cache_groups(r0_native_maps_root, sparse_manifest, stack) for stack in ("sax", "2ch", "4ch")}
        rows: list[dict[str, Any]] = []
        aggregate: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {parameter: {name: {"reference": [], "proposed": [], "fit_first": []} for name in MASKS} for parameter in PARAMETERS}
        for plane in direct_manifest["planes"]:
            stack, group = plane["stack"], int(plane["original_group_idx"])
            indices = np.flatnonzero(np.asarray(cache[stack]["group_idx"], int) == group)
            if len(indices) != 1:
                raise ValueError(f"{stack}/group{group:03d}: unavailable R0 reference")
            index = int(indices[0])
            with np.load(direct_root / plane["path"], allow_pickle=False) as direct:
                for parameter in PARAMETERS:
                    fit_path = fit_root / "planes" / parameter / stack / f"group{group:03d}.npz"
                    reference, source = np.asarray(cache[stack][f"{parameter}_ms"][index], float), np.asarray(cache[stack]["source_support"][index], bool)
                    masks, mask_provenance = load_validated_current_native_mask_plane(mask_root, parameter=parameter, stack=stack, original_group_idx=group, shape_rc=tuple(reference.shape), source_support=source)
                    with np.load(fit_path, allow_pickle=False) as fit:
                        proposed, fit_first = np.asarray(direct[f"{parameter}_ms"], float), np.asarray(fit["prediction_ms"], float)
                        proposed_valid = np.isfinite(proposed)
                        if fit_manifest.get("schema") == "fit_first_native_prediction_bundle/v4":
                            volume_geometry_valid = np.asarray(fit["volume_geometry_valid"], bool)
                            query_effective_mask = np.asarray(fit["query_effective_mask"], bool)
                            fit_valid = volume_geometry_valid & query_effective_mask & np.isfinite(fit_first)
                        else:
                            volume_geometry_valid, query_effective_mask = np.asarray(fit["valid"], bool), np.ones_like(fit_first, bool)
                            fit_valid = volume_geometry_valid & np.isfinite(fit_first)
                        if not (reference.shape == proposed.shape == fit_first.shape == source.shape == proposed_valid.shape == fit_valid.shape):
                            raise ValueError(f"{parameter}/{stack}/group{group:03d}: native plane geometry mismatch")
                        record: dict[str, Any] = {"parameter": parameter, "stack": stack, "original_group_idx": group, "fit_first_volume_geometry_valid_voxels": int(volume_geometry_valid.sum()), "fit_first_query_effective_mask_voxels": int(query_effective_mask.sum()), "masks": {}}
                        for name in MASKS:
                            support = np.asarray(masks[name], bool) & np.isfinite(reference) & proposed_valid & fit_valid
                            metrics = None if not support.any() else {"proposed": _metrics(reference, proposed, support), "fit_first": _metrics(reference, fit_first, support)}
                            record["masks"][name] = {"common_support_voxels": int(support.sum()), "support_equation": COMMON_SUPPORT_EQUATION if name == PRIMARY_MASK else f"{name} AND finite(reference) AND proposed_valid AND fit_first_volume_geometry_valid AND fit_first_query_effective_mask AND finite(proposed, fit_first)", "metrics": metrics}
                            if support.any():
                                for label, values in (("reference", reference), ("proposed", proposed), ("fit_first", fit_first)):
                                    aggregate[parameter][name][label].append(values[support])
                        record["mask_provenance"] = mask_provenance
                        rows.append(record)
        aggregate_output: dict[str, Any] = {}
        for parameter, masks in aggregate.items():
            aggregate_output[parameter] = {}
            for name, values in masks.items():
                if not values["reference"]:
                    aggregate_output[parameter][name] = {"status": "EMPTY_COMMON_SUPPORT"}
                    continue
                ref = np.concatenate(values["reference"]); support = np.ones(ref.size, bool)
                aggregate_output[parameter][name] = {"common_support_voxels": int(ref.size), "proposed": _metrics(ref, np.concatenate(values["proposed"]), support), "fit_first": _metrics(ref, np.concatenate(values["fit_first"]), support)}
        primary_aggregate = {parameter: aggregate_output[parameter][PRIMARY_MASK] for parameter in PARAMETERS}
        (staging / "aggregate_metrics.json").write_text(json.dumps({"schema": "final_pose_native_common_support_metrics/v3", "primary_mask": PRIMARY_MASK, "primary_support_equation": COMMON_SUPPORT_EQUATION, "primary_aggregate": primary_aggregate, "regions": aggregate_output}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_immutable_json(staging / "manifest.json", {"schema": "final_pose_native_bundle_metrics/v4", "subject": subject, "condition": "L100_fold0", "sparse_manifest_schema": sparse_identity["sparse_manifest_schema"], "sparse_identity": sparse_identity["sparse_identity"], "endpoint": direct_manifest["endpoint"], "direct_bundle": _record(direct_root / "manifest.json"), "fit_first_bundle": _record(fit_root / "manifest.json"), "mask_manifest": _record(mask_root / "manifest.json"), "primary_mask": PRIMARY_MASK, "primary_support_equation": COMMON_SUPPORT_EQUATION, "sensitivity_masks": [name for name in MASKS if name != PRIMARY_MASK], "metric_contract": "current-native primary mask plus final direct INR and final Fit-first bundle on one paired valid support", "per_plane_records": rows, "aggregate_metrics": _record(staging / "aggregate_metrics.json"), "no_existing_metrics_overwritten": True})
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-bundle-root", required=True); parser.add_argument("--fit-first-bundle-root", required=True); parser.add_argument("--r0-native-maps-root", required=True); parser.add_argument("--sparse-manifest", required=True); parser.add_argument("--mask-root", required=True); parser.add_argument("--output", required=True)
    print(build_final_pose_native_metrics(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
