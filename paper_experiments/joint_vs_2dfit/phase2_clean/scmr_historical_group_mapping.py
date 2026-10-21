"""Freeze explicit historical-SCMR to current-R0 group identities.

Only a proved physical-lattice identity or discrete reindex is accepted.  This
module neither samples nor resamples values; it publishes provenance only.
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
from .final_cyj_reporting import _load_all_verified_native_cache_groups
from .scmr_six_input_bundle import EXPECTED_CYJ_L100_GROUPS, PARAMETERS, STACKS, _array_and_affine, exact_grid_relation


def _record(path: str | Path) -> dict[str, str]:
    item = Path(path)
    if not item.is_file():
        raise FileNotFoundError(item)
    return {"path": str(item.resolve()), "sha256": sha256(item)}


def _historical_original_records(path: str | Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    for item in payload.get("slices", []):
        key = (str(item.get("parameter", "")).lower(), str(item.get("stack", "")), int(item["native_slice_id"]))
        if key[0] not in PARAMETERS or key[1] not in STACKS:
            continue
        if key in records:
            raise ValueError(f"duplicate historical original identity: {key}")
        if not item.get("original_ms"):
            raise ValueError(f"historical {key} lacks explicit original_ms")
        records[key] = item
    return records


def _target_affine(cache: dict[str, Any], index: int) -> np.ndarray:
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ np.asarray(cache["affine_lps_rc"][index], float)


def _relation_record(relation: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in relation.items() if key != "source_indices"}


def discover_scmr_historical_group_mapping(*, historical_common_mask_manifest: str | Path, r0_native_maps_root: str | Path, sparse_manifest: str | Path) -> dict[str, Any]:
    """Report exact historical-to-current matches without creating an artifact."""
    historical = _historical_original_records(historical_common_mask_manifest)
    current = {stack: _load_all_verified_native_cache_groups(r0_native_maps_root, sparse_manifest, stack) for stack in STACKS}
    found: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    duplicate_current: list[dict[str, Any]] = []
    coverage = {parameter: {} for parameter in PARAMETERS}

    for parameter in PARAMETERS:
        field = f"{parameter}_ms"
        for stack in STACKS:
            expected = EXPECTED_CYJ_L100_GROUPS[stack]
            records = [(key, value) for key, value in historical.items() if key[:2] == (parameter, stack)]
            groups = [int(value) for value in current[stack]["group_idx"]]
            coverage[parameter][stack] = {"expected": expected, "historical_records": len(records), "current_groups": len(groups), "matched": 0}
            for key, item in records:
                source_path = Path(item["original_ms"])
                if not source_path.is_file():
                    missing.append({"parameter": parameter, "stack": stack, "historical_native_slice_id": key[2], "reason": "historical_original_ms_missing", "path": str(source_path)})
                    continue
                values, source_affine = _array_and_affine(source_path)
                candidates: list[tuple[int, dict[str, Any]]] = []
                for index, group in enumerate(groups):
                    target_shape = tuple(int(value) for value in np.asarray(current[stack][field][index]).shape)
                    try:
                        relation = exact_grid_relation(source_affine, tuple(values.shape), _target_affine(current[stack], index), target_shape)
                    except ValueError:
                        continue
                    candidates.append((group, relation))
                if not candidates:
                    missing.append({"parameter": parameter, "stack": stack, "historical_native_slice_id": key[2], "reason": "no_exact_current_r0_group_match", "historical_original_ms": _record(source_path)})
                elif len(candidates) > 1:
                    ambiguous.append({"parameter": parameter, "stack": stack, "historical_native_slice_id": key[2], "reason": "multiple_exact_current_r0_group_matches", "candidate_original_group_idx": [group for group, _ in candidates], "historical_original_ms": _record(source_path)})
                else:
                    group, relation = candidates[0]
                    found.append({"parameter": parameter, "stack": stack, "historical_native_slice_id": key[2], "original_group_idx": group, "historical_original_ms": _record(source_path), "current_r0_provenance": current[stack]["provenance"], **_relation_record(relation), "no_interpolation": True})
                    coverage[parameter][stack]["matched"] += 1

    for parameter in PARAMETERS:
        for stack in STACKS:
            records = [item for item in found if item["parameter"] == parameter and item["stack"] == stack]
            grouped: dict[int, list[int]] = {}
            for item in records:
                grouped.setdefault(int(item["original_group_idx"]), []).append(int(item["historical_native_slice_id"]))
            for group, historical_ids in grouped.items():
                if len(historical_ids) > 1:
                    duplicate_current.append({"parameter": parameter, "stack": stack, "original_group_idx": group, "historical_native_slice_ids": historical_ids, "reason": "multiple_historical_records_match_one_current_group"})

    consistency_failures: list[dict[str, Any]] = []
    for stack in STACKS:
        t1 = {int(item["historical_native_slice_id"]): int(item["original_group_idx"]) for item in found if item["parameter"] == "t1" and item["stack"] == stack}
        t2 = {int(item["historical_native_slice_id"]): int(item["original_group_idx"]) for item in found if item["parameter"] == "t2" and item["stack"] == stack}
        if t1 != t2:
            consistency_failures.append({"stack": stack, "t1_historical_to_current": t1, "t2_historical_to_current": t2})

    completeness_failures = []
    for parameter in PARAMETERS:
        for stack, details in coverage[parameter].items():
            if details != {"expected": EXPECTED_CYJ_L100_GROUPS[stack], "historical_records": EXPECTED_CYJ_L100_GROUPS[stack], "current_groups": EXPECTED_CYJ_L100_GROUPS[stack], "matched": EXPECTED_CYJ_L100_GROUPS[stack]}:
                completeness_failures.append({"parameter": parameter, "stack": stack, **details})
    complete = not (missing or ambiguous or duplicate_current or consistency_failures or completeness_failures)
    return {"schema": "scmr_historical_group_mapping_discovery/v1", "found": found, "missing": missing, "ambiguous": ambiguous, "duplicate_current": duplicate_current, "coverage": coverage, "completeness_failures": completeness_failures, "t1_t2_consistency": {"passed": not consistency_failures, "failures": consistency_failures}, "complete": complete}


def build_scmr_historical_group_mapping(*, historical_common_mask_manifest: str | Path, r0_native_maps_root: str | Path, sparse_manifest: str | Path, output: str | Path) -> Path:
    """Create one immutable, complete group-map manifest after exact matching."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    sparse = json.loads(Path(sparse_manifest).read_text(encoding="utf-8"))
    if sparse.get("subject") != "CYJ" or sparse.get("condition") != "L100":
        raise ValueError("formal historical group mapping is CYJ/L100 only")
    discovery = discover_scmr_historical_group_mapping(historical_common_mask_manifest=historical_common_mask_manifest, r0_native_maps_root=r0_native_maps_root, sparse_manifest=sparse_manifest)
    if not discovery["complete"]:
        raise ValueError(f"SCMR_HISTORICAL_GROUP_MAPPING_INCOMPLETE: missing={len(discovery['missing'])}, ambiguous={len(discovery['ambiguous'])}, duplicate_current={len(discovery['duplicate_current'])}, consistency={discovery['t1_t2_consistency']['passed']}, completeness_failures={len(discovery['completeness_failures'])}")
    if len(discovery["found"]) != 82:
        raise ValueError(f"expected exactly 82 CYJ/L100 mappings, got {len(discovery['found'])}")
    current = {stack: _load_all_verified_native_cache_groups(r0_native_maps_root, sparse_manifest, stack) for stack in STACKS}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        write_immutable_json(staging / "manifest.json", {"schema": "scmr_historical_group_mapping/v1", "subject": "CYJ", "condition": "L100_fold0", "historical_common_mask_manifest": _record(historical_common_mask_manifest), "r0_native_root": {"path": str(Path(r0_native_maps_root).resolve()), "provenance": {stack: current[stack]["provenance"] for stack in STACKS}}, "sparse_manifest": _record(sparse_manifest), "mappings": discovery["found"], "coverage": discovery["coverage"], "completeness": {"expected_total_mappings": 82, "actual_total_mappings": len(discovery["found"]), "passed": True}, "t1_t2_consistency": discovery["t1_t2_consistency"], "no_interpolation": True, "no_resampling": True, "no_training": True, "no_reconstruction": True})
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-common-mask-manifest", required=True)
    parser.add_argument("--r0-native-maps-root", required=True)
    parser.add_argument("--sparse-manifest", required=True)
    parser.add_argument("--output")
    parser.add_argument("--mode", choices=("discover", "build"), default="build")
    args = vars(parser.parse_args())
    if args.pop("mode") == "discover":
        args.pop("output")
        print(json.dumps(discover_scmr_historical_group_mapping(**args), indent=2, sort_keys=True))
        return
    if not args.get("output"):
        raise ValueError("--output is required for build")
    print(build_scmr_historical_group_mapping(**args))


if __name__ == "__main__":
    main()
