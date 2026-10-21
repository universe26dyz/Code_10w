"""Build the formal current-native Phase4R central-component mask.

The only inputs are the current R0 source support and the frozen L100 support
already sampled on that same native lattice.  Historical reconstructions never
enter this artifact.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from .contracts import sha256, write_immutable_json

STACKS = ("sax", "2ch", "4ch")
PARAMETERS = ("t1", "t2")
EXPECTED_GROUPS = {"sax": 14, "2ch": 15, "4ch": 12}
REQUIRED_ARRAYS = ("source_support", "phase4r_l100", "base_native_support", "primary_native_mask", "reference")


def _load_all_verified_native_cache_groups(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Late import avoids a reporting-to-mask import cycle."""
    from .final_cyj_reporting import _load_all_verified_native_cache_groups as loader
    return loader(*args, **kwargs)


def _record(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _l100_records(root: str | Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    root = Path(root); manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8")); schema = payload.get("schema")
    if schema not in {"q004_direct_native_endpoint_audit/v2", "phase4r_l100_native_support/v1"}:
        raise ValueError("frozen L100 native support must be q004_direct_native_endpoint_audit/v2 or phase4r_l100_native_support/v1")
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    for item in payload.get("planes", []):
        group_key = "group_idx" if schema == "q004_direct_native_endpoint_audit/v2" else "original_group_idx"
        key = (str(item["parameter"]).lower(), str(item["stack"]), int(item[group_key]))
        if key in records:
            raise ValueError(f"duplicate frozen L100 support identity: {key}")
        relative = item["output"] if schema == "q004_direct_native_endpoint_audit/v2" else item["path"]
        path = Path(relative)
        records[key] = {"path": str((path if path.is_absolute() else root / path).resolve()), "field": "l100_native" if schema == "q004_direct_native_endpoint_audit/v2" else "phase4r_l100", "declared_sha256": item.get("sha256"), "source_schema": schema}
    return records


def select_current_native_central_component(support: np.ndarray, *, minimum_ratio: float = .25) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the unchanged SCMR-style 8-connected central/largest rule."""
    support = np.asarray(support, bool)
    if support.ndim != 2:
        raise ValueError("current-native central-component selection requires one 2-D plane")
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("minimum_ratio must be in [0, 1]")
    labels, count = ndimage.label(support, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(support), {"component_count": 0, "selected_label": None, "selection": "empty_support", "center_rc": [int(round((size - 1) / 2.0)) for size in support.shape], "largest_area": 0, "selected_area": 0, "minimum_ratio": float(minimum_ratio), "connectivity": 8}
    areas = np.bincount(labels.ravel())[1:]; largest_label = int(np.argmax(areas) + 1)
    center_rc = tuple(int(round((size - 1) / 2.0)) for size in support.shape); center_label = int(labels[center_rc])
    if center_label and areas[center_label - 1] >= minimum_ratio * areas[largest_label - 1]:
        selected_label, selection = center_label, "contains_image_center"
    else:
        selected_label, selection = largest_label, "largest_component"
    return labels == selected_label, {"component_count": int(count), "selected_label": int(selected_label), "selection": selection, "center_rc": list(center_rc), "largest_area": int(areas[largest_label - 1]), "selected_area": int(areas[selected_label - 1]), "minimum_ratio": float(minimum_ratio), "connectivity": 8}


def build_current_native_masks(source_support: np.ndarray, phase4r_l100: np.ndarray, *, minimum_ratio: float = .25) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source_support, phase4r_l100 = np.asarray(source_support, bool), np.asarray(phase4r_l100, bool)
    if source_support.shape != phase4r_l100.shape:
        raise ValueError("source_support and frozen L100 support must share one current native lattice")
    base = source_support & phase4r_l100
    primary, selection = select_current_native_central_component(base, minimum_ratio=minimum_ratio)
    return {"source_support": source_support, "phase4r_l100": phase4r_l100, "base_native_support": base, "primary_native_mask": primary}, selection


def load_validated_current_native_mask_plane(mask_root: str | Path, *, parameter: str, stack: str, original_group_idx: int, shape_rc: tuple[int, int], source_support: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = Path(mask_root); manifest_path = root / "manifest.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "phase4r_current_native_central_component_mask/v1":
        raise ValueError("formal native support requires phase4r_current_native_central_component_mask/v1")
    records = [item for item in manifest.get("planes", []) if item.get("parameter") == parameter and item.get("stack") == stack and int(item.get("original_group_idx", -1)) == int(original_group_idx)]
    if len(records) != 1:
        raise ValueError(f"current-native mask manifest lacks exactly one {parameter}/{stack}/group{original_group_idx:03d}")
    record = records[0]; target = root / record["output"]["path"]
    expected = Path("planes") / parameter / stack / f"group{original_group_idx:03d}.npz"
    if target.relative_to(root) != expected or not target.is_file() or sha256(target) != record["output"].get("sha256"):
        raise ValueError(f"current-native mask artifact provenance mismatch for {parameter}/{stack}/group{original_group_idx:03d}")
    with np.load(target, allow_pickle=False) as payload:
        missing = [name for name in REQUIRED_ARRAYS if name not in payload.files]
        if missing:
            raise ValueError(f"{target}: missing current-native mask arrays {missing}")
        values = {name: np.asarray(payload[name]) for name in REQUIRED_ARRAYS}
    if any(value.shape != tuple(shape_rc) for value in values.values()):
        raise ValueError(f"{target}: current-native mask shape does not match reference")
    artifact_source = np.asarray(values["source_support"], bool); l100 = np.asarray(values["phase4r_l100"], bool)
    if not np.array_equal(artifact_source, np.asarray(source_support, bool)):
        raise ValueError(f"{target}: source_support does not match the selected R0 plane")
    if not np.array_equal(np.asarray(values["base_native_support"], bool), artifact_source & l100):
        raise ValueError(f"{target}: base_native_support is not source_support AND phase4r_l100")
    loaded = {name: np.asarray(value, bool) for name, value in values.items() if name != "reference"}
    loaded["reference"] = np.asarray(values["reference"], float)
    return loaded, {"mask_manifest": _record(manifest_path), "mask_plane": _record(target), "primary_mask": "primary_native_mask", "selection_rule": manifest.get("selection_rule")}


def build_current_native_central_component_mask(*, r0_native_maps_root: str | Path, sparse_manifest: str | Path, l100_native_support_root: str | Path, output: str | Path, minimum_ratio: float = .25) -> Path:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    sparse = json.loads(Path(sparse_manifest).read_text(encoding="utf-8"))
    if sparse.get("subject") != "CYJ" or sparse.get("condition") != "L100":
        raise ValueError("formal current-native mask is CYJ/L100 only")
    l100 = _l100_records(l100_native_support_root)
    current = {stack: _load_all_verified_native_cache_groups(r0_native_maps_root, sparse_manifest, stack) for stack in STACKS}
    if {stack: len(current[stack]["group_idx"]) for stack in STACKS} != EXPECTED_GROUPS:
        raise ValueError("CYJ L100 group completeness is not 14/15/12")
    output.parent.mkdir(parents=True, exist_ok=True); staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        planes: list[dict[str, Any]] = []
        for parameter in PARAMETERS:
            for stack in STACKS:
                for index, group_value in enumerate(current[stack]["group_idx"]):
                    group = int(group_value); key = (parameter, stack, group)
                    if key not in l100:
                        raise ValueError(f"missing frozen L100 native support: {key}")
                    source = np.asarray(current[stack]["source_support"][index], bool); reference = np.asarray(current[stack][f"{parameter}_ms"][index], np.float32)
                    support_record = l100[key]; support_path = Path(support_record["path"])
                    if not support_path.is_file() or (support_record["declared_sha256"] is not None and sha256(support_path) != support_record["declared_sha256"]):
                        raise ValueError(f"frozen L100 provenance mismatch: {support_path}")
                    with np.load(support_path, allow_pickle=False) as archive:
                        if support_record["field"] not in archive.files:
                            raise ValueError(f"{support_path}: missing frozen L100 field {support_record['field']}")
                        frozen = np.asarray(archive[support_record["field"]], bool)
                    if frozen.shape != source.shape:
                        raise ValueError(f"{support_path}: frozen L100 support is not on the current native lattice")
                    values, selection = build_current_native_masks(source, frozen, minimum_ratio=minimum_ratio)
                    target = staging / "planes" / parameter / stack / f"group{group:03d}.npz"; target.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(target, **values, reference=reference)
                    planes.append({"parameter": parameter, "stack": stack, "original_group_idx": group, "shape_rc": list(source.shape), "current_r0_source": current[stack]["provenance"], "l100_source": {"path": str(support_path.resolve()), "sha256": sha256(support_path), "field": support_record["field"], "source_schema": support_record["source_schema"]}, "source_support_voxels": int(source.sum()), "phase4r_l100_voxels": int(frozen.sum()), "base_native_support_voxels": int(values["base_native_support"].sum()), "primary_native_mask_voxels": int(values["primary_native_mask"].sum()), "component_selection": selection, "output": {"path": str(target.relative_to(staging)), "sha256": sha256(target)}, "no_interpolation": True})
        if len(planes) != 82:
            raise ValueError(f"expected 82 current-native mask planes, got {len(planes)}")
        write_immutable_json(staging / "manifest.json", {"schema": "phase4r_current_native_central_component_mask/v1", "subject": "CYJ", "condition": "L100_fold0", "r0_native_root": {"path": str(Path(r0_native_maps_root).resolve()), "provenance": {stack: current[stack]["provenance"] for stack in STACKS}}, "sparse_manifest": _record(sparse_manifest), "frozen_l100_native_support_manifest": _record(Path(l100_native_support_root) / "manifest.json"), "planes": planes, "primary_mask": "primary_native_mask", "base_support_equation": "source_support AND phase4r_l100", "selection_rule": "SCMR-style central/largest 8-connected component", "minimum_ratio": float(minimum_ratio), "no_interpolation": True, "no_resampling": True, "no_training": True, "no_reconstruction": True})
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True); raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r0-native-maps-root", required=True); parser.add_argument("--sparse-manifest", required=True); parser.add_argument("--l100-native-support-root", required=True); parser.add_argument("--output", required=True); parser.add_argument("--minimum-ratio", type=float, default=.25)
    print(build_current_native_central_component_mask(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
