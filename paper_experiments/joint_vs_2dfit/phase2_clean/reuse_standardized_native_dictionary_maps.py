"""Fail-closed, immutable native dictionary-map reuse after slice standardization."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import sha256, write_immutable_json
from .sparse import STACKS, _git_sha


SCHEMA = "phase4r_standardized_native_dictionary_reuse/v1"


def _group_rows(data: np.lib.npyio.NpzFile, group: int, path: Path) -> np.ndarray:
    rows = np.flatnonzero(np.asarray(data["group_idx"], int) == group)
    if len(rows) != 10 or not np.array_equal(np.sort(np.asarray(data["weight_idx"])[rows]), np.arange(10)):
        raise ValueError(f"{path}: original group {group} is not a complete ten-weight group")
    return rows


def _fingerprint(data: np.lib.npyio.NpzFile, rows: np.ndarray) -> str:
    import hashlib
    digest = hashlib.sha256()
    for key in sorted(data.files):
        if key == "group_idx":
            continue
        value = np.asarray(data[key])
        digest.update(key.encode("utf-8")); digest.update(np.ascontiguousarray(value[rows] if value.shape[:1] == data["group_idx"].shape else value).tobytes())
    return digest.hexdigest()


def _native_inputs(root: Path, stack: str) -> tuple[Path, Path, Path, dict[str, Any]]:
    folder = root / stack
    maps, geometry, manifest = folder / "native_maps.npz", folder / "geometry.npz", folder / "manifest.json"
    if not all(path.is_file() for path in (maps, geometry, manifest)):
        raise FileNotFoundError(f"{stack}: native maps, geometry, and manifest are all required")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("native_maps_sha256") != sha256(maps) or payload.get("geometry_sha256") != sha256(geometry):
        raise ValueError(f"{stack}: native artifact SHA256 mismatch")
    return maps, geometry, manifest, payload


def audit_standardized_native_dictionary_maps(standardization_manifest: str | Path, full_native_maps_root: str | Path, output: str | Path) -> Path:
    source_manifest, native_root, target = Path(standardization_manifest), Path(full_native_maps_root), Path(output)
    if target.exists():
        raise FileExistsError(f"refusing overwrite: {target}")
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    standardization = json.loads(source_manifest.read_text(encoding="utf-8"))
    if standardization.get("schema") != "phase4r_common_slice_standardization/v1" or not isinstance(standardization.get("subject"), str):
        raise ValueError("standardization manifest schema/subject is invalid")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        records: dict[str, Any] = {}
        for stack in STACKS:
            record = standardization.get("stacks", {}).get(stack)
            if not isinstance(record, dict):
                raise ValueError(f"{stack}: standardization record is missing")
            original_path = Path(str(record.get("source_observation", {}).get("path", "")))
            standardized_path = Path(str(record.get("output_observation", {}).get("path", "")))
            if not original_path.is_file() or not standardized_path.is_file():
                raise FileNotFoundError(f"{stack}: standardization observation evidence is unavailable")
            if sha256(original_path) != record["source_observation"].get("sha256") or sha256(standardized_path) != record["output_observation"].get("sha256"):
                raise ValueError(f"{stack}: standardization observation SHA256 mismatch")
            mapping = {int(key): int(value) for key, value in record.get("derived_to_original_group_idx", {}).items()}
            if sorted(mapping) != list(range(int(record.get("output_count", -1)))) or list(mapping.values()) != [int(value) for value in record.get("retained_original_group_ids", [])]:
                raise ValueError(f"{stack}: standardized-to-original mapping is invalid")
            maps_path, geometry_path, native_manifest_path, native_manifest = _native_inputs(native_root, stack)
            source_observation = native_manifest.get("source_observations", {})
            if source_observation.get("sha256") != sha256(original_path):
                raise ValueError(f"{stack}: native map source observation does not match standardization source")
            with np.load(original_path, allow_pickle=False) as original, np.load(standardized_path, allow_pickle=False) as standardized, np.load(maps_path, allow_pickle=False) as maps, np.load(geometry_path, allow_pickle=False) as geometry:
                native_groups = [int(value) for value in maps["group_idx"]]
                if len(native_groups) != len(set(native_groups)):
                    raise ValueError(f"{stack}: duplicate native group identity")
                if any(np.asarray(geometry[key]).shape[0] != len(native_groups) for key in ("affine_lps_rc", "pixel_spacing_rc_mm", "slice_thickness_mm")):
                    raise ValueError(f"{stack}: native geometry length mismatch")
                positions: list[int] = []; checks: list[dict[str, Any]] = []
                for derived, original_group in mapping.items():
                    if native_groups.count(original_group) != 1:
                        raise ValueError(f"{stack}: missing native group {original_group}")
                    original_rows, derived_rows = _group_rows(original, original_group, original_path), _group_rows(standardized, derived, standardized_path)
                    content_match = _fingerprint(original, original_rows) == _fingerprint(standardized, derived_rows)
                    position = native_groups.index(original_group)
                    geometry_match = np.array_equal(np.asarray(original["affine_lps_rc"])[original_rows[0]], np.asarray(standardized["affine_lps_rc"])[derived_rows[0]]) and np.array_equal(np.asarray(geometry["affine_lps_rc"])[position], np.asarray(original["affine_lps_rc"])[original_rows[0]])
                    if not content_match or not geometry_match:
                        raise ValueError(f"{stack}: content or geometry mismatch for original group {original_group}")
                    positions.append(position); checks.append({"standardized_group_idx": derived, "original_group_idx": original_group, "content_match": True, "geometry_match": True})
                folder = staging / stack; folder.mkdir()
                np.savez_compressed(folder / "native_maps.npz", t1_ms=np.asarray(maps["t1_ms"])[positions], t2_ms=np.asarray(maps["t2_ms"])[positions], b1=np.asarray(maps["b1"])[positions], group_idx=np.arange(len(positions), dtype=np.int64))
                np.savez_compressed(folder / "geometry.npz", affine_lps_rc=np.asarray(geometry["affine_lps_rc"])[positions], pixel_spacing_rc_mm=np.asarray(geometry["pixel_spacing_rc_mm"])[positions], slice_thickness_mm=np.asarray(geometry["slice_thickness_mm"])[positions])
            records[stack] = {"status": "REUSE_VERIFIED", "source_native_maps": {"path": str(maps_path.resolve()), "sha256": sha256(maps_path)}, "source_geometry": {"path": str(geometry_path.resolve()), "sha256": sha256(geometry_path)}, "source_native_manifest": {"path": str(native_manifest_path.resolve()), "sha256": sha256(native_manifest_path)}, "source_count": len(native_groups), "target_count": len(mapping), "standardized_to_original_group_idx": {str(key): value for key, value in mapping.items()}, "per_group_checks": checks, "output_native_maps_sha256": sha256(folder / "native_maps.npz"), "output_geometry_sha256": sha256(folder / "geometry.npz")}
        write_immutable_json(staging / "reuse_manifest.json", {"schema": SCHEMA, "subject": standardization["subject"], "status": "REUSE_VERIFIED", "standardization_manifest": {"path": str(source_manifest.resolve()), "sha256": sha256(source_manifest)}, "full_native_maps_root": str(native_root.resolve()), "stacks": records, "code_sha": _git_sha(), "immutable_output": True})
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardization-manifest", required=True); parser.add_argument("--full-native-maps-root", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); print(audit_standardized_native_dictionary_maps(args.standardization_manifest, args.full_native_maps_root, args.output))


if __name__ == "__main__":
    main()
