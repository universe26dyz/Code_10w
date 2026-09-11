"""Build a protocol-validated unique timing pool from prepared observation NPZs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_timing_pool(observation_paths: list[str | Path], output_path: str | Path) -> dict[str, object]:
    if not observation_paths:
        raise ValueError("At least one prepared observations.npz path is required.")
    rows: list[np.ndarray] = []
    source_ids: list[str] = []
    source_groups: list[int] = []
    stack_ids: list[int] = []
    protocol_records: list[tuple[str, int, float, int]] = []
    for source_index, raw_path in enumerate(observation_paths):
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Prepared observations NPZ does not exist: {path}")
        with np.load(path, allow_pickle=False) as data:
            required = {"group_idx", "weight_idx", "stack_idx", "timing9_ms", "tr_ms", "vps"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"{path} lacks required timing/protocol fields: {sorted(missing)}")
            group_idx, weight_idx, stack_idx = (np.asarray(data[key], dtype=np.int64) for key in ("group_idx", "weight_idx", "stack_idx"))
            timing = np.asarray(data["timing9_ms"], dtype=np.float64)
            tr_ms, vps = np.asarray(data["tr_ms"], dtype=np.float64), np.asarray(data["vps"], dtype=np.int64)
        for group in np.unique(group_idx):
            index = np.flatnonzero(group_idx == group)
            if index.size != 10 or not np.array_equal(np.sort(weight_idx[index]), np.arange(10)):
                raise ValueError(f"{path} group {group} is not a complete 10-weight group.")
            if not np.allclose(timing[index], timing[index[0]], rtol=0, atol=1e-10):
                raise ValueError(f"{path} group {group} has inconsistent timing9.")
            if not np.allclose(tr_ms[index], tr_ms[index[0]], rtol=1e-6, atol=1e-6) or not np.all(vps[index] == vps[index[0]]):
                raise ValueError(f"{path} group {group} has inconsistent TR/VPS.")
            if np.unique(stack_idx[index]).size != 1:
                raise ValueError(f"{path} group {group} has inconsistent stack_idx.")
            rows.append(timing[index[0]])
            source_ids.append(str(path))
            source_groups.append(int(group))
            stack_ids.append(int(stack_idx[index[0]]))
            protocol_records.append((str(path), int(group), float(tr_ms[index[0]]), int(vps[index[0]])))
    first_tr, first_vps = protocol_records[0][2:]
    if any(not np.isclose(record[2], first_tr, rtol=1e-6, atol=1e-6) or record[3] != first_vps for record in protocol_records):
        detail = "; ".join(f"{path}:group={group},TR={tr:.9g},VPS={vps}" for path, group, tr, vps in protocol_records)
        raise ValueError("One 12D MLP requires identical TR/VPS across timing sources; " + detail)
    timing_rows = np.stack(rows)
    _, first_indices = np.unique(timing_rows, axis=0, return_index=True)
    first_indices = np.sort(first_indices)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Timing pool output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, timing9_ms=timing_rows[first_indices], tr_ms=np.asarray(first_tr), vps=np.asarray(first_vps, dtype=np.int64), source_id=np.asarray([source_ids[index] for index in first_indices]), group_id=np.asarray([source_groups[index] for index in first_indices], dtype=np.int64), stack_idx=np.asarray([stack_ids[index] for index in first_indices], dtype=np.int64), functional_fixture=np.asarray(False))
    return {"timing_count": int(first_indices.size), "tr_ms": first_tr, "vps": first_vps, "output": str(output)}


def build_timing_pool_from_manifest(manifest_path: str | Path, output_path: str | Path) -> dict[str, object]:
    """Formal pool: retain every group's explicit subject/stack provenance."""
    manifest_path = Path(manifest_path)
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    sources = manifest.get("sources") if isinstance(manifest, dict) else None
    if not isinstance(sources, list) or not sources:
        raise ValueError("Formal timing manifest must contain a non-empty sources list.")
    rows, subjects, stacks, group_ids, stack_ids, source_ids, records = [], [], [], [], [], [], []
    for entry in sources:
        if not isinstance(entry, dict) or any(key not in entry for key in ("subject_id", "stack", "observations")):
            raise ValueError("Each formal timing source requires explicit subject_id, stack, observations.")
        subject, stack, path = entry["subject_id"], entry["stack"], Path(entry["observations"])
        if not isinstance(subject, str) or not subject or not isinstance(stack, str) or not stack or not path.is_file():
            raise ValueError(f"Invalid explicit formal timing source: {entry}")
        with np.load(path, allow_pickle=False) as data:
            required = {"group_idx", "weight_idx", "stack_idx", "timing9_ms", "tr_ms", "vps"}
            if required.difference(data.files): raise ValueError(f"{path} lacks formal timing fields.")
            group, weight, sid = (np.asarray(data[k], dtype=np.int64) for k in ("group_idx", "weight_idx", "stack_idx"))
            timing, tr, vps = np.asarray(data["timing9_ms"], dtype=np.float64), np.asarray(data["tr_ms"], dtype=np.float64), np.asarray(data["vps"], dtype=np.int64)
        for value in np.unique(group):
            index = np.flatnonzero(group == value)
            if index.size != 10 or not np.array_equal(np.sort(weight[index]), np.arange(10)) or not np.allclose(timing[index], timing[index[0]], rtol=0, atol=1e-10) or not np.allclose(tr[index], tr[index[0]], rtol=1e-6, atol=1e-6) or not np.all(vps[index] == vps[index[0]]) or np.unique(sid[index]).size != 1:
                raise ValueError(f"{path} group {value} is not a protocol-consistent complete 10-weight group.")
            rows.append(timing[index[0]]); subjects.append(subject); stacks.append(stack); group_ids.append(int(value)); stack_ids.append(int(sid[index[0]])); source_ids.append(str(path)); records.append((subject, stack, int(value), float(tr[index[0]]), int(vps[index[0]])))
    first_tr, first_vps = records[0][3:]
    if any(not np.isclose(record[3], first_tr, rtol=1e-6, atol=1e-6) or record[4] != first_vps for record in records):
        raise ValueError("Formal 12D MLP protocol audit failed: TR must match within tolerance and VPS exactly across all groups.")
    output = Path(output_path)
    if output.exists(): raise FileExistsError(f"Timing pool output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, timing9_ms=np.stack(rows), tr_ms=np.asarray(first_tr), vps=np.asarray(first_vps, dtype=np.int64), subject_id=np.asarray(subjects), stack=np.asarray(stacks), stack_idx=np.asarray(stack_ids, dtype=np.int64), group_id=np.asarray(group_ids, dtype=np.int64), source_id=np.asarray(source_ids), functional_fixture=np.asarray(False))
    return {"timing_count": len(rows), "n_subjects": len(set(subjects)), "tr_ms": first_tr, "vps": first_vps, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", action="append")
    parser.add_argument("--manifest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if bool(args.manifest) == bool(args.observations): raise ValueError("Provide exactly one of --manifest or repeated --observations.")
    print(build_timing_pool_from_manifest(args.manifest, args.output) if args.manifest else build_timing_pool(args.observations, args.output))


if __name__ == "__main__":
    main()
