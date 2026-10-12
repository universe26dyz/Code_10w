"""Deterministic medoid-like selection of an observed VPS87 timing group."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def _in_domain(timing: np.ndarray, approved: dict) -> bool:
    lo = np.asarray(approved.get("train_timing9_min_ms", approved.get("timing9_min_ms")), dtype=float)
    hi = np.asarray(approved.get("train_timing9_max_ms", approved.get("timing9_max_ms")), dtype=float)
    return bool(np.all(timing >= lo) and np.all(timing <= hi))


def choose_timing_template(candidates: Iterable[dict], approved: dict) -> dict:
    eligible = []
    for item in candidates:
        timing = np.asarray(item["timing9_ms"], dtype=float)
        if abs(float(item["tr_ms"]) - float(approved["tr_ms"])) > 1e-6 or int(item["vps"]) != int(approved["vps"]):
            continue
        if timing.shape != (9,) or not np.all(np.isfinite(timing)) or not _in_domain(timing, approved):
            continue
        eligible.append((item, timing))
    if not eligible:
        raise ValueError("no observed VPS87 timing group lies in the approved decoder timing domain")
    matrix = np.stack([x[1] for x in eligible])
    median = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - median), axis=0)
    scale = np.where(mad > 1e-9, 1.4826 * mad, np.maximum(np.std(matrix, axis=0), 1.0))
    distances = np.linalg.norm((matrix - median) / scale, axis=1)
    # Stable tie break: subject, stack, group.  The selected vector is never
    # synthesized from the median; it is copied from this observed group.
    order = sorted(range(len(eligible)), key=lambda i: (round(float(distances[i]), 12), str(eligible[i][0].get("subject_id", "")), str(eligible[i][0].get("stack_name", "")), int(eligible[i][0].get("group_idx", 0))))
    source, timing = eligible[order[0]]
    return {
        "selection_rule": "coordinate median reference; MAD robust scaling; nearest actual observed group",
        "candidate_count": len(eligible),
        "coordinate_median_ms": median.tolist(),
        "robust_scale_ms": scale.tolist(),
        "selected_distance": float(distances[order[0]]),
        "timing9_ms": timing.tolist(),
        "tr_ms": float(source["tr_ms"]),
        "vps": int(source["vps"]),
        "source": {k: v for k, v in source.items() if k != "timing9_ms"},
        "decoder_domain_check": True,
        "approved_decoder_protocol": {"tr_ms": float(approved["tr_ms"]), "vps": int(approved["vps"])},
    }


def collect_manifest_candidates(prepared_root: str | Path, subjects: tuple[str, ...] = ("CYJ", "DYZ", "HHZ", "HJL")) -> list[dict]:
    root = Path(prepared_root)
    out: list[dict] = []
    for sid in subjects:
        for manifest in sorted((root / sid).glob("*/manifest.json")):
            data = json.loads(manifest.read_text())
            manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
            for group in data.get("groups", []):
                out.append({
                    "subject_id": sid,
                    "stack_name": data.get("stack_name", manifest.parent.name),
                    "group_idx": int(group["group_idx"]),
                    "timing9_ms": group["timing9_ms"],
                    "tr_ms": group["tr_ms"],
                    "vps": group["vps"],
                    "manifest_path": str(manifest),
                    "manifest_sha256": manifest_hash,
                })
    return out


def write_timing_yaml(selection: dict, output: str | Path) -> None:
    import yaml

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(selection, sort_keys=False))
