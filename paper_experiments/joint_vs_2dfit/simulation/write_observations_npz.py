"""Write the prepared-observation contract with explicit provenance and QC."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA_FIELDS = (
    "images", "masks", "group_idx", "weight_idx", "stack_idx",
    "acquisition_time_ms", "timing9_ms", "affine_lps_rc",
    "pixel_spacing_rc_mm", "slice_thickness_mm", "tr_ms", "vps",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_observations(output_dir: str | Path, images: np.ndarray, masks: np.ndarray, metadata: dict, provenance: dict) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / name for name in ("observations.npz", "manifest.json", "qc_summary.json", "synthetic_source_manifest.json")]
    existing = [p for p in paths if p.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite {existing[0]}")
    images = np.asarray(images, dtype=np.float32)
    masks = np.asarray(masks, dtype=bool)
    if images.ndim != 3 or masks.shape != images.shape:
        raise ValueError("images and masks must be (N, rows, cols) with equal shape")
    n = images.shape[0]
    missing = [field for field in SCHEMA_FIELDS[2:] if field not in metadata]
    if missing:
        raise ValueError(f"missing observation metadata: {missing}")
    arrays = {"images": images, "masks": masks}
    for field in SCHEMA_FIELDS[2:]:
        value = np.asarray(metadata[field])
        if value.shape[0] != n:
            raise ValueError(f"metadata {field} has {value.shape[0]} rows, expected {n}")
        arrays[field] = value
    group_idx = arrays["group_idx"].astype(int)
    weight_idx = arrays["weight_idx"].astype(int)
    groups = sorted(set(group_idx.tolist()))
    if groups != list(range(len(groups))):
        raise ValueError("group_idx must be contiguous starting at zero")
    for group in groups:
        idx = np.flatnonzero(group_idx == group)
        if idx.size != 10 or weight_idx[idx].tolist() != list(range(10)):
            raise ValueError(f"group {group} must contain weights 0..9 exactly once")
        for field in ("timing9_ms", "affine_lps_rc", "pixel_spacing_rc_mm", "slice_thickness_mm", "tr_ms", "vps"):
            if not np.all(np.asarray(arrays[field])[idx] == np.asarray(arrays[field])[idx[0]]):
                raise ValueError(f"group {group} does not share {field}")
    if not np.all(np.isfinite(images)) or not np.all(np.isfinite(arrays["timing9_ms"])):
        raise ValueError("observations and timing must be finite")
    if not np.all(masks.reshape(n, -1).any(axis=1)):
        raise ValueError("every weight must have a non-empty foreground mask")
    np.savez_compressed(paths[0], **arrays)
    manifest = {
        "schema_fields": list(SCHEMA_FIELDS),
        "n_observations": int(n),
        "group_count": len(groups),
        "weights_per_group": 10,
        "shape": list(images.shape),
        "provenance": provenance,
    }
    paths[1].write_text(json.dumps(manifest, indent=2, default=_json_safe) + "\n")
    qc = {
        "finite_images": bool(np.all(np.isfinite(images))),
        "finite_masks": bool(np.all(np.isfinite(masks))),
        "nonempty_mask_fraction": float(np.mean(masks)),
        "group_ids_contiguous": groups == list(range(len(groups))),
        "weights_per_group": sorted({int(np.sum(group_idx == g)) for g in groups}),
        "image_sha256": _sha256(paths[0]),
    }
    paths[2].write_text(json.dumps(qc, indent=2) + "\n")
    paths[3].write_text(json.dumps(provenance, indent=2, default=_json_safe) + "\n")
    return {"schema_fields": list(SCHEMA_FIELDS), "paths": [str(p) for p in paths], "qc": qc}
