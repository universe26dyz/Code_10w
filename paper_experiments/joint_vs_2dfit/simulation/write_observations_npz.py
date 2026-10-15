"""Write the prepared-observation contract with explicit provenance and QC."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from trad.modules.module_02_data_bridge.timing import compute_duration_before_acq


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


def validate_observations_file(path: str | Path) -> dict:
    """Independently revalidate an on-disk formal observation archive."""
    archive = np.load(Path(path), allow_pickle=False)
    if not set(SCHEMA_FIELDS) <= set(archive.files):
        raise ValueError("observation archive is missing required schema fields")
    arrays = {key: archive[key] for key in SCHEMA_FIELDS}
    images, masks = arrays["images"], arrays["masks"]
    if images.ndim != 3 or masks.shape != images.shape or not np.isfinite(images).all() or not np.isfinite(arrays["affine_lps_rc"]).all() or not np.isfinite(arrays["timing9_ms"]).all() or not np.isfinite(arrays["acquisition_time_ms"]).all():
        raise ValueError("invalid observation image, mask, affine, or timing values")
    groups = sorted(set(np.asarray(arrays["group_idx"], int).tolist()))
    if groups != list(range(len(groups))):
        raise ValueError("group_idx must be contiguous starting at zero")
    for group in groups:
        idx = np.flatnonzero(np.asarray(arrays["group_idx"], int) == group)
        if idx.size != 10 or np.asarray(arrays["weight_idx"], int)[idx].tolist() != list(range(10)):
            raise ValueError("each group must contain weights 0..9 exactly once")
        for field in ("timing9_ms", "affine_lps_rc", "pixel_spacing_rc_mm", "slice_thickness_mm", "tr_ms", "vps"):
            if not np.array_equal(np.asarray(arrays[field])[idx], np.repeat(np.asarray(arrays[field])[idx[:1]], len(idx), axis=0)):
                raise ValueError(f"group {group} does not share {field}")
        if not np.all(masks[idx].reshape(len(idx), -1).any(axis=1)):
            raise ValueError("every observation weight must have a non-empty foreground mask")
        if not (np.all(arrays["tr_ms"][idx] == 2.61) and np.all(arrays["vps"][idx] == 87)):
            raise ValueError("formal observations require TR=2.61 and VPS=87")
        timing = np.asarray(arrays["timing9_ms"][idx[0]], float)
        recovered = compute_duration_before_acq(np.asarray(arrays["acquisition_time_ms"][idx], float), 2.61, 87, np.array([50.0, 150.0]), np.array([35.0, 45.0, 55.0]))[1:]
        if timing.shape != (9,) or not np.allclose(recovered, timing, rtol=0.0, atol=1e-9):
            raise ValueError("acquisition_time_ms does not round-trip to timing9_ms")
    return {"group_count": len(groups), "observation_count": int(images.shape[0]), "semantic_gate": True}


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
