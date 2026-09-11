"""CLI to produce one self-contained observation NPZ/manifest from a v7.3 MAT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .build_manifest import build_manifest
from .mat_v73 import load_preprocessed_v73


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _group_mask(images: np.ndarray) -> np.ndarray:
    mean_image = images.mean(axis=0)
    mask = np.isfinite(mean_image) & (mean_image > float(mean_image.mean()) / 5.0)
    if not np.any(mask):
        raise ValueError("HHZ simple foreground mask is empty for a prepared group.")
    return mask


def prepare_observations(
    preprocessed_mat: str | Path,
    dicom_dir: str | Path,
    stack: str,
    output_dir: str | Path,
    max_groups: int | None = None,
) -> dict[str, Any]:
    """Write prepared observations without copying raw DICOMs into the code tree."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory must be absent or empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True)
    preprocessed = load_preprocessed_v73(preprocessed_mat)
    manifest = build_manifest(preprocessed, dicom_dir, stack, max_groups)
    observations = manifest["observations"]
    n_groups = len(manifest["groups"])
    if len(observations) != n_groups * 10:
        raise ValueError("Bridge manifest violates the required 10 observations per group.")

    image_blocks: list[np.ndarray] = []
    mask_blocks: list[np.ndarray] = []
    for group_idx in range(n_groups):
        group_images = np.moveaxis(preprocessed["mag_crop"][:, :, :, group_idx], 2, 0)
        if group_images.shape[0] != 10:
            raise ValueError(f"MAT group {group_idx} lacks 10 weights.")
        group_mask = _group_mask(group_images)
        image_blocks.append(group_images)
        mask_blocks.append(np.repeat(group_mask[None, :, :], 10, axis=0))
    images = np.concatenate(image_blocks, axis=0).astype(np.float32, copy=False)
    masks = np.concatenate(mask_blocks, axis=0).astype(bool, copy=False)
    np.savez_compressed(
        output_dir / "observations.npz",
        images=images,
        masks=masks,
        group_idx=np.asarray([item["group_idx"] for item in observations], dtype=np.int64),
        weight_idx=np.asarray([item["weight_idx"] for item in observations], dtype=np.int64),
        stack_idx=np.asarray([item["stack_idx"] for item in observations], dtype=np.int64),
        acquisition_time_ms=np.asarray([item["acquisition_time_ms"] for item in observations], dtype=np.float64),
        timing9_ms=np.stack([item["timing9_ms"] for item in observations]).astype(np.float64),
        affine_lps_rc=np.stack([item["affine_lps_rc"] for item in observations]).astype(np.float64),
        pixel_spacing_rc_mm=np.stack([item["pixel_spacing_rc_mm"] for item in observations]).astype(np.float64),
        slice_thickness_mm=np.asarray([item["slice_thickness_mm"] for item in observations], dtype=np.float64),
        tr_ms=np.asarray([item["tr_ms"] for item in observations], dtype=np.float64),
        vps=np.asarray([item["vps"] for item in observations], dtype=np.int64),
        sop_instance_uid=np.asarray([item["sop_instance_uid"] for item in observations], dtype=np.str_),
    )
    np.save(output_dir / "timing.npy", np.stack([group["timing9_ms"] for group in manifest["groups"]]))
    manifest_json = {
        "coordinate_system": manifest["coordinate_system"],
        "stack_name": manifest["stack_name"],
        "stack_idx": manifest["stack_idx"],
        "groups": manifest["groups"],
        "observations": [
            {key: value for key, value in item.items() if key not in {"timing9_ms", "affine_lps_rc", "pixel_spacing_rc_mm"}}
            for item in observations
        ],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest_json, handle, ensure_ascii=False, indent=2, default=_json_value)
    qc_summary = {
        "group_count": n_groups,
        "observation_count": len(observations),
        "weights_per_group": [group["weight_count"] for group in manifest["groups"]],
        "image_shape": list(images.shape),
        "timing_shape": list(np.load(output_dir / "timing.npy").shape),
        "geometry_shape": list(np.stack([group["affine_lps_rc"] for group in manifest["groups"]]).shape),
        "tr_ms": float(preprocessed["tr_ms"]),
        "vps": int(preprocessed["vps"]),
        "coordinate_system": manifest["coordinate_system"],
    }
    with (output_dir / "qc_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(qc_summary, handle, ensure_ascii=False, indent=2)
    return {"manifest": manifest, "qc_summary": qc_summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessed-mat", required=True)
    parser.add_argument("--dicom-dir", required=True)
    parser.add_argument("--stack", required=True, choices=("sax", "2ch", "4ch"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-groups", type=int)
    args = parser.parse_args()
    result = prepare_observations(**vars(args))
    subject = Path(args.dicom_dir).resolve().parent.name
    qc = result["qc_summary"]
    print(f"subject={subject}")
    print(f"stack={args.stack}")
    print(f"group_count={qc['group_count']}")
    print(f"weights_per_group={qc['weights_per_group']}")
    print(f"timing_shape={qc['timing_shape']}")
    print(f"geometry_shape={qc['geometry_shape']}")


if __name__ == "__main__":
    main()
