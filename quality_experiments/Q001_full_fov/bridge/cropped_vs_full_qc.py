"""Read-only geometry/provenance QC for paired baseline-cropped and Q001 full-FOV inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from trad.modules.module_02_data_bridge.geometry import apply_affine_rc


def _load(root: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = Path(root)
    with (root / "manifest.json").open(encoding="utf-8") as handle: manifest = json.load(handle)
    with np.load(root / "observations.npz", allow_pickle=False) as data: arrays = {key: np.asarray(data[key]) for key in data.files}
    return manifest, arrays


def _centre(affine: np.ndarray, shape: tuple[int, int]) -> list[float]:
    return apply_affine_rc(affine, np.asarray([(shape[0] - 1) / 2]), np.asarray([(shape[1] - 1) / 2]))[0].tolist()


def compare_prepared(cropped_root: str | Path, full_root: str | Path, output_json: str | Path) -> dict[str, Any]:
    """Compare grouping, SOP/timing, and physical geometry without comparing processed pixels."""

    output = Path(output_json)
    if output.exists(): raise FileExistsError(f"Refusing to overwrite Q001 QC report: {output}")
    cropped_manifest, cropped = _load(cropped_root); full_manifest, full = _load(full_root)
    cropped_shape, full_shape = list(cropped["images"].shape[1:]), list(full["images"].shape[1:])
    full_affine, cropped_affine = full["affine_lps_rc"][0], cropped["affine_lps_rc"][0]
    delta = cropped_affine[:3, 3] - full_affine[:3, 3]
    basis = np.column_stack((full_affine[:3, 0], full_affine[:3, 1]))
    offsets = np.linalg.lstsq(basis, delta, rcond=None)[0]
    report = {
        "schema": "q001_cropped_vs_full_fov_qc/v1", "full_spatial_mode": full_manifest.get("spatial_mode"), "cropped_spatial_mode": cropped_manifest.get("spatial_mode", "baseline_cropped"),
        "full_shape_rows_cols": full_shape, "cropped_shape_rows_cols": cropped_shape, "full_crop_offsets_zero_based": full_manifest.get("crop_offsets_zero_based", [None, None]),
        "expected_cropped_location_in_full_zero_based": [int(round(offsets[0])), int(round(offsets[1]))],
        "full_group_count": int(np.unique(full["group_idx"]).size), "cropped_group_count": int(np.unique(cropped["group_idx"]).size), "full_observation_count": int(full["images"].shape[0]), "cropped_observation_count": int(cropped["images"].shape[0]),
        "sop_correspondence": bool(np.array_equal(cropped["sop_instance_uid"], full["sop_instance_uid"])), "timing_equal": bool(np.array_equal(np.load(Path(cropped_root) / "timing.npy"), np.load(Path(full_root) / "timing.npy"))),
        "tr_vps_equal": bool(np.array_equal(cropped["tr_ms"], full["tr_ms"]) and np.array_equal(cropped["vps"], full["vps"])), "pixel_spacing_equal": bool(np.array_equal(cropped["pixel_spacing_rc_mm"], full["pixel_spacing_rc_mm"])), "slice_thickness_equal": bool(np.array_equal(cropped["slice_thickness_mm"], full["slice_thickness_mm"])),
        "full_dicom_centre_lps_mm": _centre(full_affine, tuple(full_shape)), "cropped_centre_lps_mm": _centre(cropped_affine, tuple(cropped_shape)), "pixel_value_equality_required": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cropped-prepared", required=True); parser.add_argument("--full-prepared", required=True); parser.add_argument("--output", required=True)
    print(json.dumps(compare_prepared(parser.parse_args().cropped_prepared, parser.parse_args().full_prepared, parser.parse_args().output), indent=2))


if __name__ == "__main__": main()
