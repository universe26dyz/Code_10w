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


def validate_acceptance(report: dict[str, Any], tolerance_mm: float = 1e-6) -> bool:
    """Reject input pairs that cannot be used as a Q001 full-FOV correspondence audit."""

    required = {
        "full spatial mode": report.get("full_spatial_mode") == "full_fov",
        "full crop offsets": report.get("full_crop_offsets_zero_based") == [0, 0],
        "SOP correspondence": report.get("sop_correspondence") is True,
        "timing": report.get("timing_equal") is True,
        "TR/VPS": report.get("tr_vps_equal") is True,
        "pixel spacing": report.get("pixel_spacing_equal") is True,
        "slice thickness": report.get("slice_thickness_equal") is True,
        "group count": report.get("full_group_count") == report.get("cropped_group_count"),
        "observation count": report.get("full_observation_count") == report.get("cropped_observation_count"),
        "affine origin relationship": report.get("affine_origin_relationship_within_tolerance") is True,
    }
    failures = [name for name, passed in required.items() if not passed]
    offset = np.asarray(report.get("expected_cropped_location_in_full_zero_based", []), dtype=float)
    full_shape = np.asarray(report.get("full_shape_rows_cols", []), dtype=float)
    cropped_shape = np.asarray(report.get("cropped_shape_rows_cols", []), dtype=float)
    if offset.shape != (2,) or not np.all(np.isfinite(offset)) or not np.allclose(offset, np.round(offset), atol=tolerance_mm): failures.append("crop offset must be finite integer")
    elif full_shape.shape != (2,) or cropped_shape.shape != (2,) or np.any(offset < 0) or np.any(offset + cropped_shape > full_shape): failures.append("crop offset must be inside full FOV")
    if failures: raise ValueError("Q001 strict QC failed: " + "; ".join(failures))
    return True


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
    baseline_group = cropped_manifest.get("groups", [{}])[0]
    declared_affine = np.asarray(baseline_group.get("affine_lps_rc", cropped_affine), dtype=float)
    declared_full_affine = np.asarray(baseline_group.get("full_affine_lps_rc", full_affine), dtype=float)
    declared_basis = np.column_stack((declared_full_affine[:3, 0], declared_full_affine[:3, 1]))
    declared_offsets = np.linalg.lstsq(declared_basis, declared_affine[:3, 3] - declared_full_affine[:3, 3], rcond=None)[0]
    expected_origins = full["affine_lps_rc"][:, :3, 3] + np.einsum("nij,j->ni", full["affine_lps_rc"][:, :3, :2], declared_offsets)
    origin_errors_mm = np.linalg.norm(cropped["affine_lps_rc"][:, :3, 3] - expected_origins, axis=1)
    origin_error_mm = float(np.max(origin_errors_mm))
    report = {
        "schema": "q001_cropped_vs_full_fov_qc/v1", "full_spatial_mode": full_manifest.get("spatial_mode"), "cropped_spatial_mode": cropped_manifest.get("spatial_mode", "baseline_cropped"),
        "full_shape_rows_cols": full_shape, "cropped_shape_rows_cols": cropped_shape, "full_crop_offsets_zero_based": full_manifest.get("crop_offsets_zero_based", [None, None]),
        "expected_cropped_location_in_full_zero_based": [float(offsets[0]), float(offsets[1])],
        "declared_cropped_location_in_full_zero_based": [float(declared_offsets[0]), float(declared_offsets[1])],
        "full_group_count": int(np.unique(full["group_idx"]).size), "cropped_group_count": int(np.unique(cropped["group_idx"]).size), "full_observation_count": int(full["images"].shape[0]), "cropped_observation_count": int(cropped["images"].shape[0]),
        "sop_correspondence": bool(np.array_equal(cropped["sop_instance_uid"], full["sop_instance_uid"])), "timing_equal": bool(np.array_equal(np.load(Path(cropped_root) / "timing.npy"), np.load(Path(full_root) / "timing.npy"))),
        "tr_vps_equal": bool(np.array_equal(cropped["tr_ms"], full["tr_ms"]) and np.array_equal(cropped["vps"], full["vps"])), "pixel_spacing_equal": bool(np.array_equal(cropped["pixel_spacing_rc_mm"], full["pixel_spacing_rc_mm"])), "slice_thickness_equal": bool(np.array_equal(cropped["slice_thickness_mm"], full["slice_thickness_mm"])),
        "full_dicom_centre_lps_mm": _centre(full_affine, tuple(full_shape)), "cropped_centre_lps_mm": _centre(cropped_affine, tuple(cropped_shape)),
        "affine_origin_relationship_within_tolerance": bool(origin_error_mm <= 1e-6), "affine_origin_error_mm": origin_error_mm,
        "pixel_value_equality_required": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cropped-prepared", required=True); parser.add_argument("--full-prepared", required=True); parser.add_argument("--output", required=True); parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(); report = compare_prepared(args.cropped_prepared, args.full_prepared, args.output)
    if args.strict: validate_acceptance(report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
