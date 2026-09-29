"""Full-FOV metadata guard around the existing SOP-resolved baseline bridge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from trad.modules.module_02_data_bridge.mat_v73 import load_preprocessed_v73
from trad.modules.module_02_data_bridge.prepare_observations import prepare_observations


FULL_FOV_REQUIRED = ("spatial_mode_ascii", "final_data_semantics_ascii", "geometry_rule_ascii", "full_fov_shape_rows_cols")


def _matlab_numeric(dataset: h5py.Dataset) -> np.ndarray:
    value = np.asarray(dataset)
    return value.transpose(tuple(range(value.ndim - 1, -1, -1))) if value.ndim > 1 else value


def _ascii(value: np.ndarray, name: str) -> str:
    try:
        return bytes(np.asarray(value, dtype=np.uint8).reshape(-1)).decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Q001 MAT {name} must be ASCII metadata.") from exc


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_full_fov_preprocessed(mat_path: str | Path) -> dict[str, Any]:
    """Load only a v7.3 MAT which proves it retained the raw full spatial FOV."""

    base = load_preprocessed_v73(mat_path)
    with h5py.File(mat_path, "r") as handle:
        missing = [name for name in FULL_FOV_REQUIRED if name not in handle]
        if missing:
            raise ValueError(f"Q001 full_fov MAT lacks required metadata: {missing}")
        metadata = {name: _matlab_numeric(handle[name]) for name in FULL_FOV_REQUIRED}
    spatial_mode = _ascii(metadata["spatial_mode_ascii"], "spatial_mode_ascii")
    if spatial_mode != "full_fov":
        raise ValueError(f"Q001 MAT spatial_mode must be full_fov, got {spatial_mode!r}")
    full_shape = tuple(int(value) for value in np.asarray(metadata["full_fov_shape_rows_cols"]).reshape(-1))
    if len(full_shape) != 2 or full_shape != base["mag"].shape[:2] or base["mag_crop"].shape[:2] != full_shape:
        raise ValueError("Q001 full_fov MAT must preserve full Mag/Mag_crop spatial shape.")
    if base["crop_row_start_zero_based"] != 0 or base["crop_col_start_zero_based"] != 0:
        raise ValueError("Q001 full_fov MAT crop offsets must both be zero.")
    return base | {
        "spatial_mode": spatial_mode,
        "full_fov_shape_rows_cols": list(full_shape),
        "final_data_semantics": _ascii(metadata["final_data_semantics_ascii"], "final_data_semantics_ascii"),
        "geometry_rule": _ascii(metadata["geometry_rule_ascii"], "geometry_rule_ascii"),
    }


def prepare_full_fov_observations(preprocessed_mat: str | Path, dicom_dir: str | Path, stack: str, output_dir: str | Path, max_groups: int | None = None) -> dict[str, Any]:
    """Use the unmodified baseline bridge only after Q001 full-FOV validation."""

    preprocessed_mat, dicom_dir, output_dir = Path(preprocessed_mat), Path(dicom_dir), Path(output_dir)
    full = load_full_fov_preprocessed(preprocessed_mat)
    result = prepare_observations(preprocessed_mat, dicom_dir, stack, output_dir, max_groups=max_groups)
    with (output_dir / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest.update({
        "spatial_mode": "full_fov", "preprocessed_mat": str(preprocessed_mat.resolve()), "preprocessed_mat_sha256": _sha256(preprocessed_mat),
        "full_fov_shape_rows_cols": full["full_fov_shape_rows_cols"], "crop_offsets_zero_based": [0, 0],
        "final_data_semantics": full["final_data_semantics"], "geometry_rule": full["geometry_rule"],
        "dicom_provenance": {"directory": str(dicom_dir.resolve()), "file_count": len(list(dicom_dir.glob("*.dcm")))},
    })
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle: json.dump(manifest, handle, indent=2, ensure_ascii=False)
    with (output_dir / "qc_summary.json").open(encoding="utf-8") as handle: qc = json.load(handle)
    qc.update({"spatial_mode": "full_fov", "full_fov_shape_rows_cols": full["full_fov_shape_rows_cols"], "crop_offsets_zero_based": [0, 0], "preprocessed_mat_sha256": _sha256(preprocessed_mat), "full_fov_geometry": "affine_lps_rc equals original full DICOM affine"})
    with (output_dir / "qc_summary.json").open("w", encoding="utf-8") as handle: json.dump(qc, handle, indent=2, ensure_ascii=False)
    return {"manifest": manifest, "qc_summary": qc}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessed-mat", required=True); parser.add_argument("--dicom-dir", required=True); parser.add_argument("--stack", required=True, choices=("sax", "2ch", "4ch")); parser.add_argument("--output-dir", required=True); parser.add_argument("--max-groups", type=int)
    args = parser.parse_args(); result = prepare_full_fov_observations(**vars(args)); print(json.dumps(result["qc_summary"], sort_keys=True))


if __name__ == "__main__": main()
