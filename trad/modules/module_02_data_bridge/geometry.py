"""DICOM-LPS geometry helpers for the HHZ central crop."""

from __future__ import annotations

from typing import Any

import numpy as np


def _vector(dataset: Any, field: str, length: int) -> np.ndarray:
    if not hasattr(dataset, field):
        raise ValueError(f"DICOM lacks required geometry field {field}.")
    value = np.asarray(getattr(dataset, field), dtype=np.float64)
    if value.shape != (length,) or not np.all(np.isfinite(value)):
        raise ValueError(f"DICOM {field} must be {length} finite values, got {value}.")
    return value


def dicom_lps_affine_rc(dataset: Any, slice_thickness_mm: float) -> np.ndarray:
    """Return a 4x4 LPS-mm affine mapping homogeneous ``[row,col,slice,1]``."""

    origin = _vector(dataset, "ImagePositionPatient", 3)
    orientation = _vector(dataset, "ImageOrientationPatient", 6)
    spacing_rc = _vector(dataset, "PixelSpacing", 2)
    if not np.isfinite(slice_thickness_mm) or slice_thickness_mm <= 0:
        raise ValueError(f"SliceThickness must be positive and finite, got {slice_thickness_mm}.")
    row_direction = orientation[3:]  # incrementing DICOM row
    col_direction = orientation[:3]  # incrementing DICOM column
    normal_direction = np.cross(col_direction, row_direction)
    if not np.isclose(np.linalg.norm(normal_direction), 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError("ImageOrientationPatient is not an orthonormal DICOM orientation.")
    affine = np.eye(4, dtype=np.float64)
    affine[:3, 0] = row_direction * spacing_rc[0]
    affine[:3, 1] = col_direction * spacing_rc[1]
    affine[:3, 2] = normal_direction * float(slice_thickness_mm)
    affine[:3, 3] = origin
    return affine


def crop_affine_lps_rc(
    full_affine_lps_rc: np.ndarray, row_start_zero_based: int, col_start_zero_based: int
) -> np.ndarray:
    """Offset a full DICOM affine so cropped pixel ``[0,0]`` is its origin."""

    if row_start_zero_based < 0 or col_start_zero_based < 0:
        raise ValueError("Crop offsets must be non-negative zero-based pixel indices.")
    affine = np.asarray(full_affine_lps_rc, dtype=np.float64)
    if affine.shape != (4, 4):
        raise ValueError(f"Expected 4x4 affine, got {affine.shape}.")
    cropped = affine.copy()
    cropped[:3, 3] = (
        affine[:3, 3]
        + int(row_start_zero_based) * affine[:3, 0]
        + int(col_start_zero_based) * affine[:3, 1]
    )
    return cropped


def apply_affine_rc(affine_lps_rc: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Map equally shaped zero-based row/column arrays to LPS-mm world points."""

    affine_lps_rc = np.asarray(affine_lps_rc, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.float64)
    cols = np.asarray(cols, dtype=np.float64)
    if affine_lps_rc.shape != (4, 4) or rows.shape != cols.shape:
        raise ValueError("Affine must be 4x4 and rows/cols must have the same shape.")
    return (
        affine_lps_rc[:3, 3]
        + rows[..., None] * affine_lps_rc[:3, 0]
        + cols[..., None] * affine_lps_rc[:3, 1]
    )
