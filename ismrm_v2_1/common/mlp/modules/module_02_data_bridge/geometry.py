"""DICOM-LPS geometry helpers for the HHZ central crop."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from third_party.nesvor.nesvor.transform import RigidTransform


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
    if not np.isclose(np.linalg.norm(col_direction), 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError("ImageOrientationPatient first direction is not unit length.")
    if not np.isclose(np.linalg.norm(row_direction), 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError("ImageOrientationPatient second direction is not unit length.")
    if not np.isclose(np.dot(col_direction, row_direction), 0.0, rtol=0.0, atol=1e-5):
        raise ValueError("ImageOrientationPatient directions are not orthogonal.")
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


def lps_to_ras(points_lps_mm: np.ndarray) -> np.ndarray:
    """Convert DICOM LPS world points to the one project-wide RAS-mm convention."""

    points_lps_mm = np.asarray(points_lps_mm, dtype=np.float64)
    if points_lps_mm.shape[-1] != 3:
        raise ValueError(f"LPS points must end in 3 coordinates, got {points_lps_mm.shape}.")
    return points_lps_mm * np.array([-1.0, -1.0, 1.0], dtype=np.float64)


def cropped_affine_lps_rc_to_initial_rigid(
    affine_lps_rc: np.ndarray,
    image_shape_rows_cols: tuple[int, int],
    resolution_xyz_mm: np.ndarray,
    device: torch.device | str = "cpu",
) -> RigidTransform:
    """Construct NeSVoR ``RigidTransform`` from centered local xyz to RAS world.

    The local contract is x=column, y=row, z=slice. ``RigidTransform`` uses
    ``trans_first=True``, i.e. world = R @ (local + T), so T is solved from the
    cropped image centre rather than introducing a project-local Euler system.
    """

    affine = np.asarray(affine_lps_rc, dtype=np.float64)
    resolution_xyz_mm = np.asarray(resolution_xyz_mm, dtype=np.float64)
    rows, cols = image_shape_rows_cols
    if affine.shape != (4, 4) or rows < 1 or cols < 1:
        raise ValueError("Expected 4x4 cropped affine and positive [rows, cols].")
    if resolution_xyz_mm.shape != (3,) or np.any(~np.isfinite(resolution_xyz_mm)) or np.any(
        resolution_xyz_mm <= 0
    ):
        raise ValueError(f"resolution_xyz_mm must be three positive values, got {resolution_xyz_mm}.")
    lps_to_ras_matrix = np.diag([-1.0, -1.0, 1.0])
    # affine columns are [row, col, slice], while local axes are [col, row, slice].
    rotation_lps = np.column_stack(
        (
            affine[:3, 1] / resolution_xyz_mm[0],
            affine[:3, 0] / resolution_xyz_mm[1],
            affine[:3, 2] / resolution_xyz_mm[2],
        )
    )
    rotation_ras = lps_to_ras_matrix @ rotation_lps
    if not np.allclose(rotation_ras.T @ rotation_ras, np.eye(3), rtol=1e-5, atol=1e-5):
        raise ValueError("Cropped DICOM affine does not induce an orthonormal local-to-RAS rotation.")
    centre_lps = apply_affine_rc(
        affine,
        np.asarray([(rows - 1) / 2.0]),
        np.asarray([(cols - 1) / 2.0]),
    )[0]
    centre_ras = lps_to_ras(centre_lps)
    translation_pre_rotation = rotation_ras.T @ centre_ras
    matrix = np.concatenate((rotation_ras, translation_pre_rotation[:, None]), axis=1)
    matrix_torch = torch.as_tensor(matrix[None], dtype=torch.float32, device=device)
    return RigidTransform(matrix_torch, trans_first=True)
