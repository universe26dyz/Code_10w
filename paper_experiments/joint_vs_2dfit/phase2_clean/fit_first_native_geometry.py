"""Exact NeSVoR I/O geometry helpers for Fit-first native-plane endpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from trad.third_party.nesvor.nesvor.image import load_slices, load_stack
from trad.third_party.nesvor.nesvor.transform import transform_points


def load_initial_stack_slices(path: str | Path) -> list[Any]:
    """Use NeSVoR's own load_stack/affine2transformation path, on CPU."""
    stack = load_stack(path, device=torch.device("cpu"))
    return [stack[index] for index in range(len(stack))]


def load_final_output_slices(path: str | Path) -> list[Any]:
    """Use NeSVoR's own load_slices/transformation path, on CPU."""
    return load_slices(path, device=torch.device("cpu"))


def negative_determinant_reindex(input_stack_nifti: str | Path, original_shape_rc: tuple[int, int]) -> dict[str, Any]:
    """Exact original [row,col] -> output NIfTI [x,y] mapping from affine2transformation.

    ``export_nesvor_native_inputs`` writes source [row,col,group] as NIfTI
    [x=row,y=col,z=group].  NeSVoR loads it as internal [group,col,row] and
    flips its final (row) axis iff the input affine has negative determinant.
    ``save_nii_volume`` writes that internal final axis back as NIfTI x.
    """
    import nibabel as nib

    affine = np.asarray(nib.load(str(input_stack_nifti)).affine, float)
    flip_row = bool(np.linalg.det(affine[:3, :3]) < 0)
    rows, cols = original_shape_rc
    return {"source": "nesvor.image.image_utils.affine2transformation negative_det branch", "input_affine_determinant": float(np.linalg.det(affine[:3, :3])), "original_shape_rc": [rows, cols], "output_nifti_x_from_original_row": "H-1-row" if flip_row else "row", "output_nifti_y_from_original_col": "col", "flip_original_row": flip_row}


def original_to_output_voxels(shape_rc: tuple[int, int], descriptor: dict[str, Any]) -> np.ndarray:
    rows, cols = np.indices(shape_rc, dtype=float)
    x = (shape_rc[0] - 1 - rows) if descriptor["flip_original_row"] else rows
    return np.stack((x, cols, np.zeros_like(rows), np.ones_like(rows)), axis=0)


def final_world_grid_original_order(final_output_slice: str | Path, shape_rc: tuple[int, int], descriptor: dict[str, Any]) -> np.ndarray:
    import nibabel as nib

    affine = np.asarray(nib.load(str(final_output_slice)).affine, float)
    return (affine @ original_to_output_voxels(shape_rc, descriptor).reshape(4, -1)).reshape(4, *shape_rc)


def slice_world_corners(slice_image: Any) -> np.ndarray:
    """Physical corners derived from NeSVoR internal Slice transformation."""
    h, w = (int(value) for value in slice_image.image.shape[-2:])
    local = torch.tensor([[-(w - 1) / 2 * slice_image.resolution_x, -(h - 1) / 2 * slice_image.resolution_y, 0.0], [(w - 1) / 2 * slice_image.resolution_x, -(h - 1) / 2 * slice_image.resolution_y, 0.0], [-(w - 1) / 2 * slice_image.resolution_x, (h - 1) / 2 * slice_image.resolution_y, 0.0], [(w - 1) / 2 * slice_image.resolution_x, (h - 1) / 2 * slice_image.resolution_y, 0.0]], dtype=torch.float32)
    return transform_points(slice_image.transformation, local).detach().cpu().numpy()


def initial_world_grid_original_order(initial_slice: Any, shape_rc: tuple[int, int], descriptor: dict[str, Any]) -> np.ndarray:
    rows, cols = np.indices(shape_rc, dtype=np.float32)
    internal_w = (shape_rc[0] - 1 - rows) if descriptor["flip_original_row"] else rows
    internal_h = cols
    local = np.stack(((internal_w - (initial_slice.image.shape[-1] - 1) / 2) * initial_slice.resolution_x, (internal_h - (initial_slice.image.shape[-2] - 1) / 2) * initial_slice.resolution_y, np.zeros_like(rows)), axis=-1)
    points = torch.as_tensor(local.reshape(-1, 3), dtype=torch.float32)
    return transform_points(initial_slice.transformation, points).detach().cpu().numpy().reshape(*shape_rc, 3)


def pose_delta(initial: Any, final: Any) -> dict[str, float]:
    """True NeSVoR internal-rigid pose difference, independent of voxel axes."""
    initial_matrix = initial.transformation.matrix(trans_first=True)[0].detach().cpu().numpy()
    final_matrix = final.transformation.matrix(trans_first=True)[0].detach().cpu().numpy()
    # NeSVoR exposes a 3x4 rigid matrix; form homogeneous matrices before
    # taking the relative transform.
    initial_h = np.eye(4, dtype=float); initial_h[:3] = initial_matrix
    final_h = np.eye(4, dtype=float); final_h[:3] = final_matrix
    relative = final_h @ np.linalg.inv(initial_h)
    cosine = np.clip((np.trace(relative[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    center_initial = initial_h[:3, 3]
    center_final = final_h[:3, 3]
    return {"rotation_deg": float(np.degrees(np.arccos(cosine))), "relative_translation_mm": float(np.linalg.norm(relative[:3, 3])), "center_displacement_mm": float(np.linalg.norm(center_final - center_initial))}
