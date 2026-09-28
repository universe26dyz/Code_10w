"""Read physical final-pose payloads without changing reconstruction geometry."""

from __future__ import annotations

import numpy as np
import torch

from trad.third_party.nesvor.nesvor.transform import RigidTransform, ax_transform_points


def pose_delta_rows(stack: str, initial_axisangle: np.ndarray, final_axisangle: np.ndarray, *, group_offset: int = 0) -> list[dict[str, object]]:
    """Report initial-to-final translation/rotation and within-stack neighbour centers."""

    initial, final = np.asarray(initial_axisangle, float), np.asarray(final_axisangle, float)
    if initial.ndim != 2 or final.shape != initial.shape or initial.shape[1] != 6:
        raise ValueError("Pose audit requires matching [group,6] trans-first physical axis-angle arrays.")
    initial_transform = RigidTransform(torch.as_tensor(initial, dtype=torch.float64), trans_first=True)
    final_tensor = torch.as_tensor(final, dtype=torch.float64)
    final_transform = RigidTransform(final_tensor, trans_first=True)
    delta = initial_transform.inv().compose(final_transform).axisangle(trans_first=True).detach().cpu().numpy()
    origin = torch.zeros((final.shape[0], 3), dtype=final_tensor.dtype)
    local_z = torch.zeros_like(origin); local_z[:, 2] = 1.0
    centers = ax_transform_points(final_tensor, origin, trans_first=True).detach().cpu().numpy()
    normals = ax_transform_points(final_tensor, local_z, trans_first=True).detach().cpu().numpy() - centers
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    rows = []
    for group in range(final.shape[0]):
        rows.append({
            "stack": stack,
            "group_idx": int(group),
            "global_group_idx": int(group_offset + group),
            "translation_magnitude_mm": float(np.linalg.norm(delta[group, 3:])),
            "rotation_magnitude_deg": float(np.linalg.norm(delta[group, :3]) * 180.0 / np.pi),
            "final_center_ras_mm": centers[group].tolist(),
            "slice_normal_ras": normals[group].tolist(),
            "neighbor_center_distance_mm": float(np.linalg.norm(centers[group + 1] - centers[group])) if group + 1 < final.shape[0] else float("nan"),
        })
    return rows
