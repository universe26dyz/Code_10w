"""Read physical final-pose payloads without changing reconstruction geometry."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def pose_delta_rows(stack: str, initial_axisangle: np.ndarray, final_axisangle: np.ndarray, *, group_offset: int = 0) -> list[dict[str, object]]:
    """Report initial-to-final translation/rotation and within-stack neighbour centers."""

    initial, final = np.asarray(initial_axisangle, float), np.asarray(final_axisangle, float)
    if initial.ndim != 2 or final.shape != initial.shape or initial.shape[1] != 6:
        raise ValueError("Pose audit requires matching [group,6] trans-first physical axis-angle arrays.")
    rotation = Rotation.from_rotvec(final[:, 3:])
    centers = rotation.apply(final[:, :3])
    relative = rotation * Rotation.from_rotvec(initial[:, 3:]).inv()
    rows = []
    for group in range(final.shape[0]):
        rows.append({
            "stack": stack,
            "group_idx": int(group),
            "global_group_idx": int(group_offset + group),
            "translation_magnitude_mm": float(np.linalg.norm(final[group, :3] - initial[group, :3])),
            "rotation_magnitude_deg": float(np.degrees(relative[group].magnitude())),
            "final_center_ras_mm": centers[group].tolist(),
            "slice_normal_ras": rotation[group].apply(np.array([0.0, 0.0, 1.0])).tolist(),
            "neighbor_center_distance_mm": float(np.linalg.norm(centers[group + 1] - centers[group])) if group + 1 < final.shape[0] else float("nan"),
        })
    return rows
