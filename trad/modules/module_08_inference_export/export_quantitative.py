"""Export training-space QuantitativeINR fields on an explicit physical RAS grid."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from modules.module_07_objective_training.training_space import TrainingSpace
from third_party.nesvor.nesvor.transform import RigidTransform


def _physical_grid(bbox_ras_mm: torch.Tensor, resolution_mm: float) -> tuple[torch.Tensor, tuple[int, int, int], np.ndarray]:
    if resolution_mm <= 0:
        raise ValueError("output_resolution_mm must be positive.")
    bbox = bbox_ras_mm.detach().cpu().numpy()
    axes = [np.arange(bbox[0, axis], bbox[1, axis] + resolution_mm * 0.5, resolution_mm, dtype=np.float32) for axis in range(3)]
    if any(axis.size < 1 for axis in axes):
        raise ValueError("Physical reconstruction bounding box has no sample points.")
    mesh = np.meshgrid(*axes, indexing="ij")
    points = torch.from_numpy(np.stack(mesh, axis=-1).reshape(-1, 3))
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = resolution_mm
    affine[:3, 3] = bbox[0]
    return points, tuple(axis.size for axis in axes), affine


def sample_quantitative_fields(model: torch.nn.Module, space: TrainingSpace, output_resolution_mm: float, batch_size: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Sample T1/T2/B1/A by physical RAS → the sole training-space conversion."""

    if batch_size <= 0:
        raise ValueError("output_batch_size must be positive.")
    if not hasattr(model, "intensity_scale"):
        raise ValueError("Model lacks required intensity_scale provenance for amplitude export.")
    intensity_scale = torch.as_tensor(model.intensity_scale)
    if intensity_scale.numel() != 1 or not torch.isfinite(intensity_scale).all() or intensity_scale.item() <= 0:
        raise ValueError("Model intensity_scale must be one finite positive scalar for amplitude export.")
    physical, shape, affine = _physical_grid(space.physical_bbox_ras_mm, output_resolution_mm)
    device = next(model.parameters()).device
    result = {"t1_ms": [], "t2_ms": [], "b1": [], "amplitude": []}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, physical.shape[0], batch_size):
            train_points = space.physical_ras_to_train(physical[start : start + batch_size].to(device))
            fields = model.inr(train_points)
            for key in result:
                value = fields[key] * intensity_scale if key == "amplitude" else fields[key]
                result[key].append(value.detach().cpu())
    if was_training:
        model.train()
    return {key: torch.cat(value).reshape(shape).numpy().astype(np.float32) for key, value in result.items()}, affine


def _physical_pose_payload(model: torch.nn.Module, space: TrainingSpace) -> dict[str, Any]:
    physical_axisangle = space.axisangle_train_to_physical(model.rigid_psf.axisangle.detach())
    matrices = RigidTransform(physical_axisangle, trans_first=True).matrix(trans_first=True).detach().cpu().tolist()
    return {
        "coordinate_convention": "physical RAS mm; trans_first=true; world=R@(local+T)",
        "center_ras_mm": space.center_ras_mm.detach().cpu().tolist(),
        "spatial_scaling": space.spatial_scaling,
        "axisangle_physical": physical_axisangle.detach().cpu().tolist(),
        "matrix_physical": matrices,
    }


def export_quantitative_outputs(model: torch.nn.Module, space: TrainingSpace, output_dir: str | Path, output_resolution_mm: float, output_batch_size: int) -> dict[str, Path]:
    """Write the four required NIfTI volumes and physical final group poses."""

    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("nibabel is required for requested NIfTI export.") from exc
    output = Path(output_dir)
    if not output.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output}")
    fields, affine = sample_quantitative_fields(model, space, output_resolution_mm, output_batch_size)
    names = {"t1_ms": "T1_3D.nii.gz", "t2_ms": "T2_3D.nii.gz", "b1": "B1_3D.nii.gz", "amplitude": "amplitude_3D.nii.gz"}
    paths: dict[str, Path] = {}
    for field, filename in names.items():
        path = output / filename
        image = nib.Nifti1Image(fields[field], affine)
        image.header.set_xyzt_units("mm")
        image.header["descrip"] = b"physical RAS-mm quantitative field"
        nib.save(image, path)
        paths[field] = path
    pose_path = output / "final_rigid_poses.json"
    with pose_path.open("w", encoding="utf-8") as handle:
        json.dump(_physical_pose_payload(model, space), handle, indent=2)
    paths["poses"] = pose_path
    return paths
