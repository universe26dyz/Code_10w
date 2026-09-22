"""Export training-space QuantitativeINR fields on an explicit physical RAS grid."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from trad.modules.module_07_objective_training.training_space import TrainingSpace
from trad.third_party.nesvor.nesvor.transform import RigidTransform, ax_transform_points


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


def sample_quantitative_fields(model: torch.nn.Module, space: TrainingSpace, output_resolution_mm: float, batch_size: int, *, bbox_ras_mm: torch.Tensor | None = None) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Sample T1/T2/B1/A by physical RAS → the sole training-space conversion."""

    if batch_size <= 0:
        raise ValueError("output_batch_size must be positive.")
    if not hasattr(model, "intensity_scale"):
        raise ValueError("Model lacks required intensity_scale provenance for amplitude export.")
    intensity_scale = torch.as_tensor(model.intensity_scale)
    if intensity_scale.numel() != 1 or not torch.isfinite(intensity_scale).all() or intensity_scale.item() <= 0:
        raise ValueError("Model intensity_scale must be one finite positive scalar for amplitude export.")
    physical, shape, affine = _physical_grid(space.physical_bbox_ras_mm if bbox_ras_mm is None else bbox_ras_mm, output_resolution_mm)
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
        "dicom_initial_axisangle_physical": space.group_axisangle_dicom_physical.detach().cpu().tolist(),
        "post_stack_init_axisangle_physical": space.group_axisangle_init_physical.detach().cpu().tolist(),
        "axisangle_physical": physical_axisangle.detach().cpu().tolist(),
        "matrix_physical": matrices,
    }


def _final_support_bbox(model: torch.nn.Module, space: TrainingSpace, dataset: Any, margin_mm: float) -> torch.Tensor:
    poses = space.axisangle_train_to_physical(model.rigid_psf.axisangle.detach())
    world = ax_transform_points(poses[dataset.group_idx], dataset.xyz, trans_first=True)
    bbox = torch.stack((world.amin(0), world.amax(0)), 0)
    bbox[0] -= margin_mm; bbox[1] += margin_mm
    return bbox


def _gradient_fields(model: torch.nn.Module, space: TrainingSpace, bbox: torch.Tensor, resolution_mm: float, batch_size: int) -> dict[str, np.ndarray]:
    physical, shape, _ = _physical_grid(bbox, resolution_mm)
    device = next(model.parameters()).device
    result = {"gradient_amplitude": [], "gradient_t1": [], "gradient_t2": [], "gradient_b1": []}
    was_training = model.training; model.eval()
    for start in range(0, physical.shape[0], batch_size):
        points = space.physical_ras_to_train(physical[start : start + batch_size].to(device)).detach().requires_grad_(True)
        fields = model.inr(points)
        for output, field in (("gradient_amplitude", "amplitude"), ("gradient_t1", "t1_ms"), ("gradient_t2", "t2_ms"), ("gradient_b1", "b1")):
            gradient = torch.autograd.grad(fields[field].sum(), points, retain_graph=True)[0] / space.spatial_scaling
            result[output].append(torch.linalg.vector_norm(gradient, dim=-1).detach().cpu())
    if was_training: model.train()
    return {key: torch.cat(values).reshape(shape).numpy().astype(np.float32) for key, values in result.items()}


def _support_and_coverage(model: torch.nn.Module, space: TrainingSpace, dataset: Any, bbox: torch.Tensor, resolution_mm: float) -> tuple[np.ndarray, np.ndarray]:
    _, shape, affine = _physical_grid(bbox, resolution_mm)
    poses = space.axisangle_train_to_physical(model.rigid_psf.axisangle.detach())
    world = ax_transform_points(poses[dataset.group_idx], dataset.xyz, trans_first=True).detach().cpu().numpy()
    origin = affine[:3, 3]
    voxel = np.rint((world - origin) / resolution_mm).astype(np.int64)
    valid = np.all((voxel >= 0) & (voxel < np.asarray(shape)), axis=1)
    flat = np.ravel_multi_index(tuple(voxel[valid].T), shape)
    grouped = np.column_stack((dataset.group_idx.detach().cpu().numpy()[valid], flat))
    unique = np.unique(grouped, axis=0)
    coverage = np.bincount(unique[:, 1], minlength=int(np.prod(shape))).reshape(shape).astype(np.float32)
    return (coverage > 0).astype(np.uint8), coverage


def export_quantitative_outputs(model: torch.nn.Module, space: TrainingSpace, output_dir: str | Path, output_resolution_mm: float, output_batch_size: int, *, dataset: Any | None = None, export_config: dict[str, Any] | None = None) -> dict[str, Path]:
    """Write the four required NIfTI volumes and physical final group poses."""

    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("nibabel is required for requested NIfTI export.") from exc
    output = Path(output_dir)
    if not output.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output}")
    settings = export_config or {}
    bbox_settings = settings.get("bbox", {})
    use_final_support = bbox_settings.get("mode") == "final_registered_support"
    bbox = _final_support_bbox(model, space, dataset, float(bbox_settings.get("margin_mm", 0.0))) if use_final_support and dataset is not None else space.physical_bbox_ras_mm
    fields, affine = sample_quantitative_fields(model, space, output_resolution_mm, output_batch_size, bbox_ras_mm=bbox)
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
    gradients = _gradient_fields(model, space, bbox, output_resolution_mm, output_batch_size)
    for field, values in gradients.items():
        path = output / f"{field}_3D.nii.gz"
        nib.save(nib.Nifti1Image(values, affine), path)
        paths[field] = path
    amplitude_mask = (fields["amplitude"] >= float(settings.get("amplitude_qc_relative_threshold", 0.05)) * float(np.max(fields["amplitude"]))).astype(np.uint8)
    amplitude_path = output / "amplitude_qc_mask.nii.gz"
    nib.save(nib.Nifti1Image(amplitude_mask, affine), amplitude_path)
    paths["amplitude_qc_mask"] = amplitude_path
    if dataset is not None:
        support, coverage = _support_and_coverage(model, space, dataset, bbox, output_resolution_mm)
        for key, values in (("support_mask", support), ("coverage", coverage)):
            path = output / f"{key}_3D.nii.gz"
            nib.save(nib.Nifti1Image(values, affine), path)
            paths[key] = path
    return paths
