"""NeSVoR-compatible centering/scaling adapter without mutating physical data."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset
from third_party.nesvor.nesvor.transform import RigidTransform


@dataclass(frozen=True)
class TrainingSpace:
    """Physical RAS-mm <-> centered/scaled training-space conversion.

    This is the direct grouped analogue of vendored ``nesvor.inr.train``:
    compose ``-center`` with physical group poses, then divide translation and
    local geometry by ``spatial_scaling``.
    """

    center_ras_mm: torch.Tensor
    spatial_scaling: float
    physical_bbox_ras_mm: torch.Tensor
    bbox_train: torch.Tensor
    group_axisangle_init_physical: torch.Tensor
    group_axisangle_init_train: torch.Tensor
    group_resolution_xyz_mm: torch.Tensor
    group_resolution_train: torch.Tensor

    @classmethod
    def from_dataset(cls, dataset: QuantPointDataset, spatial_scaling: float) -> "TrainingSpace":
        if spatial_scaling <= 0:
            raise ValueError("spatial_scaling must be positive.")
        physical_bbox = dataset.bounding_box.detach().clone()
        center = (physical_bbox[0] + physical_bbox[1]) / 2.0
        center_transform = RigidTransform(
            torch.cat((torch.zeros_like(center), -center))[None], trans_first=True
        )
        physical = RigidTransform(dataset.group_axisangle_init.detach().clone(), trans_first=True)
        train_axisangle = center_transform.compose(physical).axisangle(trans_first=True)
        train_axisangle[:, 3:] /= spatial_scaling
        return cls(
            center_ras_mm=center,
            spatial_scaling=float(spatial_scaling),
            physical_bbox_ras_mm=physical_bbox,
            bbox_train=(physical_bbox - center) / spatial_scaling,
            group_axisangle_init_physical=dataset.group_axisangle_init.detach().clone(),
            group_axisangle_init_train=train_axisangle,
            group_resolution_xyz_mm=dataset.group_resolution_xyz_mm.detach().clone(),
            group_resolution_train=dataset.group_resolution_xyz_mm.detach().clone() / spatial_scaling,
        )

    def local_to_train(self, local_xyz_mm: torch.Tensor) -> torch.Tensor:
        return local_xyz_mm / self.spatial_scaling

    def physical_ras_to_train(self, physical_ras_mm: torch.Tensor) -> torch.Tensor:
        return (physical_ras_mm - self.center_ras_mm.to(physical_ras_mm)) / self.spatial_scaling

    def train_ras_to_physical(self, train_ras: torch.Tensor) -> torch.Tensor:
        return train_ras * self.spatial_scaling + self.center_ras_mm.to(train_ras)

    def axisangle_train_to_physical(self, axisangle_train: torch.Tensor) -> torch.Tensor:
        if axisangle_train.ndim != 2 or axisangle_train.shape[1] != 6:
            raise ValueError("axisangle_train must be [G,6].")
        axisangle = axisangle_train.clone()
        axisangle[:, 3:] *= self.spatial_scaling
        centered = RigidTransform(axisangle, trans_first=True)
        return RigidTransform(
            torch.cat((torch.zeros_like(self.center_ras_mm), self.center_ras_mm))[None], trans_first=True
        ).compose(centered).axisangle(trans_first=True)

    def axisangle_physical_to_train(self, axisangle_physical: torch.Tensor) -> torch.Tensor:
        physical = RigidTransform(axisangle_physical, trans_first=True)
        centered = RigidTransform(
            torch.cat((torch.zeros_like(self.center_ras_mm), -self.center_ras_mm))[None], trans_first=True
        ).compose(physical).axisangle(trans_first=True)
        centered[:, 3:] /= self.spatial_scaling
        return centered

    def state_dict(self) -> dict[str, object]:
        return {
            "center_ras_mm": self.center_ras_mm.detach().cpu(),
            "spatial_scaling": self.spatial_scaling,
            "physical_bbox_ras_mm": self.physical_bbox_ras_mm.detach().cpu(),
            "bbox_train": self.bbox_train.detach().cpu(),
            "group_resolution_xyz_mm": self.group_resolution_xyz_mm.detach().cpu(),
            "group_axisangle_init_physical": self.group_axisangle_init_physical.detach().cpu(),
            "group_axisangle_init_train": self.group_axisangle_init_train.detach().cpu(),
        }
