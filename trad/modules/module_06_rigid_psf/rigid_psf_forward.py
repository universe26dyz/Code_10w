"""Rigid group pose and local anisotropic PSF for the Trad forward model."""

from __future__ import annotations

import torch
from torch import nn

from third_party.nesvor.nesvor.transform import RigidTransform, ax_transform_points
from third_party.nesvor.nesvor.utils.psf import resolution2sigma


class GroupRigidPSF(nn.Module):
    """One NeSVoR axis-angle transform per native spatial group, no deformation."""

    deformable = False

    def __init__(self, group_axisangle_init: torch.Tensor, group_resolution_xyz_mm: torch.Tensor) -> None:
        super().__init__()
        initial = torch.as_tensor(group_axisangle_init, dtype=torch.float32)
        resolution = torch.as_tensor(group_resolution_xyz_mm, dtype=torch.float32)
        if initial.ndim != 2 or initial.shape[1] != 6:
            raise ValueError("group_axisangle_init must be [G,6] NeSVoR axis-angle parameters.")
        if resolution.shape != (initial.shape[0], 3) or not torch.isfinite(resolution).all() or torch.any(resolution <= 0):
            raise ValueError("group_resolution_xyz_mm must be finite positive [G,3].")
        self.axisangle = nn.Parameter(initial.clone())
        self.register_buffer("group_resolution_xyz_mm", resolution)

    @property
    def group_count(self) -> int:
        return int(self.axisangle.shape[0])

    @property
    def rigid_transform(self) -> RigidTransform:
        return RigidTransform(self.axisangle, trans_first=True)

    def sigma_for_groups(self, group_idx: torch.Tensor) -> torch.Tensor:
        self._validate_group_idx(group_idx)
        return resolution2sigma(self.group_resolution_xyz_mm[group_idx], isotropic=False)

    def transform_local_to_ras(self, local_xyz_mm: torch.Tensor, group_idx: torch.Tensor) -> torch.Tensor:
        if local_xyz_mm.shape[-1] != 3 or local_xyz_mm.shape[:-1] != group_idx.shape:
            raise ValueError("local_xyz_mm must be [B,3] and group_idx must be [B].")
        self._validate_group_idx(group_idx)
        return ax_transform_points(self.axisangle[group_idx], local_xyz_mm, trans_first=True)

    def sample_local_then_transform(
        self, local_xyz_mm: torch.Tensor, group_idx: torch.Tensor, n_samples: int
    ) -> torch.Tensor:
        """Sample anisotropic PSF in local slice axes, then apply group rigid pose."""

        if n_samples < 1:
            raise ValueError("n_samples must be at least one.")
        if local_xyz_mm.shape != (group_idx.numel(), 3):
            raise ValueError("local_xyz_mm must be [B,3] with one group index per point.")
        self._validate_group_idx(group_idx)
        if n_samples == 1:
            local_samples = local_xyz_mm[:, None, :]
        else:
            noise = torch.randn(
                (local_xyz_mm.shape[0], n_samples, 3), dtype=local_xyz_mm.dtype, device=local_xyz_mm.device
            )
            local_samples = local_xyz_mm[:, None, :] + noise * self.sigma_for_groups(group_idx)[:, None, :]
        axisangle = self.axisangle[group_idx][:, None, :].expand(-1, n_samples, -1).reshape(-1, 6)
        return ax_transform_points(axisangle, local_samples.reshape(-1, 3), trans_first=True).reshape_as(local_samples)

    def _validate_group_idx(self, group_idx: torch.Tensor) -> None:
        if group_idx.ndim != 1 or group_idx.dtype != torch.long:
            raise TypeError("group_idx must be a rank-1 torch.long tensor.")
        if group_idx.numel() and (torch.any(group_idx < 0) or torch.any(group_idx >= self.group_count)):
            raise ValueError("group_idx contains an out-of-range group pose index.")


class TradQuantitativeForward(nn.Module):
    """Evaluate INR at PSF samples, simulate each signal, then average pixels."""

    def __init__(self, quantitative_inr: nn.Module, signal_simulator: nn.Module, rigid_psf: GroupRigidPSF, protocol: object) -> None:
        super().__init__()
        self.quantitative_inr = quantitative_inr
        self.signal_simulator = signal_simulator
        self.rigid_psf = rigid_psf
        self.protocol = protocol

    def forward(
        self,
        local_xyz_mm: torch.Tensor,
        group_idx: torch.Tensor,
        weight_idx: torch.Tensor,
        timing9_ms: torch.Tensor,
        n_psf_samples: int,
    ) -> torch.Tensor:
        """Predict observed-weight pixels without averaging tissue parameters first."""

        batch_size = local_xyz_mm.shape[0]
        if weight_idx.shape != (batch_size,) or weight_idx.dtype != torch.long:
            raise TypeError("weight_idx must be [B] torch.long.")
        if torch.any(weight_idx < 0) or torch.any(weight_idx > 9):
            raise ValueError("weight_idx must contain observed weights 0..9.")
        if timing9_ms.shape != (batch_size, 9):
            raise ValueError("timing9_ms must be [B,9].")
        world_samples = self.rigid_psf.sample_local_then_transform(local_xyz_mm, group_idx, n_psf_samples)
        flat_world = world_samples.reshape(-1, 3)
        fields = self.quantitative_inr(flat_world)
        fingerprint = self.signal_simulator(
            fields["t1_ms"], fields["t2_ms"], fields["b1"],
            timing9_ms[:, None, :].expand(-1, n_psf_samples, -1).reshape(-1, 9),
            self.protocol,
            normalize=True,
        )
        selected = fingerprint.gather(1, weight_idx[:, None, None].expand(-1, n_psf_samples, 1).reshape(-1, 1)).squeeze(1)
        amplitude = fields["amplitude"]
        return (selected * amplitude).reshape(batch_size, n_psf_samples).mean(dim=1)
