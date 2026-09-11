"""Four-parameter quantitative INR built from NeSVoR's hash-grid primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from third_party.nesvor.nesvor.inr import models as nesvor_models


@dataclass(frozen=True)
class QuantitativeINRConfig:
    """Explicit CPU-compatible NeSVoR HashGrid and MLP dimensions."""

    coarsest_resolution_mm: float = 8.0
    finest_resolution_mm: float = 2.0
    level_scale: float = 1.5
    n_features_per_level: int = 2
    log2_hashmap_size: int = 12
    latent_features: int = 16
    width: int = 32
    depth: int = 2


class QuantitativeINR(nn.Module):
    """Map RAS-mm coordinates to continuous ``T1, T2, B1, A`` fields.

    The encoding and both networks are constructed only with NeSVoR's vendored
    ``build_encoding``/``build_network`` helpers.  The shared network produces
    a latent vector, then the parameter head applies the v1 physical ranges.
    """

    def __init__(
        self,
        bounding_box_ras_mm: torch.Tensor,
        config: QuantitativeINRConfig = QuantitativeINRConfig(),
        spatial_scaling: float = 1.0,
    ) -> None:
        super().__init__()
        bbox = torch.as_tensor(bounding_box_ras_mm, dtype=torch.float32)
        if bbox.shape != (2, 3) or not torch.isfinite(bbox).all() or not torch.all(bbox[1] > bbox[0]):
            raise ValueError("bounding_box_ras_mm must be finite [2,3] with positive extent.")
        if config.coarsest_resolution_mm <= 0 or config.finest_resolution_mm <= 0:
            raise ValueError("HashGrid resolutions must be positive.")
        if not torch.isfinite(torch.tensor(spatial_scaling)) or spatial_scaling <= 0:
            raise ValueError("spatial_scaling must be positive and finite.")
        self.config = config
        self.spatial_scaling = float(spatial_scaling)
        self.register_buffer("bounding_box_ras_mm", bbox)
        # Match NeSVoR.NeSVoR: local CPU smoke explicitly selects its vendored
        # PyTorch HashGrid; CUDA retains NeSVoR's tinycudann decision.
        if bbox.device.type == "cpu":
            nesvor_models.USE_TORCH = True
        base_resolution, n_levels = nesvor_models.compute_resolution_nlevel(
            bbox,
            config.coarsest_resolution_mm,
            config.finest_resolution_mm,
            config.level_scale,
            self.spatial_scaling,
        )
        if base_resolution < 1 or n_levels < 1:
            raise ValueError("NeSVoR HashGrid resolution calculation produced an invalid level count.")
        self.encoding = nesvor_models.build_encoding(
            n_input_dims=3,
            otype="HashGrid",
            n_levels=n_levels,
            n_features_per_level=config.n_features_per_level,
            log2_hashmap_size=config.log2_hashmap_size,
            base_resolution=base_resolution,
            per_level_scale=config.level_scale,
            dtype=torch.float32,
        )
        encoded_features = n_levels * config.n_features_per_level
        self.shared_latent = nesvor_models.build_network(
            n_input_dims=encoded_features,
            n_output_dims=config.latent_features,
            activation="ReLU",
            output_activation="None",
            n_neurons=config.width,
            n_hidden_layers=config.depth,
            dtype=torch.float32,
        )
        self.parameter_head = nesvor_models.build_network(
            n_input_dims=config.latent_features,
            n_output_dims=4,
            activation="ReLU",
            output_activation="None",
            n_neurons=config.width,
            n_hidden_layers=1,
            dtype=torch.float32,
        )

    def forward(self, xyz_ras_mm: torch.Tensor) -> dict[str, torch.Tensor]:
        """Evaluate physical fields at RAS-mm points ending in coordinate dimension 3."""

        if xyz_ras_mm.shape[-1] != 3:
            raise ValueError(f"xyz_ras_mm must end in 3 coordinates, got {tuple(xyz_ras_mm.shape)}.")
        if not torch.is_floating_point(xyz_ras_mm):
            raise TypeError("xyz_ras_mm must be a floating tensor.")
        flat_xyz = xyz_ras_mm.reshape(-1, 3)
        normalized = (flat_xyz - self.bounding_box_ras_mm[0]) / (
            self.bounding_box_ras_mm[1] - self.bounding_box_ras_mm[0]
        )
        encoded = self.encoding(normalized)
        latent = self.shared_latent(encoded)
        raw = self.parameter_head(latent)
        shape = xyz_ras_mm.shape[:-1]
        t2_ms = (5.0 + 195.0 * torch.sigmoid(raw[..., 0])).reshape(shape)
        t1_ms = (t2_ms.reshape(-1) + (2500.0 - t2_ms.reshape(-1)) * torch.sigmoid(raw[..., 1])).reshape(shape)
        b1 = (0.1 + 1.1 * torch.sigmoid(raw[..., 2])).reshape(shape)
        amplitude = (F.softplus(raw[..., 3]) + torch.finfo(raw.dtype).eps).reshape(shape)
        return {"t1_ms": t1_ms, "t2_ms": t2_ms, "b1": b1, "amplitude": amplitude}
