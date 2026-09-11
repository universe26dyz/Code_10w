"""mDM-style 12→200→200→200→10 normalized signal simulator."""

from __future__ import annotations

import torch
from torch import nn


class MdmSignalMLP(nn.Module):
    """Minimal 11→12 input adaptation of vendored mDM ``Models/MLP``."""

    input_dim = 12
    output_dim = 10
    width = 200

    def __init__(self) -> None:
        super().__init__()
        self.mlp_block1 = nn.Sequential(nn.Linear(12, 200), nn.BatchNorm1d(200), nn.LeakyReLU())
        self.mlp_block2 = nn.Sequential(nn.Linear(200, 200), nn.BatchNorm1d(200), nn.LeakyReLU())
        self.mlp_block3 = nn.Sequential(nn.Linear(200, 200), nn.BatchNorm1d(200), nn.LeakyReLU())
        self.mlp_block4 = nn.Linear(200, 10)

    def raw_forward(self, input12: torch.Tensor) -> torch.Tensor:
        if input12.shape[-1] != 12:
            raise ValueError(f"MLP input must end in 12 features, got {tuple(input12.shape)}.")
        shape = input12.shape[:-1]
        x = input12.reshape(-1, 12)
        x = self.mlp_block1(x)
        x = self.mlp_block2(x)
        x = self.mlp_block3(x)
        return self.mlp_block4(x).reshape(*shape, 10)

    def forward(self, input12: torch.Tensor) -> torch.Tensor:
        raw = self.raw_forward(input12)
        return raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True).clamp_min(torch.finfo(raw.dtype).eps)


def make_input12(t1_ms: torch.Tensor, t2_ms: torch.Tensor, b1: torch.Tensor, timing9_ms: torch.Tensor) -> torch.Tensor:
    """Build the sole 12D contract: T1/1000, T2/1000, B1, timing9/1000."""

    if t1_ms.shape != t2_ms.shape or t1_ms.shape != b1.shape or timing9_ms.shape[:-1] != t1_ms.shape or timing9_ms.shape[-1] != 9:
        raise ValueError("T1/T2/B1 shapes and timing9 [...,9] must agree.")
    return torch.cat((t1_ms[..., None] / 1000.0, t2_ms[..., None] / 1000.0, b1[..., None], timing9_ms / 1000.0), dim=-1)
