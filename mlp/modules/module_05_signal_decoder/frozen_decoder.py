"""Frozen BatchNorm-safe MLP decoder that preserves gradients to tissue inputs."""

from __future__ import annotations

import torch
from torch import nn

from .mlp_model import MdmSignalMLP, make_input12


class FrozenMLPSignalDecoder(nn.Module):
    """Future online adapter only; it has no `torch.no_grad()` forward path."""

    def __init__(self, mlp: MdmSignalMLP) -> None:
        super().__init__()
        self.mlp = mlp
        for parameter in self.mlp.parameters():
            parameter.requires_grad_(False)
        self.mlp.eval()

    def train(self, mode: bool = True) -> "FrozenMLPSignalDecoder":
        super().train(mode)
        self.mlp.eval()  # BatchNorm statistics must remain frozen under outer model.train().
        return self

    def forward(self, t1_ms: torch.Tensor, t2_ms: torch.Tensor, b1: torch.Tensor, timing9_ms: torch.Tensor) -> torch.Tensor:
        return self.mlp(make_input12(t1_ms, t2_ms, b1, timing9_ms))
