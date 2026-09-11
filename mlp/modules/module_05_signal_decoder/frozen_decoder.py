"""Frozen BatchNorm-safe MLP decoder that preserves gradients to tissue inputs."""

from __future__ import annotations

import torch
from torch import nn

from .mlp_model import MdmSignalMLP, make_input12
from .trad_teacher.trad_signal_simulator import TradProtocol


class FrozenMLPSignalDecoder(nn.Module):
    """Future online adapter only; it has no `torch.no_grad()` forward path."""

    def __init__(self, mlp: MdmSignalMLP, protocol: TradProtocol | None = None) -> None:
        super().__init__()
        self.mlp = mlp
        self.protocol = protocol
        for parameter in self.mlp.parameters():
            parameter.requires_grad_(False)
        self.mlp.eval()

    def train(self, mode: bool = True) -> "FrozenMLPSignalDecoder":
        super().train(mode)
        self.mlp.eval()  # BatchNorm statistics must remain frozen under outer model.train().
        return self

    def forward(self, t1_ms: torch.Tensor, t2_ms: torch.Tensor, b1: torch.Tensor, timing9_ms: torch.Tensor, protocol: TradProtocol | None = None, normalize: bool = True) -> torch.Tensor:
        if normalize is not True:
            raise ValueError("FrozenMLPSignalDecoder only provides L2-normalized HHZ fingerprints.")
        if protocol is not None and self.protocol is not None and protocol != self.protocol:
            raise ValueError("FrozenMLPSignalDecoder received a protocol different from its validated checkpoint protocol.")
        return self.mlp(make_input12(t1_ms, t2_ms, b1, timing9_ms))
