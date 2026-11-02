"""Decoder factories for controlled quantitative reconstruction experiments."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import torch
from torch import nn

from .trad_signal_simulator import TradProtocol, TradSignalSimulator


DecoderFactory = Callable[[Any, Mapping[str, Any], TradProtocol, torch.device], tuple[nn.Module, dict[str, str]]]


def build_bloch_decoder(*_args: Any, **_kwargs: Any) -> TradSignalSimulator:
    """Construct the direct HHZ decoder used by the baseline route."""

    return TradSignalSimulator()


def bloch_decoder_factory(dataset: Any, config: Mapping[str, Any], protocol: TradProtocol, device: torch.device) -> tuple[nn.Module, dict[str, str]]:
    return build_bloch_decoder().to(device), {"decoder_type": "Bloch", "decoder_source": "TradSignalSimulator"}
