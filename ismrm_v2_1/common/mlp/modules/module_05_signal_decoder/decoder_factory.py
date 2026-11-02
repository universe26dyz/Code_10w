"""Frozen-MLP factory for the shared decoder-neutral reconstruction engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .checkpoint_loader import load_frozen_mlp_decoder
from .provenance import sha256_file


def frozen_mlp_decoder_factory(dataset: Any, config: Mapping[str, Any], protocol: object, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    decoder_config = config.get("decoder")
    if not isinstance(decoder_config, Mapping):
        raise ValueError("decoder must be a mapping for FrozenMLP reconstruction.")
    checkpoint = decoder_config.get("checkpoint")
    allowance = decoder_config.get("allow_functional_fixture_checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("decoder.checkpoint must name a validated MLP checkpoint.")
    if not isinstance(allowance, bool):
        raise ValueError("decoder.allow_functional_fixture_checkpoint must be an explicit boolean.")
    protocol_path = decoder_config.get("protocol_yaml", "trad/configs/protocol_hhz_v1.yaml")
    decoder = load_frozen_mlp_decoder(checkpoint, dataset, protocol_path, allow_functional_fixture=allowance, device=device)
    metadata = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    scientific_checkpoint = (
        not bool(metadata.get("functional_fixture"))
        and bool(metadata.get("formal_candidate"))
        and metadata.get("dataset_schema") == "mlp_rr_synthetic/v1"
        and metadata.get("dataset_split_mode") == "rhythm"
        and metadata.get("validation_status") == "approved_by_manual_review"
    )
    source_metadata = dict(metadata)
    source_metadata["scientific_checkpoint"] = scientific_checkpoint
    return decoder, {
        "decoder_type": "FrozenMLP", "decoder_source": str(Path(checkpoint)),
        "source_mlp_checkpoint_path": str(checkpoint), "source_mlp_checkpoint_sha256": sha256_file(checkpoint),
        "source_mlp_checkpoint_metadata": source_metadata, "scientific_checkpoint": scientific_checkpoint,
    }
