"""MLP compatibility facade over the sole reconstruction engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from reconstruction_core.orchestration import (
    ReconstructionTrainingModel,
    _quantitative_regularization,
    build_training_model,
    load_checkpoint,
    train_reconstruction,
)
from mlp.modules.module_05_signal_decoder.decoder_factory import frozen_mlp_decoder_factory


MLPTrainingModel = ReconstructionTrainingModel


def _with_protocol_path(config: Mapping[str, Any], protocol_yaml: str | Path) -> dict[str, Any]:
    resolved = dict(config)
    training = dict(resolved.get("training", {}))
    # This is the already-established MLP trimmed-mean normalization, made
    # explicit so the shared engine enforces the identical Trad contract.
    training.setdefault("intensity_normalization", {"enabled": True, "method": "trimmed_mean", "lower_quantile": 0.1, "upper_quantile": 0.9})
    resolved["training"] = training
    decoder = dict(resolved.get("decoder", {}))
    decoder["protocol_yaml"] = str(protocol_yaml)
    resolved["decoder"] = decoder
    return resolved


def build_mlp_training_model(dataset: Any, config: Mapping[str, Any], protocol_yaml: str | Path, device: torch.device, *, initial_group_axisangle_physical: torch.Tensor | None = None):
    model, space, protocol, metadata = build_training_model(
        dataset, _with_protocol_path(config, protocol_yaml), protocol_yaml, device,
        initial_group_axisangle_physical=initial_group_axisangle_physical,
        decoder_factory=frozen_mlp_decoder_factory,
    )
    return model, space, protocol, metadata.get("source_mlp_checkpoint_metadata", {})


def load_mlp_reconstruction_checkpoint(path: str | Path, model: MLPTrainingModel, device: torch.device) -> dict[str, Any]:
    checkpoint = load_checkpoint(path, model, device)
    required = {"source_mlp_checkpoint_sha256", "source_mlp_checkpoint_metadata", "scientific_checkpoint"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"MLP reconstruction checkpoint lacks required fields: {sorted(missing)}")
    return checkpoint


def train_mlp_reconstruction(dataset: Any, config: Mapping[str, Any], protocol_yaml: str | Path, output_dir: str | Path, **kwargs: Any) -> dict[str, Any]:
    return train_reconstruction(
        dataset, _with_protocol_path(config, protocol_yaml), protocol_yaml, output_dir,
        decoder_factory=frozen_mlp_decoder_factory, route="mlp_surrogate_v1", **kwargs,
    )


__all__ = ["MLPTrainingModel", "_quantitative_regularization", "build_mlp_training_model", "load_mlp_reconstruction_checkpoint", "train_mlp_reconstruction"]
