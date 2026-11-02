"""Compatibility facade for the decoder-neutral reconstruction engine."""

from reconstruction_core.orchestration import (
    TradTrainingModel,
    _balanced_mse,
    _data_loss,
    _optimizer,
    _quantitative_regularization,
    build_protocol_from_dataset,
    build_training_model as _build_training_model,
    load_checkpoint,
    save_checkpoint,
    train_reconstruction,
)


def build_training_model(*args, **kwargs):
    model, space, protocol, _metadata = _build_training_model(*args, **kwargs)
    return model, space, protocol


def train_trad(*args, **kwargs):
    return train_reconstruction(*args, **kwargs, route="trad_bloch")


__all__ = ["TradTrainingModel", "_balanced_mse", "_data_loss", "_optimizer", "_quantitative_regularization", "build_protocol_from_dataset", "build_training_model", "load_checkpoint", "save_checkpoint", "train_trad"]
