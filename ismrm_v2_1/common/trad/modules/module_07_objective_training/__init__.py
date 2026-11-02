"""Training-space adaptation, v1 loss, staged optimization, and checkpoints."""

from .training_space import TrainingSpace


def __getattr__(name: str):
    if name in {"TradTrainingModel", "build_protocol_from_dataset", "train_trad"}:
        from . import trad_trainer
        return getattr(trad_trainer, name)
    raise AttributeError(name)

__all__ = ["TrainingSpace", "TradTrainingModel", "build_protocol_from_dataset", "train_trad"]
