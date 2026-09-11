"""Training-space adaptation, v1 loss, staged optimization, and checkpoints."""

from .training_space import TrainingSpace
from .trad_trainer import TradTrainingModel, build_protocol_from_dataset, train_trad

__all__ = ["TrainingSpace", "TradTrainingModel", "build_protocol_from_dataset", "train_trad"]
