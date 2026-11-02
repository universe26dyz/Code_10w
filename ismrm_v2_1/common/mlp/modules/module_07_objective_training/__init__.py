"""MLP compatibility exports without eager legacy-runtime imports."""


def __getattr__(name: str):
    if name in {"MLPTrainingModel", "train_mlp_reconstruction"}:
        from . import mlp_trainer
        return getattr(mlp_trainer, name)
    if name == "TrainingSpace":
        from trad.modules.module_07_objective_training.training_space import TrainingSpace
        return TrainingSpace
    raise AttributeError(name)


__all__ = ["TrainingSpace", "MLPTrainingModel", "train_mlp_reconstruction"]
