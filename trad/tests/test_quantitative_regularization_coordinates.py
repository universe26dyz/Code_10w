import copy

import torch

from modules.module_07_objective_training.trad_trainer import (
    _quantitative_regularization,
    _regularization_world_points,
    build_training_model,
    train_trad,
)
from tests.test_end_to_end_tiny import _config, _dataset


def test_regularization_uses_detached_current_training_world_coordinates(tmp_path):
    dataset = _dataset(tmp_path / "observations.npz")
    config = _config()
    config["loss"]["quantitative"] = {"t1": 1.0, "t2": 1.0, "b1": 1.0}
    model, space, _ = build_training_model(dataset, config, "configs/protocol_hhz_v1.yaml", torch.device("cpu"))
    local_train = space.local_to_train(dataset.xyz[:10])
    world_train = _regularization_world_points(model, local_train, dataset.group_idx[:10])
    assert not torch.allclose(world_train, local_train)
    assert world_train.requires_grad is False
    regularization = _quantitative_regularization(model, world_train, space.spatial_scaling, config["loss"]["quantitative"])
    sum(regularization.values()).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.inr.parameters())
    assert model.rigid_psf.axisangle.grad is None


def test_stage_a_nonzero_field_regularization_keeps_group_pose_at_initial(tmp_path):
    dataset = _dataset(tmp_path / "observations.npz")
    config = copy.deepcopy(_config())
    config["training"]["stage_a_iterations"] = 1
    config["training"]["stage_b_iterations"] = 0
    config["loss"]["quantitative"] = {"t1": 1e-4, "t2": 1e-4, "b1": 1e-4}
    result = train_trad(dataset, config, "configs/protocol_hhz_v1.yaml", tmp_path / "outputs")
    torch.testing.assert_close(result["stage_a_axisangle_final"], result["model"].rigid_psf.axisangle_init)
