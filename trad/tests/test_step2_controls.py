import copy
import json

import torch

from modules.module_07_objective_training.trad_trainer import (
    _balanced_mse,
    _data_loss,
    _quantitative_regularization,
    build_training_model,
    train_trad,
)
from tests.test_end_to_end_tiny import _config, _dataset


def test_regularization_modes_and_detached_amplitude_guidance_are_finite(tmp_path):
    dataset = _dataset(tmp_path / "observations.npz")
    config = _config(); config["loss"]["quantitative"] = {"t1": 1.0, "t2": 1.0, "b1": 1.0}
    model, space, _ = build_training_model(dataset, config, "configs/protocol_hhz_v1.yaml", torch.device("cpu"))
    points = space.local_to_train(dataset.xyz[:10])
    for mode in ("none", "TV", "L2", "edge-preserving"):
        settings = {"mode": mode, "n_points": 7, "edge_epsilon": 1e-3, "amplitude_guidance": {"enabled": True, "alpha": 1.0}}
        values = _quantitative_regularization(model, points, space.spatial_scaling, config["loss"]["quantitative"], settings)
        assert all(torch.isfinite(value) for value in values.values())
        if mode == "none":
            assert all(value.item() == 0 for value in values.values())
        else:
            sum(values.values()).backward()
            model.zero_grad(set_to_none=True)


def test_variance_off_is_exact_balanced_mse_and_enabled_branch_is_trainable(tmp_path):
    dataset = _dataset(tmp_path / "observations.npz")
    config = _config()
    model, space, _ = build_training_model(dataset, config, "configs/protocol_hhz_v1.yaml", torch.device("cpu"))
    indices = torch.cat([torch.nonzero(dataset.weight_idx == weight, as_tuple=False)[0] for weight in range(10)])
    batch = {"xyz": space.local_to_train(dataset.xyz[indices]), "group_idx": dataset.group_idx[indices], "weight_idx": dataset.weight_idx[indices]}
    prediction, observed, weights = torch.arange(10, dtype=torch.float32), torch.zeros(10), torch.ones(10)
    torch.testing.assert_close(_data_loss(model, prediction, observed, batch, weights), _balanced_mse(prediction, observed, batch["weight_idx"], weights))
    config["variance"] = {"enabled": True, "pixel": True, "slice": True}
    model, _, _ = build_training_model(dataset, config, "configs/protocol_hhz_v1.yaml", torch.device("cpu"))
    loss = _data_loss(model, prediction, observed, batch, weights)
    loss.backward()
    assert model.variance_head is not None and all(parameter.grad is not None for parameter in model.variance_head.parameters())


def test_stack_initialization_is_saved_before_frozen_stage_a(tmp_path):
    source = tmp_path / "observations.npz"; dataset = _dataset(source)
    config = copy.deepcopy(_config())
    config["training"]["stage_a_iterations"] = config["training"]["stage_b_iterations"] = 0
    config["stack_initialization"] = {"enabled": True, "args_registration": {"num_levels": 1, "num_steps": 1, "max_iter": 1}}
    output = tmp_path / "outputs"
    result = train_trad(dataset, config, "configs/protocol_hhz_v1.yaml", output, prepared_inputs=[source])
    assert result["stack_initialization"] is not None
    assert torch.equal(result["stage_a_axisangle_final"], result["model"].rigid_psf.axisangle_init)
    payload = json.loads((output / "stack_initialization_poses.json").read_text())
    assert len(payload["stacks"]) == 1
