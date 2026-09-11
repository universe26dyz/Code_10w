import torch

from modules.module_07_objective_training.mlp_trainer import build_mlp_training_model
from online_helpers import recon_config, write_functional_checkpoint, write_observations


def test_outer_reconstruction_train_keeps_mlp_batchnorm_eval_and_buffers_constant(tmp_path):
    dataset = write_observations(tmp_path / "observations.npz")
    checkpoint = tmp_path / "checkpoint.pth"; write_functional_checkpoint(checkpoint)
    model, _, _, _ = build_mlp_training_model(dataset, recon_config(checkpoint), "configs/protocol_hhz_v1.yaml", torch.device("cpu"))
    before = [(layer.running_mean.clone(), layer.running_var.clone()) for layer in model.signal_simulator.mlp.modules() if isinstance(layer, torch.nn.BatchNorm1d)]
    model.train()
    after = [(layer.running_mean, layer.running_var) for layer in model.signal_simulator.mlp.modules() if isinstance(layer, torch.nn.BatchNorm1d)]
    assert all(layer.training is False for layer in model.signal_simulator.mlp.modules() if isinstance(layer, torch.nn.BatchNorm1d))
    for previous, current in zip(before, after):
        torch.testing.assert_close(previous[0], current[0]); torch.testing.assert_close(previous[1], current[1])
