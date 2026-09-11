"""Held-out signal and input-gradient fidelity metrics for the offline surrogate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from .mlp_model import MdmSignalMLP
from .synthetic_dataset import _protocol
from .trad_teacher.trad_signal_simulator import TradSignalSimulator


def _inputs_targets(path: str | Path, limit: int) -> tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(path, "r") as handle:
        n = min(limit, handle["input12"].shape[0])
        return torch.from_numpy(np.asarray(handle["input12"][:n], dtype=np.float32)), torch.from_numpy(np.asarray(handle["target_signal10"][:n], dtype=np.float32))


def fidelity_metrics(model: MdmSignalMLP, test_h5: str | Path, pool: dict[str, np.ndarray], batch_size: int, gradient_samples: int) -> dict[str, Any]:
    if batch_size < 1 or gradient_samples < 1:
        raise ValueError("batch_size and gradient_samples must be positive.")
    inputs, target = _inputs_targets(test_h5, 1_000_000_000)
    device = next(model.parameters()).device
    inputs, target = inputs.to(device), target.to(device)
    was_training = model.training; model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            predictions.append(model(inputs[start:start + batch_size]))
    prediction = torch.cat(predictions)
    error = prediction - target
    per_weight_rmse = error.pow(2).mean(0).sqrt()
    metrics: dict[str, Any] = {"overall_rmse": float(error.pow(2).mean().sqrt()), "mae": float(error.abs().mean()), "max_abs_error": float(error.abs().max()), "per_weight_rmse": per_weight_rmse.tolist()}
    n = min(gradient_samples, inputs.shape[0])
    x = inputs[:n].clone().requires_grad_(True)
    predicted = model(x)
    mlp_grads = []
    for weight in range(10):
        gradient = torch.autograd.grad(predicted[:, weight].sum(), x, retain_graph=True)[0][:, :3]
        mlp_grads.append(torch.stack((gradient[:, 0] / 1000.0, gradient[:, 1] / 1000.0, gradient[:, 2]), dim=-1))
    t1 = (inputs[:n, 0] * 1000.0).detach().clone().requires_grad_(True)
    t2 = (inputs[:n, 1] * 1000.0).detach().clone().requires_grad_(True)
    b1 = inputs[:n, 2].detach().clone().requires_grad_(True)
    timing = inputs[:n, 3:] * 1000.0
    teacher_output = TradSignalSimulator()(t1, t2, b1, timing, _protocol(pool, pool["protocol_path"]), normalize=True)
    teacher_grads = []
    for weight in range(10):
        gradients = torch.autograd.grad(teacher_output[:, weight].sum(), (t1, t2, b1), retain_graph=True)
        teacher_grads.append(torch.stack(gradients, dim=-1))
    mlp_gradient = torch.stack(mlp_grads, dim=1)
    teacher_gradient = torch.stack(teacher_grads, dim=1)
    absolute = (mlp_gradient - teacher_gradient).abs()
    mask = teacher_gradient.abs() > 1e-8
    cosine = torch.nn.functional.cosine_similarity(mlp_gradient.reshape(n, -1), teacher_gradient.reshape(n, -1), dim=1)
    metrics["gradient"] = {"absolute_error_mean": float(absolute.mean()), "relative_error_mean": float((absolute[mask] / teacher_gradient.abs()[mask]).mean()) if mask.any() else None, "cosine_similarity_mean": float(cosine.mean())}
    if was_training: model.train()
    return metrics
