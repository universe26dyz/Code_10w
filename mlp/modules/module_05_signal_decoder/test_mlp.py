"""Held-out signal and input-gradient fidelity metrics for the offline surrogate."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import TensorDataset

from .mlp_model import MdmSignalMLP
from .synthetic_dataset import _protocol
from .trad_teacher.trad_signal_simulator import TradSignalSimulator


def fidelity_metrics(model: MdmSignalMLP, test_data: TensorDataset, pool: dict[str, Any], protocol_path: str, batch_size: int, gradient_samples: int) -> dict[str, Any]:
    if batch_size < 1 or gradient_samples < 1:
        raise ValueError("batch_size and gradient_samples must be positive.")
    inputs, target = test_data.tensors
    device = next(model.parameters()).device
    was_training = model.training; model.eval()
    squared_sum = torch.zeros((), device=device); absolute_sum = torch.zeros((), device=device)
    maximum = torch.zeros((), device=device); per_weight_squared = torch.zeros(10, device=device)
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            error = model(inputs[start:start + batch_size].to(device)) - target[start:start + batch_size].to(device)
            squared_sum += error.pow(2).sum(); absolute_sum += error.abs().sum()
            maximum = torch.maximum(maximum, error.abs().max()); per_weight_squared += error.pow(2).sum(0)
    count = inputs.shape[0]
    metrics: dict[str, Any] = {"overall_rmse": float((squared_sum / (count * 10)).sqrt()), "mae": float(absolute_sum / (count * 10)), "max_abs_error": float(maximum), "per_weight_rmse": (per_weight_squared / count).sqrt().detach().cpu().tolist()}
    n = min(gradient_samples, inputs.shape[0])
    x = inputs[:n].to(device).clone().requires_grad_(True)
    predicted = model(x)
    mlp_grads = []
    for weight in range(10):
        gradient = torch.autograd.grad(predicted[:, weight].sum(), x, retain_graph=True)[0][:, :3]
        mlp_grads.append(torch.stack((gradient[:, 0] / 1000.0, gradient[:, 1] / 1000.0, gradient[:, 2]), dim=-1))
    t1 = (inputs[:n, 0].to(device) * 1000.0).detach().clone().requires_grad_(True)
    t2 = (inputs[:n, 1].to(device) * 1000.0).detach().clone().requires_grad_(True)
    b1 = inputs[:n, 2].to(device).detach().clone().requires_grad_(True)
    timing = inputs[:n, 3:].to(device) * 1000.0
    teacher_output = TradSignalSimulator()(t1, t2, b1, timing, _protocol(pool, protocol_path), normalize=True)
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
