"""Minimal staged Trad optimization modeled on vendored ``nesvor.inr.train``."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
import yaml

from modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset, robust_trimmed_mean_intensity
from modules.module_04_quantitative_inr.quantitative_inr import QuantitativeINR, QuantitativeINRConfig
from modules.module_05_signal_decoder.trad_signal_simulator import TradProtocol, TradSignalSimulator
from modules.module_06_rigid_psf.rigid_psf_forward import GroupRigidPSF, TradQuantitativeForward
from .training_space import TrainingSpace


def _require(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Training config lacks required key: {key}")
    return mapping[key]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required YAML does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML must contain one mapping: {path}")
    return value


def build_protocol_from_dataset(dataset: QuantPointDataset, protocol_yaml: str | Path) -> TradProtocol:
    """Construct the only online Trad protocol from validated DICOM and HHZ YAML."""

    protocol = _load_yaml(protocol_yaml)
    tr_ms, vps = dataset.validated_tr_vps()
    dicom_fields = _require(protocol, "dicom_fields")
    if not isinstance(dicom_fields, dict) or dicom_fields.get("tr_ms") != "RepetitionTime" or dicom_fields.get("vps") != "EchoTrainLength":
        raise ValueError("protocol_hhz_v1.yaml must declare DICOM RepetitionTime and EchoTrainLength provenance.")
    result = TradProtocol(
        tr_ms=tr_ms,
        vps=vps,
        fa_deg=tuple(float(x) for x in _require(protocol, "flip_angle_degrees")),
        ti_ms=tuple(float(x) for x in _require(protocol, "inversion_times_ms")),
        t2prep_ms=tuple(float(x) for x in _require(protocol, "t2prep_ms")),
        n_ramp_up=int(_require(protocol, "ramp_up_pulses")),
    )
    result.validate()
    return result


class TradTrainingModel(nn.Module):
    """Shared INR + group-rigid state with Trad's direct signal forward only."""

    def __init__(self, inr: QuantitativeINR, rigid_psf: GroupRigidPSF, protocol: TradProtocol) -> None:
        super().__init__()
        self.inr = inr
        self.rigid_psf = rigid_psf
        self.signal_simulator = TradSignalSimulator()
        self.forward_model = TradQuantitativeForward(inr, self.signal_simulator, rigid_psf, protocol)
        self.protocol = protocol
        self.register_buffer("intensity_scale", torch.ones(()))

    def forward(self, batch: Mapping[str, torch.Tensor], n_psf_samples: int) -> torch.Tensor:
        return self.forward_model(batch["xyz"], batch["group_idx"], batch["weight_idx"], batch["timing"], n_psf_samples)


def build_training_model(
    dataset: QuantPointDataset, config: Mapping[str, Any], protocol_yaml: str | Path, device: torch.device
) -> tuple[TradTrainingModel, TrainingSpace, TradProtocol]:
    inr_cfg = _require(config, "inr")
    training_cfg = _require(config, "training")
    if not isinstance(inr_cfg, dict) or not isinstance(training_cfg, dict):
        raise ValueError("inr and training must be YAML mappings.")
    required_inr = ("coarsest_resolution_mm", "finest_resolution_mm", "level_scale", "n_features_per_level", "log2_hashmap_size", "latent_features", "width", "depth")
    missing = [key for key in required_inr if key not in inr_cfg]
    if missing:
        raise ValueError(f"inr config lacks explicit values: {missing}")
    space = TrainingSpace.from_dataset(dataset, float(_require(training_cfg, "spatial_scaling")))
    protocol = build_protocol_from_dataset(dataset, protocol_yaml)
    inr = QuantitativeINR(
        space.bbox_train.to(device),
        QuantitativeINRConfig(**{key: inr_cfg[key] for key in required_inr}),
        spatial_scaling=space.spatial_scaling,
    ).to(device)
    rigid = GroupRigidPSF(
        space.group_axisangle_init_train.to(device), space.group_resolution_train.to(device)
    ).to(device)
    return TradTrainingModel(inr, rigid, protocol).to(device), space, protocol


def _balanced_batch(dataset: QuantPointDataset, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    if batch_size < 10 or batch_size % 10:
        raise ValueError("batch_size must be a positive multiple of 10 for explicit 10-weight balance.")
    indices: list[torch.Tensor] = []
    per_weight = batch_size // 10
    for weight in range(10):
        candidates = torch.nonzero(dataset.weight_idx == weight, as_tuple=False).flatten()
        if candidates.numel() == 0:
            raise ValueError(f"No valid samples exist for weight {weight}.")
        indices.append(candidates[torch.randint(candidates.numel(), (per_weight,), device=device)])
    index = torch.cat(indices)
    permutation = torch.randperm(index.numel(), device=device)
    index = index[permutation]
    return {
        "xyz": dataset.xyz[index] / 1.0, "v": dataset.v[index], "group_idx": dataset.group_idx[index],
        "weight_idx": dataset.weight_idx[index], "stack_idx": dataset.stack_idx[index], "timing": dataset.timing[index],
    }


def _stack_weight_tensor(stack_idx: torch.Tensor, weights: Mapping[int, float]) -> torch.Tensor:
    values = torch.empty_like(stack_idx, dtype=torch.float32)
    for stack, weight in weights.items():
        values[stack_idx == stack] = weight
    return values


def _balanced_mse(prediction: torch.Tensor, observed: torch.Tensor, weight_idx: torch.Tensor, stack_weights: torch.Tensor) -> torch.Tensor:
    weighted = (prediction - observed).pow(2) * stack_weights
    means = [weighted[weight_idx == weight].mean() for weight in range(10)]
    return torch.stack(means).mean()


def _intensity_normalization_provenance(training: Mapping[str, Any], dataset: QuantPointDataset) -> dict[str, object]:
    settings = _require(training, "intensity_normalization")
    if not isinstance(settings, Mapping):
        raise ValueError("training.intensity_normalization must be a mapping.")
    required = ("enabled", "method", "lower_quantile", "upper_quantile")
    missing = [key for key in required if key not in settings]
    if missing:
        raise ValueError(f"training.intensity_normalization lacks explicit values: {missing}")
    if settings["enabled"] is not True or settings["method"] != "trimmed_mean":
        raise ValueError("Trad v1 requires enabled=true and method=trimmed_mean intensity normalization.")
    lower, upper = float(settings["lower_quantile"]), float(settings["upper_quantile"])
    if lower != 0.1 or upper != 0.9:
        raise ValueError("Trad v1 requires intensity normalization lower_quantile=0.1 and upper_quantile=0.9.")
    scale = robust_trimmed_mean_intensity(dataset.v.detach(), lower_quantile=lower, upper_quantile=upper)
    return {
        "enabled": True,
        "method": "trimmed_mean",
        "lower_quantile": lower,
        "upper_quantile": upper,
        "scale": float(scale.detach().cpu()),
        "training_units": "normalized_subject_intensity",
        "amplitude_export_units": "original_input_intensity",
    }


def _assert_finite_gradients(model: nn.Module, stage: str, global_iteration: int) -> None:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"Non-finite gradient at stage {stage}, iteration {global_iteration}, parameter {name}.")


def _assert_finite_parameters(model: nn.Module, stage: str, global_iteration: int) -> None:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and not torch.isfinite(parameter).all():
            raise FloatingPointError(f"Non-finite parameter at stage {stage}, iteration {global_iteration}, parameter {name}.")


def _quantitative_regularization(model: TradTrainingModel, train_xyz: torch.Tensor, spatial_scaling: float, weights: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    requested = {"t1": float(_require(weights, "t1")), "t2": float(_require(weights, "t2")), "b1": float(_require(weights, "b1"))}
    if not any(requested.values()):
        zero = torch.zeros((), dtype=train_xyz.dtype, device=train_xyz.device)
        return {key: zero for key in requested}
    points = train_xyz[: min(4, train_xyz.shape[0])].detach().clone().requires_grad_(True)
    fields = model.inr(points)
    normalized = {
        "t1": fields["t1_ms"] / 2500.0,
        "t2": (fields["t2_ms"] - 5.0) / 195.0,
        "b1": (fields["b1"] - 0.1) / 1.1,
    }
    for key, value in normalized.items():
        gradient = torch.autograd.grad(value.sum(), points, create_graph=True)[0] / spatial_scaling
        values[key] = torch.sqrt(gradient.pow(2).sum(dim=-1) + 1e-12).mean()
    return values


def _regularization_world_points(model: TradTrainingModel, local_train: torch.Tensor, group_idx: torch.Tensor) -> torch.Tensor:
    """Use current rigid training-world points, without regularization driving pose."""

    return model.rigid_psf.transform_local_to_ras(local_train, group_idx).detach()


def _optimizer(model: TradTrainingModel, learning_rates: Mapping[str, Any], joint: bool) -> AdamW:
    encoding, networks = [], []
    for name, parameter in model.inr.named_parameters():
        (encoding if name.startswith("encoding") else networks).append(parameter)
    groups: list[dict[str, Any]] = [
        {"name": "encoding", "params": encoding, "lr": float(_require(learning_rates, "encoding"))},
        {"name": "network", "params": networks, "lr": float(_require(learning_rates, "network")), "weight_decay": float(_require(learning_rates, "weight_decay"))},
    ]
    if joint:
        groups.append({"name": "rigid", "params": [model.rigid_psf.axisangle], "lr": float(_require(learning_rates, "rigid"))})
    return AdamW(groups, betas=(0.9, 0.99), eps=1e-15)


def _write_resolved_config(path: Path, config: Mapping[str, Any], protocol: TradProtocol, dataset: QuantPointDataset, space: TrainingSpace, stack_weights: Mapping[int, float], intensity_normalization: Mapping[str, object]) -> None:
    resolved = dict(config)
    resolved["training"] = dict(_require(config, "training"))
    resolved["training"]["intensity_normalization"] = dict(intensity_normalization)
    resolved.update({
        "validated_protocol": {"tr_ms": protocol.tr_ms, "vps": protocol.vps, "fa_deg": list(protocol.fa_deg), "ti_ms": list(protocol.ti_ms), "t2prep_ms": list(protocol.t2prep_ms), "n_ramp_up": protocol.n_ramp_up}, "stack_weights": {int(key): float(value) for key, value in stack_weights.items()},
        "training_space": {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in space.state_dict().items()},
        "group_tr_ms": dataset.group_tr_ms.detach().cpu().tolist(), "group_vps": dataset.group_vps.detach().cpu().tolist(),
    })
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, allow_unicode=True, sort_keys=False)


def save_checkpoint(path: str | Path, model: TradTrainingModel, config: Mapping[str, Any], protocol: TradProtocol, dataset: QuantPointDataset, space: TrainingSpace, seed: int, intensity_normalization: Mapping[str, object]) -> None:
    torch.save({
        "model_state": model.state_dict(), "resolved_config": dict(config), "protocol_hhz_v1": asdict(protocol),
        "validated_tr_ms": protocol.tr_ms, "validated_vps": protocol.vps, "training_space": space.state_dict(),
        "group_resolution_xyz_mm": dataset.group_resolution_xyz_mm.detach().cpu(),
        "group_axisangle_init_physical": space.group_axisangle_init_physical.detach().cpu(),
        "group_axisangle_init_train": space.group_axisangle_init_train.detach().cpu(),
        "trained_axisangle_train": model.rigid_psf.axisangle.detach().cpu(), "random_seed": int(seed),
        "intensity_normalization": dict(intensity_normalization),
    }, Path(path))


def load_checkpoint(path: str | Path, model: TradTrainingModel, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    required = {"model_state", "protocol_hhz_v1", "training_space", "trained_axisangle_train", "intensity_normalization"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint lacks required fields: {sorted(missing)}")
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def train_trad(dataset: QuantPointDataset, config: Mapping[str, Any], protocol_yaml: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Run Stage A then Stage B continuously on one model and write requested artifacts."""

    training = _require(config, "training")
    loss_cfg = _require(config, "loss")
    if not isinstance(training, dict) or not isinstance(loss_cfg, dict):
        raise ValueError("training and loss must be mappings.")
    required_training = ("device", "seed", "batch_size", "psf_samples", "stage_a_iterations", "stage_b_iterations", "spatial_scaling", "learning_rates", "scheduler_milestones", "scheduler_gamma")
    missing = [key for key in required_training if key not in training]
    if missing:
        raise ValueError(f"training config lacks explicit values: {missing}")
    device = torch.device(training["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device is unavailable: {device}")
    seed = int(training["seed"])
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    stack_weights = dataset.validate_balanced_samples()
    intensity_normalization = _intensity_normalization_provenance(training, dataset)
    model, space, protocol = build_training_model(dataset, config, protocol_yaml, device)
    model.intensity_scale.copy_(torch.as_tensor(intensity_normalization["scale"], dtype=model.intensity_scale.dtype, device=device))
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output_dir must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _write_resolved_config(output / "config_resolved.yaml", config, protocol, dataset, space, stack_weights, intensity_normalization)
    log_path = output / "training_log.csv"
    global_iter = 0
    stage_a_axisangle_final: torch.Tensor | None = None
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "iteration", "data_mse", "reg_t1", "reg_t2", "reg_b1", "transformation", "total", "intensity_scale", "lr_encoding", "lr_network", "lr_rigid"])
        writer.writeheader()
        for stage, iterations, joint in (("A", int(training["stage_a_iterations"]), False), ("B", int(training["stage_b_iterations"]), True)):
            if iterations < 0:
                raise ValueError(f"Stage {stage} iterations must be non-negative.")
            model.rigid_psf.axisangle.requires_grad_(joint)
            optimizer = _optimizer(model, _require(training, "learning_rates"), joint)
            milestones = [int(float(value) * iterations) for value in training["scheduler_milestones"] if 0 < float(value) < 1]
            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=float(training["scheduler_gamma"]))
            for _ in range(iterations):
                global_iter += 1
                batch = _balanced_batch(dataset, int(training["batch_size"]), device)
                batch["xyz"] = space.local_to_train(batch["xyz"])
                prediction = model(batch, int(training["psf_samples"]))
                data_mse = _balanced_mse(prediction, batch["v"] / model.intensity_scale, batch["weight_idx"], _stack_weight_tensor(batch["stack_idx"], stack_weights))
                world_train = _regularization_world_points(model, batch["xyz"], batch["group_idx"])
                regularization = _quantitative_regularization(model, world_train, space.spatial_scaling, _require(loss_cfg, "quantitative"))
                trans = model.rigid_psf.transformation_loss(space.spatial_scaling)
                total = data_mse + sum(float(_require(loss_cfg, "quantitative")[key]) * regularization[key] for key in regularization)
                if joint:
                    total = total + float(_require(loss_cfg, "transformation")) * trans
                if not torch.isfinite(total):
                    raise FloatingPointError(f"Non-finite loss at stage {stage}, iteration {global_iter}.")
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                _assert_finite_gradients(model, stage, global_iter)
                optimizer.step()
                _assert_finite_parameters(model, stage, global_iter)
                scheduler.step()
                writer.writerow({"stage": stage, "iteration": global_iter, "data_mse": float(data_mse.detach()), "reg_t1": float(regularization["t1"].detach()), "reg_t2": float(regularization["t2"].detach()), "reg_b1": float(regularization["b1"].detach()), "transformation": float(trans.detach()), "total": float(total.detach()), "intensity_scale": float(model.intensity_scale.detach()), "lr_encoding": optimizer.param_groups[0]["lr"], "lr_network": optimizer.param_groups[1]["lr"], "lr_rigid": optimizer.param_groups[2]["lr"] if joint else ""})
            if stage == "A":
                stage_a_axisangle_final = model.rigid_psf.axisangle.detach().clone()
    save_checkpoint(output / "model.pt", model, config, protocol, dataset, space, seed, intensity_normalization)
    return {"model": model, "training_space": space, "protocol": protocol, "stack_weights": stack_weights, "intensity_normalization": intensity_normalization, "output_dir": output, "stage_a_axisangle_final": stage_a_axisangle_final if stage_a_axisangle_final is not None else model.rigid_psf.axisangle_init.detach().clone()}
