"""Trad-identical staged MLP reconstruction, differing only in signal decoder."""

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

from modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset
from modules.module_04_quantitative_inr.quantitative_inr import QuantitativeINR, QuantitativeINRConfig
from modules.module_05_signal_decoder.checkpoint_loader import load_frozen_mlp_decoder
from modules.module_05_signal_decoder.provenance import sha256_file
from modules.module_05_signal_decoder.trad_teacher.trad_signal_simulator import TradProtocol
from modules.module_06_rigid_psf.rigid_psf_forward import GroupRigidPSF, TradQuantitativeForward
from modules.module_06_rigid_psf.hb1_stack_adapter import StackInitialization, initialize_group_poses_from_hb1
from .training_space import TrainingSpace
from .experiment_infrastructure import CachedBalancedSampler, FixedMonitorSet, IterationProfiler, git_provenance, normalize_step1_config, write_experiment_manifest


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
    protocol = _load_yaml(protocol_yaml)
    tr_ms, vps = dataset.validated_tr_vps()
    dicom_fields = _require(protocol, "dicom_fields")
    if not isinstance(dicom_fields, dict) or dicom_fields.get("tr_ms") != "RepetitionTime" or dicom_fields.get("vps") != "EchoTrainLength":
        raise ValueError("protocol_hhz_v1.yaml must declare DICOM RepetitionTime and EchoTrainLength provenance.")
    result = TradProtocol(tr_ms, vps, tuple(float(x) for x in _require(protocol, "flip_angle_degrees")), tuple(float(x) for x in _require(protocol, "inversion_times_ms")), tuple(float(x) for x in _require(protocol, "t2prep_ms")), int(_require(protocol, "ramp_up_pulses")))
    result.validate()
    return result


class MLPTrainingModel(nn.Module):
    """The frozen MLP is injected into the exact existing rigid/PSF forward."""

    def __init__(self, inr: QuantitativeINR, rigid_psf: GroupRigidPSF, signal_simulator: nn.Module, protocol: TradProtocol, variance: Mapping[str, Any]) -> None:
        super().__init__()
        self.inr, self.rigid_psf, self.signal_simulator, self.protocol = inr, rigid_psf, signal_simulator, protocol
        self.forward_model = TradQuantitativeForward(inr, signal_simulator, rigid_psf, protocol)
        self.register_buffer("intensity_scale", torch.ones(()))
        self.variance_head = nn.Sequential(nn.Linear(3, int(variance.get("pixel_hidden_features", 16))), nn.SiLU(), nn.Linear(int(variance.get("pixel_hidden_features", 16)), 1)) if bool(variance.get("enabled", False)) and bool(variance.get("pixel", False)) else None
        self.slice_log_variance = nn.Embedding(rigid_psf.group_count, 1) if bool(variance.get("enabled", False)) and bool(variance.get("slice", False)) else None

    def forward(self, batch: Mapping[str, torch.Tensor], n_psf_samples: int, profile: Any = None) -> torch.Tensor:
        return self.forward_model(batch["xyz"], batch["group_idx"], batch["weight_idx"], batch["timing"], n_psf_samples, profile)


def build_mlp_training_model(dataset: QuantPointDataset, config: Mapping[str, Any], protocol_yaml: str | Path, device: torch.device, *, initial_group_axisangle_physical: torch.Tensor | None = None) -> tuple[MLPTrainingModel, TrainingSpace, TradProtocol, dict[str, Any]]:
    config = normalize_step1_config(config)
    inr_cfg, training_cfg, decoder_cfg = _require(config, "inr"), _require(config, "training"), _require(config, "decoder")
    if not isinstance(inr_cfg, dict) or not isinstance(training_cfg, dict) or not isinstance(decoder_cfg, dict):
        raise ValueError("inr, training, and decoder must be YAML mappings.")
    required_inr = ("coarsest_resolution_mm", "finest_resolution_mm", "level_scale", "n_features_per_level", "log2_hashmap_size", "latent_features", "width", "depth")
    missing = [key for key in required_inr if key not in inr_cfg]
    if missing:
        raise ValueError(f"inr config lacks explicit values: {missing}")
    checkpoint_path, allowance = _require(decoder_cfg, "checkpoint"), _require(decoder_cfg, "allow_functional_fixture_checkpoint")
    if not isinstance(allowance, bool):
        raise ValueError("decoder.allow_functional_fixture_checkpoint must be explicit boolean.")
    bbox = _require(config, "bbox")
    space = TrainingSpace.from_dataset(dataset, float(_require(training_cfg, "spatial_scaling")), group_axisangle_init_physical=initial_group_axisangle_physical, bbox_margin_mm=float(bbox["margin_mm"]))
    protocol = build_protocol_from_dataset(dataset, protocol_yaml)
    inr = QuantitativeINR(space.bbox_train.to(device), QuantitativeINRConfig(**{key: inr_cfg[key] for key in required_inr}), spatial_scaling=space.spatial_scaling).to(device)
    rigid = GroupRigidPSF(space.group_axisangle_init_train.to(device), space.group_resolution_train.to(device)).to(device)
    decoder = load_frozen_mlp_decoder(checkpoint_path, dataset, protocol_yaml, allow_functional_fixture=allowance, device=device)
    source_metadata = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    return MLPTrainingModel(inr, rigid, decoder, protocol, _require(config, "variance")).to(device), space, protocol, source_metadata


def _balanced_batch(dataset: QuantPointDataset, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    if batch_size < 10 or batch_size % 10:
        raise ValueError("batch_size must be a positive multiple of 10 for explicit 10-weight balance.")
    per_weight = batch_size // 10; indices = []
    for weight in range(10):
        candidates = torch.nonzero(dataset.weight_idx == weight, as_tuple=False).flatten()
        if candidates.numel() == 0:
            raise ValueError(f"No valid samples exist for weight {weight}.")
        indices.append(candidates[torch.randint(candidates.numel(), (per_weight,), device=device)])
    index = torch.cat(indices); index = index[torch.randperm(index.numel(), device=device)]
    return {"xyz": dataset.xyz[index], "v": dataset.v[index], "group_idx": dataset.group_idx[index], "weight_idx": dataset.weight_idx[index], "stack_idx": dataset.stack_idx[index], "timing": dataset.timing[index]}


def _stack_weight_tensor(stack_idx: torch.Tensor, weights: Mapping[int, float]) -> torch.Tensor:
    values = torch.empty_like(stack_idx, dtype=torch.float32)
    for stack, weight in weights.items():
        values[stack_idx == stack] = weight
    return values


def _balanced_mse(prediction: torch.Tensor, observed: torch.Tensor, weight_idx: torch.Tensor, stack_weights: torch.Tensor) -> torch.Tensor:
    weighted = (prediction - observed).pow(2) * stack_weights
    return torch.stack([weighted[weight_idx == weight].mean() for weight in range(10)]).mean()


def _data_loss(model: MLPTrainingModel, prediction: torch.Tensor, observed: torch.Tensor, batch: Mapping[str, torch.Tensor], stack_weights: torch.Tensor) -> torch.Tensor:
    if model.variance_head is None and model.slice_log_variance is None:
        return _balanced_mse(prediction, observed, batch["weight_idx"], stack_weights)
    log_variance = torch.zeros_like(prediction)
    if model.variance_head is not None: log_variance = log_variance + model.variance_head(batch["xyz"]).squeeze(-1)
    if model.slice_log_variance is not None: log_variance = log_variance + model.slice_log_variance(batch["group_idx"]).squeeze(-1)
    log_variance = log_variance.clamp(-10, 10); variance = log_variance.exp()
    weighted = ((prediction - observed).pow(2) / variance + log_variance) * 0.5 * stack_weights
    return torch.stack([weighted[batch["weight_idx"] == weight].mean() for weight in range(10)]).mean()


def _intensity_scale(dataset: QuantPointDataset) -> torch.Tensor:
    lower, upper = torch.quantile(dataset.v, 0.1), torch.quantile(dataset.v, 0.9)
    trimmed = dataset.v[(dataset.v > lower) & (dataset.v < upper)]
    if trimmed.numel() == 0 or not torch.isfinite(trimmed).all() or trimmed.mean() <= 0:
        raise ValueError("MLP subject intensity normalization requires finite positive trimmed mean.")
    return trimmed.mean()


def _assert_finite(model: nn.Module, stage: str, iteration: int) -> None:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and ((parameter.grad is not None and not torch.isfinite(parameter.grad).all()) or not torch.isfinite(parameter).all()):
            raise FloatingPointError(f"Non-finite MLP parameter/gradient at {stage} iteration {iteration}: {name}.")


def _quantitative_regularization(model: MLPTrainingModel, train_xyz: torch.Tensor, spatial_scaling: float, weights: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    requested = {"t1": float(_require(weights, "t1")), "t2": float(_require(weights, "t2")), "b1": float(_require(weights, "b1"))}
    if not any(requested.values()):
        zero = torch.zeros((), dtype=train_xyz.dtype, device=train_xyz.device)
        return {key: zero for key in requested}
    points = train_xyz[: min(4, train_xyz.shape[0])].detach().clone().requires_grad_(True)
    fields = model.inr(points)
    normalized = {"t1": fields["t1_ms"] / 2500.0, "t2": (fields["t2_ms"] - 5.0) / 195.0, "b1": (fields["b1"] - 0.1) / 1.1}
    values = {}
    for key, value in normalized.items():
        values[key] = (torch.autograd.grad(value.sum(), points, create_graph=True)[0] / spatial_scaling).pow(2).sum(dim=-1).sqrt().mean()
    return values


def _regularization_world_points(model: MLPTrainingModel, local_train: torch.Tensor, group_idx: torch.Tensor) -> torch.Tensor:
    return model.rigid_psf.transform_local_to_ras(local_train, group_idx).detach()


def _optimizer(model: MLPTrainingModel, rates: Mapping[str, Any], joint: bool) -> AdamW:
    encoding, networks = [], []
    for name, parameter in model.inr.named_parameters():
        (encoding if name.startswith("encoding") else networks).append(parameter)
    groups = [{"name": "encoding", "params": encoding, "lr": float(_require(rates, "encoding"))}, {"name": "network", "params": networks, "lr": float(_require(rates, "network")), "weight_decay": float(_require(rates, "weight_decay"))}]
    if joint:
        groups.append({"name": "rigid", "params": [model.rigid_psf.axisangle], "lr": float(_require(rates, "rigid"))})
    variance = ([] if model.variance_head is None else list(model.variance_head.parameters())) + ([] if model.slice_log_variance is None else list(model.slice_log_variance.parameters()))
    if variance: groups.append({"name": "variance", "params": variance, "lr": float(_require(rates, "variance"))})
    return AdamW(groups, betas=(0.9, 0.99), eps=1e-15)


def _bn_buffers(decoder: nn.Module) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(layer.running_mean.detach().clone(), layer.running_var.detach().clone()) for layer in decoder.modules() if isinstance(layer, nn.BatchNorm1d)]


def _assert_decoder_bn_frozen(decoder: nn.Module, before: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    after = [(layer.running_mean, layer.running_var) for layer in decoder.modules() if isinstance(layer, nn.BatchNorm1d)]
    if any(layer.training for layer in decoder.modules() if isinstance(layer, nn.BatchNorm1d)):
        raise RuntimeError("Frozen MLP BatchNorm entered training mode during reconstruction.")
    if len(before) != len(after) or any(not torch.equal(old[0], new[0]) or not torch.equal(old[1], new[1]) for old, new in zip(before, after)):
        raise RuntimeError("Frozen MLP BatchNorm running statistics changed during reconstruction.")


def _write_resolved_config(path: Path, config: Mapping[str, Any], protocol: TradProtocol, dataset: QuantPointDataset, space: TrainingSpace, stack_weights: Mapping[int, float]) -> None:
    resolved = dict(config)
    resolved.update({"validated_protocol": {"tr_ms": protocol.tr_ms, "vps": protocol.vps, "fa_deg": list(protocol.fa_deg), "ti_ms": list(protocol.ti_ms), "t2prep_ms": list(protocol.t2prep_ms), "n_ramp_up": protocol.n_ramp_up}, "stack_weights": {int(key): float(value) for key, value in stack_weights.items()}, "training_space": {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in space.state_dict().items()}, "group_tr_ms": dataset.group_tr_ms.detach().cpu().tolist(), "group_vps": dataset.group_vps.detach().cpu().tolist()})
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, allow_unicode=True, sort_keys=False)


def save_mlp_reconstruction_checkpoint(path: str | Path, model: MLPTrainingModel, config: Mapping[str, Any], protocol: TradProtocol, dataset: QuantPointDataset, space: TrainingSpace, seed: int, source_checkpoint_path: str | Path, source_metadata: Mapping[str, Any]) -> None:
    torch.save({"model_state": model.state_dict(), "resolved_config": dict(config), "protocol_hhz_v1": asdict(protocol), "validated_tr_ms": protocol.tr_ms, "validated_vps": protocol.vps, "training_space": space.state_dict(), "group_resolution_xyz_mm": dataset.group_resolution_xyz_mm.detach().cpu(), "group_axisangle_dicom_physical": space.group_axisangle_dicom_physical.detach().cpu(), "group_axisangle_post_stack_init_physical": space.group_axisangle_init_physical.detach().cpu(), "group_axisangle_init_physical": space.group_axisangle_init_physical.detach().cpu(), "group_axisangle_init_train": space.group_axisangle_init_train.detach().cpu(), "trained_axisangle_train": model.rigid_psf.axisangle.detach().cpu(), "random_seed": int(seed), "intensity_normalization": {"method": "trimmed_mean", "scale": float(model.intensity_scale.detach().cpu()), "amplitude_export_units": "original_input_intensity"}, "source_mlp_checkpoint_path": str(source_checkpoint_path), "source_mlp_checkpoint_sha256": sha256_file(source_checkpoint_path), "source_mlp_checkpoint_metadata": dict(source_metadata), "scientific_checkpoint": bool(source_metadata["scientific_checkpoint"])}, Path(path))


def load_mlp_reconstruction_checkpoint(path: str | Path, model: MLPTrainingModel, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    required = {"model_state", "protocol_hhz_v1", "training_space", "trained_axisangle_train", "source_mlp_checkpoint_sha256", "source_mlp_checkpoint_metadata", "scientific_checkpoint"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"MLP reconstruction checkpoint lacks required fields: {sorted(missing)}")
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def train_mlp_reconstruction(dataset: QuantPointDataset, config: Mapping[str, Any], protocol_yaml: str | Path, output_dir: str | Path, *, prepared_inputs: list[str | Path] | None = None, subject_id: str | None = None, command: str = "") -> dict[str, Any]:
    config = normalize_step1_config(config)
    training, loss_cfg = _require(config, "training"), _require(config, "loss")
    if not isinstance(training, dict) or not isinstance(loss_cfg, dict):
        raise ValueError("training and loss must be mappings.")
    required = ("device", "seed", "batch_size", "psf_samples", "stage_a_iterations", "stage_b_iterations", "spatial_scaling", "learning_rates", "scheduler_milestones", "scheduler_gamma")
    missing = [key for key in required if key not in training]
    if missing:
        raise ValueError(f"training config lacks explicit values: {missing}")
    device = torch.device(training["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device is unavailable: {device}")
    seed = int(training["seed"]); torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    stack_weights = dataset.validate_balanced_samples(); sampler = CachedBalancedSampler(dataset)
    monitor = FixedMonitorSet.from_dataset(dataset, seed=seed, samples_per_weight_per_stack=int(training["monitor_samples_per_weight_per_stack"])) if int(training["monitor_every"]) else None
    stack_initialization: StackInitialization | None = None
    if bool(_require(config, "stack_initialization")["enabled"]):
        if not prepared_inputs: raise ValueError("stack_initialization.enabled requires prepared_inputs.")
        stack_initialization = initialize_group_poses_from_hb1(dataset.group_axisangle_init, list(prepared_inputs), device=device, args_registration=_require(config, "stack_initialization")["args_registration"])
    initial = stack_initialization.post_stack_init_axisangle_physical.to(device) if stack_initialization is not None else None
    model, space, protocol, source_metadata = build_mlp_training_model(dataset, config, protocol_yaml, device, initial_group_axisangle_physical=initial)
    model.intensity_scale.copy_(_intensity_scale(dataset).to(device))
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output_dir must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True); _write_resolved_config(output / "config_resolved.yaml", config, protocol, dataset, space, stack_weights)
    if stack_initialization is not None:
        (output / "stack_initialization_poses.json").write_text(json.dumps({"coordinate_convention": "physical RAS mm; trans_first=true", "stacks": list(stack_initialization.stack_pose_records)}, indent=2), encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[3]
    if bool(training["require_clean_git"]) and git_provenance(repo_root)["git_dirty"]: raise RuntimeError("require_clean_git=true but the repository has uncommitted changes.")
    stack_group_counts = {f"stack_{stack}": int(torch.unique(dataset.group_idx[dataset.stack_idx == stack]).numel()) for stack in dataset.stack_idx.unique().tolist()}
    write_experiment_manifest(output, route="mlp", subject_id=subject_id, repo_root=repo_root, method_root=Path(__file__).resolve().parents[2], config_resolved=output / "config_resolved.yaml", prepared_inputs=list(prepared_inputs or ()), protocol={"tr_ms": protocol.tr_ms, "vps": protocol.vps}, stack_group_counts=stack_group_counts, seed=seed, command=command)
    bn_before = _bn_buffers(model.signal_simulator); global_iter = 0; stage_a_axisangle_final = None
    profiler = IterationProfiler(device, every=int(training["profile_every"]), warmup_samples=int(training["profile_warmup_samples"]))
    with (output / "training_log.csv").open("w", newline="", encoding="utf-8") as handle, (output / "monitor_log.csv").open("w", newline="", encoding="utf-8") as monitor_handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "iteration", "data_mse", "reg_t1", "reg_t2", "reg_b1", "transformation", "total", "lr_encoding", "lr_network", "lr_rigid"]); writer.writeheader()
        monitor_writer = csv.DictWriter(monitor_handle, fieldnames=["iteration", "stage", "monitor_mse", "per_weight_mse", "per_stack_mse"]); monitor_writer.writeheader()
        for stage, iterations, joint in (("A", int(training["stage_a_iterations"]), False), ("B", int(training["stage_b_iterations"]), True)):
            if iterations < 0:
                raise ValueError(f"Stage {stage} iterations must be non-negative.")
            model.rigid_psf.axisangle.requires_grad_(joint); model.train(); _assert_decoder_bn_frozen(model.signal_simulator, bn_before)
            optimizer = _optimizer(model, _require(training, "learning_rates"), joint)
            milestones = [int(float(value) * iterations) for value in training["scheduler_milestones"] if 0 < float(value) < 1]
            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=float(training["scheduler_gamma"]))
            for _ in range(iterations):
                global_iter += 1; profiler.begin(global_iter, stage)
                with profiler.section("batch_sampling"): batch = sampler.sample(int(training["batch_size"]))
                with profiler.section("coordinate_conversion"): batch["xyz"] = space.local_to_train(batch["xyz"])
                prediction = model(batch, int(training["psf_samples"]), profiler.section)
                with profiler.section("data_loss"): data_mse = _data_loss(model, prediction, batch["v"] / model.intensity_scale, batch, _stack_weight_tensor(batch["stack_idx"], stack_weights))
                world_train = _regularization_world_points(model, batch["xyz"], batch["group_idx"])
                with profiler.section("spatial_regularization"): regularization = _quantitative_regularization(model, world_train, space.spatial_scaling, _require(loss_cfg, "quantitative"))
                trans = model.rigid_psf.transformation_loss(space.spatial_scaling)
                total = data_mse + sum(float(_require(loss_cfg, "quantitative")[key]) * regularization[key] for key in regularization)
                if joint:
                    total = total + float(_require(loss_cfg, "transformation")) * trans
                if not torch.isfinite(total):
                    raise FloatingPointError(f"Non-finite loss at stage {stage}, iteration {global_iter}.")
                optimizer.zero_grad(set_to_none=True)
                with profiler.section("backward"): total.backward()
                _assert_finite(model, stage, global_iter)
                with profiler.section("optimizer_step"): optimizer.step()
                scheduler.step()
                writer.writerow({"stage": stage, "iteration": global_iter, "data_mse": float(data_mse.detach()), "reg_t1": float(regularization["t1"].detach()), "reg_t2": float(regularization["t2"].detach()), "reg_b1": float(regularization["b1"].detach()), "transformation": float(trans.detach()), "total": float(total.detach()), "lr_encoding": optimizer.param_groups[0]["lr"], "lr_network": optimizer.param_groups[1]["lr"], "lr_rigid": optimizer.param_groups[2]["lr"] if joint else ""})
                if monitor is not None and global_iter % int(training["monitor_every"]) == 0:
                    monitor_batch = dict(monitor.batch); monitor_batch["xyz"] = space.local_to_train(monitor_batch["xyz"])
                    with torch.no_grad(): monitor_prediction = model(monitor_batch, int(training["psf_samples"]))
                    monitor_error = (monitor_prediction - monitor_batch["v"]).pow(2)
                    monitor_writer.writerow({"iteration": global_iter, "stage": stage, "monitor_mse": float(monitor_error.mean()), "per_weight_mse": json.dumps({weight: float(monitor_error[monitor_batch["weight_idx"] == weight].mean()) for weight in range(10)}), "per_stack_mse": json.dumps({int(stack): float(monitor_error[monitor_batch["stack_idx"] == stack].mean()) for stack in monitor_batch["stack_idx"].unique().tolist()})})
                if int(training["checkpoint_every"]) and global_iter % int(training["checkpoint_every"]) == 0: save_mlp_reconstruction_checkpoint(output / f"model_iter_{global_iter:04d}.pt", model, config, protocol, dataset, space, seed, _require(_require(config, "decoder"), "checkpoint"), source_metadata)
                profiler.end()
            if stage == "A":
                stage_a_axisangle_final = model.rigid_psf.axisangle.detach().clone()
    _assert_decoder_bn_frozen(model.signal_simulator, bn_before)
    save_mlp_reconstruction_checkpoint(output / "model.pt", model, config, protocol, dataset, space, seed, _require(_require(config, "decoder"), "checkpoint"), source_metadata)
    profiler.write_csv(output / "timing_profile.csv")
    return {"model": model, "training_space": space, "protocol": protocol, "stack_weights": stack_weights, "output_dir": output, "stage_a_axisangle_final": stage_a_axisangle_final if stage_a_axisangle_final is not None else model.rigid_psf.axisangle_init.detach().clone()}
