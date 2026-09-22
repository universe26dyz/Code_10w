"""Decoder-neutral staged quantitative reconstruction orchestration."""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
import yaml

from trad.modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset, robust_trimmed_mean_intensity
from trad.modules.module_04_quantitative_inr.quantitative_inr import QuantitativeINR, QuantitativeINRConfig
from trad.modules.module_05_signal_decoder.decoder_factory import DecoderFactory, bloch_decoder_factory
from trad.modules.module_05_signal_decoder.trad_signal_simulator import TradProtocol
from trad.modules.module_06_rigid_psf.rigid_psf_forward import GroupRigidPSF, TradQuantitativeForward
from trad.modules.module_06_rigid_psf.hb1_stack_adapter import StackInitialization, initialize_group_poses_from_hb1
from trad.modules.module_07_objective_training.training_space import TrainingSpace
from trad.modules.module_07_objective_training.experiment_infrastructure import CachedBalancedSampler, FixedMonitorSet, IterationProfiler, git_provenance, normalize_step1_config, write_experiment_manifest
from trad.evaluation.profiling.summarize_timing_profile import summarize_timing_profile


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


class _ExperimentalHeteroscedasticVarianceHead(nn.Module):
    """Project-specific experimental heteroscedastic branch, not NeSVoR sigma_net."""

    def __init__(self, group_count: int, settings: Mapping[str, Any]) -> None:
        super().__init__()
        self.pixel = bool(settings["pixel"])
        self.slice = bool(settings["slice"])
        hidden = int(settings["pixel_hidden_features"])
        self.pixel_network = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, 1)) if self.pixel else None
        self.slice_log_variance = nn.Embedding(group_count, 1) if self.slice else None
        if self.slice_log_variance is not None:
            nn.init.zeros_(self.slice_log_variance.weight)

    def forward(self, train_xyz: torch.Tensor, group_idx: torch.Tensor) -> torch.Tensor:
        value = torch.zeros(train_xyz.shape[0], dtype=train_xyz.dtype, device=train_xyz.device)
        if self.pixel_network is not None:
            value = value + self.pixel_network(train_xyz).squeeze(-1)
        if self.slice_log_variance is not None:
            value = value + self.slice_log_variance(group_idx).squeeze(-1)
        return value.clamp(-10.0, 10.0)


class ReconstructionTrainingModel(nn.Module):
    """Shared INR, rigid PSF, and an injected 10-weight signal decoder."""

    def __init__(self, inr: QuantitativeINR, rigid_psf: GroupRigidPSF, signal_decoder: nn.Module, protocol: TradProtocol, variance: Mapping[str, Any]) -> None:
        super().__init__()
        self.inr = inr
        self.rigid_psf = rigid_psf
        self.signal_simulator = signal_decoder
        self.forward_model = TradQuantitativeForward(inr, self.signal_simulator, rigid_psf, protocol)
        self.protocol = protocol
        self.variance_head = _ExperimentalHeteroscedasticVarianceHead(rigid_psf.group_count, variance) if bool(variance["enabled"]) else None
        self.register_buffer("intensity_scale", torch.ones(()))

    def forward(self, batch: Mapping[str, torch.Tensor], n_psf_samples: int, profile: Any = None) -> torch.Tensor:
        return self.forward_model(batch["xyz"], batch["group_idx"], batch["weight_idx"], batch["timing"], n_psf_samples, profile)


# Existing checkpoint state-dict keys retain the ``signal_simulator`` name;
# this alias preserves historical Trad imports while the engine is neutral.
TradTrainingModel = ReconstructionTrainingModel


def build_training_model(
    dataset: QuantPointDataset, config: Mapping[str, Any], protocol_yaml: str | Path, device: torch.device,
    *, initial_group_axisangle_physical: torch.Tensor | None = None, decoder_factory: DecoderFactory = bloch_decoder_factory,
) -> tuple[ReconstructionTrainingModel, TrainingSpace, TradProtocol, dict[str, str]]:
    config = normalize_step1_config(config)
    inr_cfg = _require(config, "inr")
    training_cfg = _require(config, "training")
    if not isinstance(inr_cfg, dict) or not isinstance(training_cfg, dict):
        raise ValueError("inr and training must be YAML mappings.")
    required_inr = ("coarsest_resolution_mm", "finest_resolution_mm", "level_scale", "n_features_per_level", "log2_hashmap_size", "latent_features", "width", "depth")
    missing = [key for key in required_inr if key not in inr_cfg]
    if missing:
        raise ValueError(f"inr config lacks explicit values: {missing}")
    bbox = _require(config, "bbox")
    space = TrainingSpace.from_dataset(dataset, float(_require(training_cfg, "spatial_scaling")), group_axisangle_init_physical=initial_group_axisangle_physical, bbox_margin_mm=float(bbox["margin_mm"]))
    protocol = build_protocol_from_dataset(dataset, protocol_yaml)
    inr = QuantitativeINR(
        space.bbox_train.to(device),
        QuantitativeINRConfig(**{key: inr_cfg[key] for key in required_inr}),
        spatial_scaling=space.spatial_scaling,
    ).to(device)
    rigid = GroupRigidPSF(
        space.group_axisangle_init_train.to(device), space.group_resolution_train.to(device)
    ).to(device)
    decoder, decoder_metadata = decoder_factory(dataset, config, protocol, device)
    return ReconstructionTrainingModel(inr, rigid, decoder, protocol, _require(config, "variance")).to(device), space, protocol, decoder_metadata


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


def _data_loss(model: TradTrainingModel, prediction: torch.Tensor, observed: torch.Tensor, batch: Mapping[str, torch.Tensor], stack_weights: torch.Tensor) -> torch.Tensor:
    """Balanced MSE when OFF; optional heteroscedastic likelihood when enabled."""

    if model.variance_head is None:
        return _balanced_mse(prediction, observed, batch["weight_idx"], stack_weights)
    log_variance = model.variance_head(batch["xyz"], batch["group_idx"])
    variance = torch.exp(log_variance)
    weighted = ((prediction - observed).pow(2) / variance + log_variance) * 0.5 * stack_weights
    return torch.stack([weighted[batch["weight_idx"] == weight].mean() for weight in range(10)]).mean()


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


def _quantitative_regularization(model: TradTrainingModel, train_xyz: torch.Tensor, spatial_scaling: float, weights: Mapping[str, Any], settings: Mapping[str, Any] | None = None) -> dict[str, torch.Tensor]:
    """Normalized per-field smoothness plus separately weighted amplitude guide."""

    settings = dict(settings or {"mode": "L2", "n_points": train_xyz.shape[0], "edge_epsilon": 1e-3, "amplitude_guidance": {"enabled": False}})
    settings.setdefault("n_points", train_xyz.shape[0])
    settings.setdefault("edge_epsilon", 1e-3)
    fields = ("t1", "t2", "b1")
    field_settings = {
        key: dict(settings.get(key, {"mode": settings.get("mode", "L2"), "weight": float(_require(weights, key))}))
        for key in fields
    }
    guidance = dict(settings.get("amplitude_guidance", {}))
    guidance.setdefault("enabled", False); guidance.setdefault("alpha", 1.0)
    guidance.setdefault("t1_weight", 0.0); guidance.setdefault("t2_weight", 0.0)
    zero = torch.zeros((), dtype=train_xyz.dtype, device=train_xyz.device)
    if all(field_settings[key].get("mode", "none") == "none" for key in fields):
        return {"t1": zero, "t2": zero, "b1": zero, "amplitude_t1": zero, "amplitude_t2": zero}
    points = train_xyz[: min(int(settings["n_points"]), train_xyz.shape[0])].detach().clone().requires_grad_(True)
    fields = model.inr(points)
    normalized = {
        "t1": fields["t1_ms"] / 2500.0,
        "t2": (fields["t2_ms"] - 5.0) / 195.0,
        "b1": (fields["b1"] - 0.1) / 1.1,
    }
    amplitude_edge_weight: torch.Tensor | None = None
    amplitude_active = bool(guidance["enabled"]) and (float(guidance["t1_weight"]) > 0 or float(guidance["t2_weight"]) > 0)
    if amplitude_active:
        amplitude_gradient = torch.autograd.grad(fields["amplitude"].sum(), points, create_graph=False, retain_graph=True)[0].detach() / spatial_scaling
        amplitude_edge_weight = torch.exp(-float(guidance.get("alpha", 1.0)) * torch.linalg.vector_norm(amplitude_gradient, dim=-1))
    values: dict[str, torch.Tensor] = {}
    for key, value in normalized.items():
        mode = str(field_settings[key].get("mode", settings.get("mode", "none")))
        if mode == "none":
            values[key] = zero
            if key in {"t1", "t2"}:
                values[f"amplitude_{key}"] = zero
            continue
        gradient = torch.autograd.grad(value.sum(), points, create_graph=True)[0] / spatial_scaling
        magnitude_sq = gradient.pow(2).sum(dim=-1)
        if mode == "TV":
            penalty = torch.sqrt(magnitude_sq + 1e-12)
        elif mode == "L2":
            penalty = magnitude_sq
        elif mode == "edge-preserving":
            epsilon = float(settings["edge_epsilon"])
            penalty = torch.sqrt(magnitude_sq + epsilon**2) - epsilon
        else:
            raise ValueError(f"Unsupported spatial regularization mode: {mode}")
        values[key] = penalty.mean()
        # This is an additional, independently weighted regularizer.  The
        # detached amplitude gradient never affects amplitude INR parameters,
        # and B1 intentionally has no amplitude-derived term.
        if key in {"t1", "t2"}:
            values[f"amplitude_{key}"] = (penalty * amplitude_edge_weight).mean() if amplitude_edge_weight is not None and float(guidance[f"{key}_weight"]) > 0 else zero
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
    if model.variance_head is not None:
        groups.append({"name": "variance", "params": list(model.variance_head.parameters()), "lr": float(_require(learning_rates, "variance"))})
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


def save_checkpoint(path: str | Path, model: TradTrainingModel, config: Mapping[str, Any], protocol: TradProtocol, dataset: QuantPointDataset, space: TrainingSpace, seed: int, intensity_normalization: Mapping[str, object], decoder_metadata: Mapping[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "model_state": model.state_dict(), "resolved_config": dict(config), "protocol_hhz_v1": asdict(protocol),
        "validated_tr_ms": protocol.tr_ms, "validated_vps": protocol.vps, "training_space": space.state_dict(),
        "group_resolution_xyz_mm": dataset.group_resolution_xyz_mm.detach().cpu(),
        "group_axisangle_dicom_physical": space.group_axisangle_dicom_physical.detach().cpu(),
        "group_axisangle_post_stack_init_physical": space.group_axisangle_init_physical.detach().cpu(),
        "group_axisangle_init_physical": space.group_axisangle_init_physical.detach().cpu(),
        "group_axisangle_init_train": space.group_axisangle_init_train.detach().cpu(),
        "trained_axisangle_train": model.rigid_psf.axisangle.detach().cpu(), "random_seed": int(seed),
        "intensity_normalization": dict(intensity_normalization), "decoder_metadata": dict(decoder_metadata or {}),
    }
    # MLP provenance is additive: baseline checkpoint fields and state-dict
    # keys remain untouched, while downstream MLP export keeps its contract.
    for key in ("source_mlp_checkpoint_path", "source_mlp_checkpoint_sha256", "source_mlp_checkpoint_metadata", "scientific_checkpoint"):
        if decoder_metadata is not None and key in decoder_metadata:
            payload[key] = decoder_metadata[key]
    torch.save(payload, Path(path))


def load_checkpoint(path: str | Path, model: TradTrainingModel, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    required = {"model_state", "protocol_hhz_v1", "training_space", "trained_axisangle_train", "intensity_normalization"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint lacks required fields: {sorted(missing)}")
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def train_reconstruction(dataset: QuantPointDataset, config: Mapping[str, Any], protocol_yaml: str | Path, output_dir: str | Path, *, decoder_factory: DecoderFactory = bloch_decoder_factory, route: str = "trad_bloch", prepared_inputs: list[str | Path] | None = None, subject_id: str | None = None, command: str = "") -> dict[str, Any]:
    """Run the sole staged reconstruction loop with an injected decoder."""

    run_started = time.perf_counter()
    config = normalize_step1_config(config)
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
    sampler = CachedBalancedSampler(dataset)
    monitor = FixedMonitorSet.from_dataset(dataset, seed=seed, samples_per_weight_per_stack=int(training["monitor_samples_per_weight_per_stack"])) if int(training["monitor_every"]) else None
    intensity_normalization = _intensity_normalization_provenance(training, dataset)
    stack_initialization: StackInitialization | None = None
    if bool(_require(config, "stack_initialization")["enabled"]):
        if not prepared_inputs:
            raise ValueError("stack_initialization.enabled requires prepared_inputs.")
        stack_initialization = initialize_group_poses_from_hb1(dataset.group_axisangle_init, list(prepared_inputs), device=device, args_registration=_require(config, "stack_initialization")["args_registration"])
    initial_pose = stack_initialization.post_stack_init_axisangle_physical.to(device) if stack_initialization is not None else None
    initialization_started = time.perf_counter()
    model, space, protocol, decoder_metadata = build_training_model(dataset, config, protocol_yaml, device, initial_group_axisangle_physical=initial_pose, decoder_factory=decoder_factory)
    initialization_ms = (time.perf_counter() - initialization_started) * 1000.0
    model.intensity_scale.copy_(torch.as_tensor(intensity_normalization["scale"], dtype=model.intensity_scale.dtype, device=device))
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output_dir must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if stack_initialization is not None:
        with (output / "stack_initialization_poses.json").open("w", encoding="utf-8") as handle:
            json.dump({"coordinate_convention": "physical RAS mm; trans_first=true", "stacks": list(stack_initialization.stack_pose_records)}, handle, indent=2)
    _write_resolved_config(output / "config_resolved.yaml", config, protocol, dataset, space, stack_weights, intensity_normalization)
    repo_root = Path(__file__).resolve().parents[1]
    if bool(training["require_clean_git"]) and git_provenance(repo_root)["git_dirty"]:
        raise RuntimeError("require_clean_git=true but the repository has uncommitted changes.")
    stack_group_counts = {f"stack_{stack}": int(torch.unique(dataset.group_idx[dataset.stack_idx == stack]).numel()) for stack in dataset.stack_idx.unique().tolist()}
    write_experiment_manifest(output, route=route, subject_id=subject_id, repo_root=repo_root, method_root=repo_root / "trad", config_resolved=output / "config_resolved.yaml", prepared_inputs=list(prepared_inputs or ()), protocol={"tr_ms": protocol.tr_ms, "vps": protocol.vps}, stack_group_counts=stack_group_counts, seed=seed, command=command)
    log_path = output / "training_log.csv"
    global_iter = 0
    stage_a_axisangle_final: torch.Tensor | None = None
    profiler = IterationProfiler(device, every=int(training["profile_every"]), warmup_samples=int(training["profile_warmup_samples"]))
    optimization_started = time.perf_counter()
    with log_path.open("w", newline="", encoding="utf-8") as handle, (output / "monitor_log.csv").open("w", newline="", encoding="utf-8") as monitor_handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "iteration", "data_mse", "reg_t1", "reg_t2", "reg_b1", "amplitude_reg_t1", "amplitude_reg_t2", "transformation", "total", "intensity_scale", "lr_encoding", "lr_network", "lr_rigid", "lr_variance"])
        writer.writeheader()
        monitor_writer = csv.DictWriter(monitor_handle, fieldnames=["iteration", "stage", "monitor_mse", "per_weight_mse", "per_stack_mse"]); monitor_writer.writeheader()
        for stage, iterations, joint in (("A", int(training["stage_a_iterations"]), False), ("B", int(training["stage_b_iterations"]), True)):
            if iterations < 0:
                raise ValueError(f"Stage {stage} iterations must be non-negative.")
            model.rigid_psf.axisangle.requires_grad_(joint)
            optimizer = _optimizer(model, _require(training, "learning_rates"), joint)
            milestones = [int(float(value) * iterations) for value in training["scheduler_milestones"] if 0 < float(value) < 1]
            scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=float(training["scheduler_gamma"]))
            for _ in range(iterations):
                global_iter += 1
                profiler.begin(global_iter, stage)
                with profiler.section("batch_sampling"): batch = sampler.sample(int(training["batch_size"]))
                with profiler.section("coordinate_conversion"): batch["xyz"] = space.local_to_train(batch["xyz"])
                prediction = model(batch, int(training["psf_samples"]), profiler.section)
                with profiler.section("loss"): data_mse = _data_loss(model, prediction, batch["v"] / model.intensity_scale, batch, _stack_weight_tensor(batch["stack_idx"], stack_weights))
                world_train = _regularization_world_points(model, batch["xyz"], batch["group_idx"])
                regularization_cfg = _require(config, "spatial_regularization")
                with profiler.section("spatial_regularization"): regularization = _quantitative_regularization(model, world_train, space.spatial_scaling, _require(loss_cfg, "quantitative"), regularization_cfg)
                trans = model.rigid_psf.transformation_loss(space.spatial_scaling)
                total = data_mse + sum(float(regularization_cfg[key]["weight"]) * regularization[key] for key in ("t1", "t2", "b1"))
                total = total + float(regularization_cfg["amplitude_guidance"]["t1_weight"]) * regularization["amplitude_t1"] + float(regularization_cfg["amplitude_guidance"]["t2_weight"]) * regularization["amplitude_t2"]
                if joint:
                    total = total + float(_require(loss_cfg, "transformation")) * trans
                if not torch.isfinite(total):
                    raise FloatingPointError(f"Non-finite loss at stage {stage}, iteration {global_iter}.")
                optimizer.zero_grad(set_to_none=True)
                with profiler.section("backward"): total.backward()
                _assert_finite_gradients(model, stage, global_iter)
                with profiler.section("optimizer_step"): optimizer.step()
                _assert_finite_parameters(model, stage, global_iter)
                scheduler.step()
                lrs = {group["name"]: group["lr"] for group in optimizer.param_groups}
                writer.writerow({"stage": stage, "iteration": global_iter, "data_mse": float(data_mse.detach()), "reg_t1": float(regularization["t1"].detach()), "reg_t2": float(regularization["t2"].detach()), "reg_b1": float(regularization["b1"].detach()), "amplitude_reg_t1": float(regularization["amplitude_t1"].detach()), "amplitude_reg_t2": float(regularization["amplitude_t2"].detach()), "transformation": float(trans.detach()), "total": float(total.detach()), "intensity_scale": float(model.intensity_scale.detach()), "lr_encoding": lrs["encoding"], "lr_network": lrs["network"], "lr_rigid": lrs.get("rigid", ""), "lr_variance": lrs.get("variance", "")})
                if monitor is not None and global_iter % int(training["monitor_every"]) == 0:
                    monitor_batch = dict(monitor.batch); monitor_batch["xyz"] = space.local_to_train(monitor_batch["xyz"])
                    with torch.no_grad(): monitor_prediction = model(monitor_batch, int(training["psf_samples"]))
                    monitor_error = (monitor_prediction - monitor_batch["v"] / model.intensity_scale).pow(2)
                    monitor_writer.writerow({"iteration": global_iter, "stage": stage, "monitor_mse": float(monitor_error.mean()), "per_weight_mse": json.dumps({weight: float(monitor_error[monitor_batch["weight_idx"] == weight].mean()) for weight in range(10)}), "per_stack_mse": json.dumps({int(stack): float(monitor_error[monitor_batch["stack_idx"] == stack].mean()) for stack in monitor_batch["stack_idx"].unique().tolist()})})
                if int(training["checkpoint_every"]) and global_iter % int(training["checkpoint_every"]) == 0:
                    save_checkpoint(output / f"model_iter_{global_iter:04d}.pt", model, config, protocol, dataset, space, seed, intensity_normalization, decoder_metadata)
                profiler.end()
            if stage == "A":
                stage_a_axisangle_final = model.rigid_psf.axisangle.detach().clone()
    save_checkpoint(output / "model.pt", model, config, protocol, dataset, space, seed, intensity_normalization, decoder_metadata)
    profiler.write_csv(output / "timing_profile.csv")
    optimization_ms = (time.perf_counter() - optimization_started) * 1000.0
    decoder_type = decoder_metadata.get("decoder_type", "unknown")
    summary = summarize_timing_profile(output / "timing_profile.csv", stage_iterations={"A": int(training["stage_a_iterations"]), "B": int(training["stage_b_iterations"])}, decoder_type=decoder_type, run_level_ms={"checkpoint_load": initialization_ms if decoder_type == "FrozenMLP" else 0.0, "data_loading": 0.0, "initialization": initialization_ms, "optimization_total": optimization_ms, "validation": 0.0, "export": 0.0, "total_runtime": (time.perf_counter() - run_started) * 1000.0})
    return {"model": model, "training_space": space, "protocol": protocol, "decoder_metadata": decoder_metadata, "timing_profile_summary": summary, "stack_weights": stack_weights, "intensity_normalization": intensity_normalization, "output_dir": output, "stack_initialization": stack_initialization, "stage_a_axisangle_final": stage_a_axisangle_final if stage_a_axisangle_final is not None else model.rigid_psf.axisangle_init.detach().clone()}
