"""Q002 vector-loss loop that deliberately reuses B6 reconstruction controls."""

from __future__ import annotations

import random
import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import MultiStepLR

from mlp.modules.module_07_objective_training.mlp_trainer import _with_protocol_path, build_mlp_training_model
from reconstruction_core.orchestration import (
    _assert_finite_gradients, _assert_finite_parameters, _intensity_normalization_provenance,
    _optimizer, _quantitative_regularization, _regularization_world_points, _require,
    save_checkpoint,
)
from trad.modules.module_07_objective_training.experiment_infrastructure import normalize_step1_config
from trad.modules.module_07_objective_training.experiment_infrastructure import CachedBalancedSampler
from trad.modules.module_06_rigid_psf.hb1_stack_adapter import initialize_group_poses_from_hb1


def q002_training_controls(training: Mapping[str, Any]) -> dict[str, int]:
    """Validate the scientific controls that Q002 is not allowed to tune."""

    expected = {"stage_a_iterations": 2000, "stage_b_iterations": 4000, "batch_size": 640, "psf_samples": 8, "seed": 20260911}
    for key, value in expected.items():
        if int(training.get(key, -1)) != value:
            raise ValueError(f"Q002 requires B6 training.{key}={value}, got {training.get(key)!r}.")
    return {"stage_a_iterations": 2000, "stage_b_iterations": 4000, "effective_scalar_budget": 640, "anchor_batch_size": 64, "weights_per_anchor": 10, "psf_samples": 8, "seed": 20260911}


class JointAnchorSampler:
    """Uniform-with-replacement anchor sampler; B6 inverse stack weights remain in the loss."""

    def __init__(self, dataset: Any, *, seed: int) -> None:
        self.dataset = dataset
        self.generator = torch.Generator(device=dataset.xyz.device).manual_seed(int(seed))

    def sample(self, anchor_batch_size: int) -> dict[str, Any]:
        if anchor_batch_size != 64:
            raise ValueError("Q002 anchor_batch_size must be exactly 64 (effective scalar budget 640).")
        index = torch.randint(self.dataset.xyz.shape[0], (anchor_batch_size,), generator=self.generator, device=self.dataset.xyz.device)
        return {name: getattr(self.dataset, name)[index] for name in ("xyz", "observed", "group_idx", "stack_idx", "timing", "row_col")} | {"anchor_idx": index, "effective_scalar_budget": 640}


def joint_vector_mse(prediction: torch.Tensor, observed: torch.Tensor, stack_idx: torch.Tensor, stack_weights: Mapping[int, float]) -> torch.Tensor:
    if prediction.shape != observed.shape or prediction.ndim != 2 or prediction.shape[1] != 10:
        raise ValueError("Q002 prediction/observed must be matching [B,10] tensors.")
    weights = torch.empty_like(stack_idx, dtype=prediction.dtype)
    for stack, value in stack_weights.items():
        weights[stack_idx == int(stack)] = float(value)
    if not torch.isfinite(weights).all():
        raise ValueError("Q002 stack weights do not cover every sampled anchor.")
    return (weights * (prediction - observed).pow(2).mean(dim=1)).mean()


def regularization_batch_from_b6_scalar_support(sampler: CachedBalancedSampler, *, n_points: int) -> dict[str, Any]:
    """Reuse the B6 scalar sampler so Q002 regularization keeps its 640→256 semantics."""
    if n_points != 256:
        raise ValueError("Q002 requires B6 spatial_regularization.n_points=256.")
    batch = sampler.sample(640)
    if batch["xyz"].shape[0] < n_points:
        raise ValueError("B6 scalar regularization candidates must cover all 256 regularization points.")
    return {"batch": batch, "candidate_count": int(batch["xyz"].shape[0]), "effective_point_count": int(n_points), "sampling_source": "B6 scalar cropped support"}


def train_q002_reconstruction(scalar_dataset: Any, joint_dataset: Any, config: Mapping[str, Any], protocol_yaml: str | Path, output_dir: str | Path, *, prepared_inputs: list[str | Path]) -> dict[str, Any]:
    """Train fresh Q002 state; only sampler and data MSE differ from scalar B6."""

    resolved = normalize_step1_config(_with_protocol_path(config, protocol_yaml))
    training, loss_cfg = _require(resolved, "training"), _require(resolved, "loss")
    controls = q002_training_controls(training)
    device = torch.device(training["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device is unavailable: {device}")
    torch.manual_seed(controls["seed"]); np.random.seed(controls["seed"]); random.seed(controls["seed"])
    normalization = _intensity_normalization_provenance(training, scalar_dataset)
    stack_initialization = None
    initial_pose = None
    if bool(resolved["stack_initialization"]["enabled"]):
        stack_initialization = initialize_group_poses_from_hb1(scalar_dataset.group_axisangle_init, prepared_inputs, device=device, args_registration=resolved["stack_initialization"]["args_registration"])
        initial_pose = stack_initialization.post_stack_init_axisangle_physical.to(device)
    model, space, protocol, decoder_metadata = build_mlp_training_model(scalar_dataset, resolved, protocol_yaml, device, initial_group_axisangle_physical=initial_pose)
    model.intensity_scale.copy_(torch.as_tensor(normalization["scale"], device=device))
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Q002 output_dir must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    sampler = JointAnchorSampler(joint_dataset, seed=controls["seed"])
    regularization_sampler = CachedBalancedSampler(scalar_dataset)
    monitor = joint_dataset.fixed_monitor(seed=controls["seed"], samples_per_stack=int(training.get("monitor_samples_per_weight_per_stack", 1)))
    (output / "fixed_monitor.json").write_text(json.dumps({"seed": monitor["seed"], "samples_per_stack": monitor["samples_per_stack"], "anchor_identity_sha256": monitor["anchor_identity_sha256"]}, indent=2) + "\n")
    resolved_record = dict(resolved); resolved_record["q002_sampling"] = controls; resolved_record["q002_joint_stack_weights"] = {int(k): float(v) for k, v in joint_dataset.stack_weights.items()}; resolved_record["q002_intensity_normalization"] = normalization
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(resolved_record, sort_keys=False))
    started = time.perf_counter(); global_iteration = 0
    fields = ["stage", "iteration", "joint_signal_mse", "reg_t1", "reg_t2", "reg_b1", "amplitude_reg_t1", "amplitude_reg_t2", "transformation", "total", "intensity_scale", "lr_encoding", "lr_network", "lr_rigid", "regularization_candidate_count", "regularization_effective_point_count", "regularization_sampling_source"]
    log_handle = (output / "training_log.csv").open("w", newline="", encoding="utf-8"); monitor_handle = (output / "monitor_log.csv").open("w", newline="", encoding="utf-8")
    writer, monitor_writer = csv.DictWriter(log_handle, fieldnames=fields), csv.DictWriter(monitor_handle, fieldnames=["iteration", "stage", "monitor_mse", "per_weight_mse", "per_stack_mse", "fixed_monitor_anchor_identity_sha256"])
    writer.writeheader(); monitor_writer.writeheader()
    for stage, iterations, joint in (("A", 2000, False), ("B", 4000, True)):
        model.rigid_psf.axisangle.requires_grad_(joint)
        optimizer = _optimizer(model, training["learning_rates"], joint)
        milestones = [int(float(value) * iterations) for value in training["scheduler_milestones"] if 0 < float(value) < 1]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=float(training["scheduler_gamma"]))
        for iteration in range(iterations):
            global_iteration += 1
            batch = sampler.sample(64); batch["xyz"] = space.local_to_train(batch["xyz"].to(device)); batch["group_idx"] = batch["group_idx"].to(device); batch["stack_idx"] = batch["stack_idx"].to(device); batch["timing"] = batch["timing"].to(device); batch["observed"] = batch["observed"].to(device)
            prediction = model.forward_fingerprint(batch, 8)
            data = joint_vector_mse(prediction, batch["observed"] / model.intensity_scale, batch["stack_idx"], joint_dataset.stack_weights)
            regularization_cfg = _require(resolved, "spatial_regularization")
            regularization_batch = regularization_batch_from_b6_scalar_support(regularization_sampler, n_points=int(regularization_cfg["n_points"]))
            scalar_regularization = regularization_batch["batch"]
            regularization_world = _regularization_world_points(model, space.local_to_train(scalar_regularization["xyz"].to(device)), scalar_regularization["group_idx"].to(device))
            regularization = _quantitative_regularization(model, regularization_world, space.spatial_scaling, _require(loss_cfg, "quantitative"), regularization_cfg)
            total = data + sum(float(regularization_cfg[key]["weight"]) * regularization[key] for key in ("t1", "t2", "b1"))
            total = total + float(regularization_cfg["amplitude_guidance"]["t1_weight"]) * regularization["amplitude_t1"] + float(regularization_cfg["amplitude_guidance"]["t2_weight"]) * regularization["amplitude_t2"]
            if joint: total = total + float(_require(loss_cfg, "transformation")) * model.rigid_psf.transformation_loss(space.spatial_scaling)
            optimizer.zero_grad(set_to_none=True); total.backward(); _assert_finite_gradients(model, stage, iteration + 1); optimizer.step(); _assert_finite_parameters(model, stage, iteration + 1); scheduler.step()
            lrs = {group["name"]: group["lr"] for group in optimizer.param_groups}
            writer.writerow({"stage": stage, "iteration": global_iteration, "joint_signal_mse": float(data.detach()), "reg_t1": float(regularization["t1"].detach()), "reg_t2": float(regularization["t2"].detach()), "reg_b1": float(regularization["b1"].detach()), "amplitude_reg_t1": float(regularization["amplitude_t1"].detach()), "amplitude_reg_t2": float(regularization["amplitude_t2"].detach()), "transformation": float(model.rigid_psf.transformation_loss(space.spatial_scaling).detach()), "total": float(total.detach()), "intensity_scale": float(model.intensity_scale.detach()), "lr_encoding": lrs["encoding"], "lr_network": lrs["network"], "lr_rigid": lrs.get("rigid", ""), "regularization_candidate_count": regularization_batch["candidate_count"], "regularization_effective_point_count": regularization_batch["effective_point_count"], "regularization_sampling_source": regularization_batch["sampling_source"]})
            if int(training.get("monitor_every", 0)) and global_iteration % int(training["monitor_every"]) == 0:
                monitor_batch = {key: value.to(device) for key, value in monitor.items() if key in {"xyz", "observed", "group_idx", "stack_idx", "timing"}}
                monitor_batch["xyz"] = space.local_to_train(monitor_batch["xyz"])
                with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []), torch.no_grad(): prediction_monitor = model.forward_fingerprint(monitor_batch, 8)
                error = (prediction_monitor - monitor_batch["observed"] / model.intensity_scale).pow(2)
                monitor_writer.writerow({"iteration": global_iteration, "stage": stage, "monitor_mse": float(error.mean()), "per_weight_mse": json.dumps({weight: float(error[:, weight].mean()) for weight in range(10)}), "per_stack_mse": json.dumps({int(stack): float(error[monitor_batch["stack_idx"] == stack].mean()) for stack in monitor_batch["stack_idx"].unique().tolist()}), "fixed_monitor_anchor_identity_sha256": monitor["anchor_identity_sha256"]})
    log_handle.close(); monitor_handle.close()
    save_checkpoint(output / "model.pt", model, resolved, protocol, scalar_dataset, space, controls["seed"], normalization, decoder_metadata)
    if stack_initialization is not None:
        (output / "stack_initialization_poses.json").write_text(json.dumps({"coordinate_convention": "physical RAS mm; trans_first=true", "stacks": list(stack_initialization.stack_pose_records)}, indent=2) + "\n")
    (output / "timing_profile.csv").write_text("section,milliseconds\noptimization_total,{:.6f}\n".format((time.perf_counter() - started) * 1000.0))
    return {"model": model, "training_space": space, "protocol": protocol, "decoder_metadata": decoder_metadata, "intensity_normalization": normalization, "controls": controls, "stack_weights": joint_dataset.stack_weights, "output_dir": output, "stack_initialization": stack_initialization, "fixed_monitor": monitor}
