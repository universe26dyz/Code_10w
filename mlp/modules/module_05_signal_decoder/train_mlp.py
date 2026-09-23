"""mDM-style Adam training on one-time generated, provenance-checked HHZ data."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, TensorDataset
import yaml

from .mlp_model import MdmSignalMLP
from .synthetic_dataset import protocol_from_record
from .test_mlp import fidelity_metrics


def _require(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"MLP config lacks required key: {key}")
    return mapping[key]


def _protocol_record(protocol: Any) -> dict[str, Any]:
    return {
        "tr_ms": float(protocol.tr_ms), "vps": int(protocol.vps),
        "fa_deg": list(protocol.fa_deg), "ti_ms": list(protocol.ti_ms),
        "t2prep_ms": list(protocol.t2prep_ms), "n_ramp_up": int(protocol.n_ramp_up),
    }


def load_h5_split(path: str | Path, split: str, *, include_jacobian: bool = False) -> TensorDataset:
    """Open one split once and retain its float32 tensors in CPU RAM."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{split} HDF5 split does not exist: {path}")
    with h5py.File(path, "r") as handle:
        if set(("input12", "target_signal10")).difference(handle) or not ({"timing_id", "rhythm_id"} & set(handle)):
            raise ValueError(f"{path} lacks input12/target_signal10 and timing_id|rhythm_id.")
        inputs = np.asarray(handle["input12"][:], dtype=np.float32)
        targets = np.asarray(handle["target_signal10"][:], dtype=np.float32)
        jacobian = np.asarray(handle["target_jacobian10x3"][:], dtype=np.float32) if include_jacobian and "target_jacobian10x3" in handle else None
    if inputs.ndim != 2 or inputs.shape[1] != 12 or targets.shape != (inputs.shape[0], 10) or inputs.shape[0] < 1:
        raise ValueError(f"{path} has invalid MLP split shapes.")
    if include_jacobian:
        if jacobian is None or jacobian.shape != (inputs.shape[0], 10, 3):
            raise ValueError(f"{path} lacks valid target_jacobian10x3 [N,10,3].")
        return TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets), torch.from_numpy(jacobian))
    return TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))


class LazyRRSplit(Dataset):
    """Process-local, row-on-demand RR HDF5 dataset; never materializes arrays."""
    def __init__(self, path: str | Path, split: str, *, include_jacobian: bool = False) -> None:
        self.path, self.split, self.include_jacobian, self._handle = Path(path), split, include_jacobian, None
        with h5py.File(self.path, "r") as handle:
            if not {"input12", "target_signal10", "rhythm_id"}.issubset(handle):
                raise ValueError(f"{self.path} is not an RR HDF5 split.")
            self._length = int(handle["input12"].shape[0])
            if handle["input12"].shape != (self._length, 12) or handle["target_signal10"].shape != (self._length, 10):
                raise ValueError(f"{self.path} has invalid RR input/target shapes.")
            if include_jacobian and "target_jacobian10x3" not in handle:
                raise ValueError(f"{self.path} lacks requested Jacobian targets.")
    def __len__(self) -> int: return self._length
    def __getstate__(self):
        state = self.__dict__.copy(); state["_handle"] = None; return state
    def _file(self):
        if self._handle is None: self._handle = h5py.File(self.path, "r")
        return self._handle
    def __getitem__(self, index: int):
        handle = self._file(); values = [torch.from_numpy(np.asarray(handle["input12"][index], dtype=np.float32)), torch.from_numpy(np.asarray(handle["target_signal10"][index], dtype=np.float32))]
        if self.include_jacobian: values.append(torch.from_numpy(np.asarray(handle["target_jacobian10x3"][index], dtype=np.float32)))
        return tuple(values)


def _mlp_loss_settings(settings: Mapping[str, Any] | None) -> dict[str, float]:
    value = dict(settings or {})
    value.setdefault("signal_mse_weight", 1.0)
    value.setdefault("jacobian_weight", 0.0)
    value.setdefault("cosine_weight", 0.0)
    if value["signal_mse_weight"] != 1.0 or float(value["jacobian_weight"]) < 0 or float(value["cosine_weight"]) < 0:
        raise ValueError("mlp_loss requires signal_mse_weight=1 and non-negative optional weights.")
    return {key: float(value[key]) for key in ("signal_mse_weight", "jacobian_weight", "cosine_weight")}


def _mlp_output_jacobian(prediction: torch.Tensor, input12: torch.Tensor) -> torch.Tensor:
    columns = []
    for weight in range(10):
        gradient = torch.autograd.grad(prediction[:, weight].sum(), input12, retain_graph=True, create_graph=True)[0]
        columns.append(gradient[:, :3])
    return torch.stack(columns, dim=1)


def mlp_training_loss(model: MdmSignalMLP, input12: torch.Tensor, target_signal10: torch.Tensor, target_jacobian10x3: torch.Tensor | None, settings: Mapping[str, Any] | None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """mDM signal MSE plus optional normalized-coordinate Jacobian/cosine terms."""

    cfg = _mlp_loss_settings(settings)
    optional = cfg["jacobian_weight"] > 0 or cfg["cosine_weight"] > 0
    if optional and target_jacobian10x3 is None:
        raise ValueError("Jacobian/cosine MLP losses require target_jacobian10x3.")
    x = input12.requires_grad_(optional)
    prediction = model(x)
    signal_mse = (prediction - target_signal10).pow(2).mean()
    zero = torch.zeros((), device=prediction.device, dtype=prediction.dtype)
    jacobian_mse, cosine = zero, zero
    if optional:
        predicted_jacobian = _mlp_output_jacobian(prediction, x)
        # Both targets and predictions differentiate normalized signal with
        # respect to normalized [T1/1000,T2/1000,B1] coordinates.
        jacobian_mse = (predicted_jacobian - target_jacobian10x3).pow(2).mean()
        cosine = 1.0 - torch.nn.functional.cosine_similarity(predicted_jacobian.reshape(prediction.shape[0], -1), target_jacobian10x3.reshape(prediction.shape[0], -1), dim=-1, eps=1e-12).mean()
    total = signal_mse + cfg["jacobian_weight"] * jacobian_mse + cfg["cosine_weight"] * cosine
    return total, {"signal_mse": signal_mse, "jacobian_mse": jacobian_mse, "cosine": cosine}


def validate_dataset_provenance(dataset_dir: str | Path, protocol_path: str | Path) -> tuple[dict[str, Any], Any]:
    dataset_dir = Path(dataset_dir)
    metadata_path = dataset_dir / "dataset_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Synthetic dataset metadata does not exist: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("dataset_metadata.json must be a JSON object.")
    required = {"schema", "split_mode", "rhythm_split_ids", "timing9_min_ms", "timing9_max_ms", "train_timing9_min_ms", "train_timing9_max_ms", "protocol", "sizes"}
    missing = required.difference(metadata)
    if missing:
        raise ValueError(f"dataset_metadata.json lacks: {sorted(missing)}")
    if metadata.get("schema") != "mlp_rr_synthetic/v1" or metadata.get("split_mode") != "rhythm":
        raise ValueError("Only the active mlp_rr_synthetic/v1 rhythm-disjoint dataset is supported; subject/timing-pool datasets are archived.")
    protocol = protocol_from_record(metadata["protocol"], protocol_path)
    if len(metadata["timing9_min_ms"]) != 9 or len(metadata["timing9_max_ms"]) != 9:
        raise ValueError("Synthetic dataset timing range metadata must have exactly nine dimensions.")
    for split in ("train", "valid", "test"):
        if split not in metadata["sizes"] or int(metadata["sizes"][split]) < 1 or not (dataset_dir / f"{split}.h5").is_file():
            raise ValueError(f"Synthetic dataset lacks a valid declared {split} split.")
    for split in ("train", "valid", "test"):
        with h5py.File(dataset_dir / f"{split}.h5", "r") as handle:
            if "rhythm_id" not in handle or "timing_id" in handle:
                raise ValueError("RR synthetic HDF5 must contain rhythm_id and no legacy timing_id.")
    return metadata, protocol


def _check_loaded_split_sizes(metadata: Mapping[str, Any], splits: Mapping[str, Dataset]) -> None:
    for name, dataset in splits.items():
        if len(dataset) != int(metadata["sizes"][name]):
            raise ValueError(f"{name}.h5 size does not match dataset_metadata.json.")


def train_mlp(dataset_dir: str | Path, config: Mapping[str, Any], protocol_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    train_cfg = _require(config, "training")
    required = ("device", "seed", "epochs", "batch_size", "learning_rate", "scheduler_step_size", "scheduler_gamma", "gradient_samples")
    missing = [key for key in required if key not in train_cfg]
    if missing:
        raise ValueError(f"MLP training config lacks: {missing}")
    device = torch.device(train_cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA unavailable: {device}")
    batch_size = int(train_cfg["batch_size"])
    if batch_size < 2:
        raise ValueError("MLP BatchNorm training requires training.batch_size >= 2.")
    seed = int(train_cfg["seed"]); torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    dataset_dir, output = Path(dataset_dir), Path(output_dir)
    loss_settings = _mlp_loss_settings(config.get("mlp_loss"))
    metadata, teacher_protocol = validate_dataset_provenance(dataset_dir, protocol_path)
    include_jacobian = loss_settings["jacobian_weight"] > 0 or loss_settings["cosine_weight"] > 0
    splits = {name: LazyRRSplit(dataset_dir / f"{name}.h5", name, include_jacobian=include_jacobian) for name in ("train", "valid", "test")}
    _check_loaded_split_sizes(metadata, splits)
    if len(splits["train"]) < batch_size:
        raise ValueError("Training split is smaller than batch_size; refusing a BatchNorm singleton/fallback batch.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"MLP output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    loaders = {
        "train": DataLoader(splits["train"], batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0),
        "valid": DataLoader(splits["valid"], batch_size=batch_size, shuffle=False, num_workers=0),
    }
    model = MdmSignalMLP().to(device)
    optimizer = Adam(model.parameters(), lr=float(train_cfg["learning_rate"]))
    scheduler = StepLR(optimizer, step_size=int(train_cfg["scheduler_step_size"]), gamma=float(train_cfg["scheduler_gamma"]))
    best, best_epoch = float("inf"), None
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train(); train_total, train_count = 0.0, 0
        for batch in loaders["train"]:
            x, y = batch[0].to(device), batch[1].to(device)
            jacobian = batch[2].to(device) if include_jacobian else None
            loss, _ = mlp_training_loss(model, x, y, jacobian, loss_settings)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite MLP train loss at epoch {epoch}.")
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            train_total += float(loss.detach()) * x.shape[0]; train_count += x.shape[0]
        if train_count == 0:
            raise RuntimeError("No complete BatchNorm training batch was produced.")
        model.eval(); valid_total = 0.0
        with torch.no_grad():
            for batch in loaders["valid"]:
                x, y = batch[0], batch[1]
                valid_total += float((model(x.to(device)) - y.to(device)).pow(2).mean()) * x.shape[0]
        train_loss, valid_loss = train_total / train_count, valid_total / len(splits["valid"])
        scheduler.step(); history.append({"epoch": epoch, "train_mse": train_loss, "valid_mse": valid_loss, "lr": optimizer.param_groups[0]["lr"]})
        if valid_loss < best:
            best, best_epoch = valid_loss, epoch
            torch.save({
                "state_dict": model.state_dict(), "architecture": "12-200-200-200-10",
                "input_normalization": "T1/1000,T2/1000,B1,timing9/1000",
                "output_normalization": "raw_l2_normalized", "protocol_hhz_v1": _protocol_record(teacher_protocol),
                "dataset_schema": metadata["schema"], "dataset_split_mode": metadata["split_mode"],
                "timing9_min_ms": metadata["timing9_min_ms"], "timing9_max_ms": metadata["timing9_max_ms"],
                "train_timing9_min_ms": metadata["train_timing9_min_ms"], "train_timing9_max_ms": metadata["train_timing9_max_ms"],
                "functional_fixture": False, "formal_candidate": True, "validation_status": "unvalidated",
                "parameter_ranges": {"t1_ms": [20, 2500], "t2_ms": [5, 200], "b1": [0.1, 1.2], "constraint": "T1>T2"}, "mlp_loss": loss_settings, "jacobian_target_units": metadata.get("jacobian_target_units"), "seed": seed,
            }, output / "signal_simulator_best.pth")
    with (output / "train_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_mse", "valid_mse", "lr"]); writer.writeheader(); writer.writerows(history)
    checkpoint = torch.load(output / "signal_simulator_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    metrics = fidelity_metrics(model, splits["test"], batch_size, int(train_cfg["gradient_samples"]), protocol=teacher_protocol)
    with (output / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    resolved = dict(config)
    resolved.update({"resolved_protocol": _protocol_record(teacher_protocol), "dataset_metadata": metadata, "best_epoch": best_epoch, "best_valid_mse": best})
    with (output / "mlp_config_resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False)
    return {"model": model, "metrics": metrics, "best_checkpoint": output / "signal_simulator_best.pth"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--dataset-dir", required=True); parser.add_argument("--protocol", required=True); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("MLP config must be a YAML mapping.")
    result = train_mlp(args.dataset_dir, config, args.protocol, args.output_dir)
    print(json.dumps({"best_checkpoint": str(result["best_checkpoint"]), "metrics": result["metrics"]}, sort_keys=True))


if __name__ == "__main__":
    main()
