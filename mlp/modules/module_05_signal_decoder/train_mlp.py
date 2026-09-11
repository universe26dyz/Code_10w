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
from torch.utils.data import DataLoader, TensorDataset
import yaml

from .mlp_model import MdmSignalMLP
from .provenance import sha256_file
from .synthetic_dataset import _protocol, load_timing_pool
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


def load_h5_split(path: str | Path, split: str) -> TensorDataset:
    """Open one split once and retain its float32 tensors in CPU RAM."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{split} HDF5 split does not exist: {path}")
    with h5py.File(path, "r") as handle:
        if set(("input12", "target_signal10", "timing_id")).difference(handle):
            raise ValueError(f"{path} lacks input12/target_signal10/timing_id.")
        inputs = np.asarray(handle["input12"][:], dtype=np.float32)
        targets = np.asarray(handle["target_signal10"][:], dtype=np.float32)
    if inputs.ndim != 2 or inputs.shape[1] != 12 or targets.shape != (inputs.shape[0], 10) or inputs.shape[0] < 1:
        raise ValueError(f"{path} has invalid MLP split shapes.")
    return TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))


def validate_dataset_provenance(dataset_dir: str | Path, timing_pool_path: str | Path, protocol_path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray], Any]:
    dataset_dir = Path(dataset_dir)
    metadata_path = dataset_dir / "dataset_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Synthetic dataset metadata does not exist: {metadata_path}")
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("dataset_metadata.json must be a JSON object.")
    required = {"timing_pool_sha256", "functional_fixture", "timing9_min_ms", "timing9_max_ms", "tr_ms", "vps", "protocol", "sizes"}
    missing = required.difference(metadata)
    if missing:
        raise ValueError(f"dataset_metadata.json lacks: {sorted(missing)}")
    actual_hash = sha256_file(timing_pool_path)
    if metadata["timing_pool_sha256"] != actual_hash:
        raise ValueError("Synthetic dataset timing-pool SHA256 does not match the current --timing-pool.")
    pool = load_timing_pool(timing_pool_path)
    protocol = _protocol(pool, protocol_path)
    record = _protocol_record(protocol)
    if metadata["protocol"] != record or float(metadata["tr_ms"]) != record["tr_ms"] or int(metadata["vps"]) != record["vps"]:
        raise ValueError("Synthetic dataset metadata protocol does not match the current timing pool/protocol YAML.")
    if bool(metadata["functional_fixture"]) != bool(pool["functional_fixture"]):
        raise ValueError("Synthetic dataset functional_fixture flag does not match timing pool provenance.")
    if len(metadata["timing9_min_ms"]) != 9 or len(metadata["timing9_max_ms"]) != 9:
        raise ValueError("Synthetic dataset timing range metadata must have exactly nine dimensions.")
    for split in ("train", "valid", "test"):
        if split not in metadata["sizes"] or int(metadata["sizes"][split]) < 1 or not (dataset_dir / f"{split}.h5").is_file():
            raise ValueError(f"Synthetic dataset lacks a valid declared {split} split.")
    return metadata, pool, protocol


def _check_loaded_split_sizes(metadata: Mapping[str, Any], splits: Mapping[str, TensorDataset]) -> None:
    for name, dataset in splits.items():
        if len(dataset) != int(metadata["sizes"][name]):
            raise ValueError(f"{name}.h5 size does not match dataset_metadata.json.")


def train_mlp(dataset_dir: str | Path, timing_pool_path: str | Path, config: Mapping[str, Any], protocol_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
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
    metadata, pool, teacher_protocol = validate_dataset_provenance(dataset_dir, timing_pool_path, protocol_path)
    splits = {name: load_h5_split(dataset_dir / f"{name}.h5", name) for name in ("train", "valid", "test")}
    _check_loaded_split_sizes(metadata, splits)
    if len(splits["train"]) < batch_size:
        raise ValueError("Training split is smaller than batch_size; refusing a BatchNorm singleton/fallback batch.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"MLP output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    loaders = {
        "train": DataLoader(splits["train"], batch_size=batch_size, shuffle=True, drop_last=True),
        "valid": DataLoader(splits["valid"], batch_size=batch_size, shuffle=False),
    }
    model = MdmSignalMLP().to(device)
    optimizer = Adam(model.parameters(), lr=float(train_cfg["learning_rate"]))
    scheduler = StepLR(optimizer, step_size=int(train_cfg["scheduler_step_size"]), gamma=float(train_cfg["scheduler_gamma"]))
    best, best_epoch = float("inf"), None
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train(); train_total, train_count = 0.0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device); loss = (model(x) - y).pow(2).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite MLP train loss at epoch {epoch}.")
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            train_total += float(loss.detach()) * x.shape[0]; train_count += x.shape[0]
        if train_count == 0:
            raise RuntimeError("No complete BatchNorm training batch was produced.")
        model.eval(); valid_total = 0.0
        with torch.no_grad():
            for x, y in loaders["valid"]:
                valid_total += float((model(x.to(device)) - y.to(device)).pow(2).mean()) * x.shape[0]
        train_loss, valid_loss = train_total / train_count, valid_total / len(splits["valid"])
        scheduler.step(); history.append({"epoch": epoch, "train_mse": train_loss, "valid_mse": valid_loss, "lr": optimizer.param_groups[0]["lr"]})
        if valid_loss < best:
            best, best_epoch = valid_loss, epoch
            torch.save({
                "state_dict": model.state_dict(), "architecture": "12-200-200-200-10",
                "input_normalization": "T1/1000,T2/1000,B1,timing9/1000",
                "output_normalization": "raw_l2_normalized", "protocol_hhz_v1": _protocol_record(teacher_protocol),
                "timing_pool_provenance": str(timing_pool_path), "timing_pool_sha256": metadata["timing_pool_sha256"],
                "timing9_min_ms": metadata["timing9_min_ms"], "timing9_max_ms": metadata["timing9_max_ms"],
                "functional_fixture": bool(metadata["functional_fixture"]), "scientific_checkpoint": not bool(metadata["functional_fixture"]),
                "parameter_ranges": {"t1_ms": [20, 2500], "t2_ms": [5, 200], "b1": [0.1, 1.2], "constraint": "T1>T2"}, "seed": seed,
            }, output / "signal_simulator_best.pth")
    with (output / "train_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_mse", "valid_mse", "lr"]); writer.writeheader(); writer.writerows(history)
    checkpoint = torch.load(output / "signal_simulator_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    metrics = fidelity_metrics(model, splits["test"], pool, str(protocol_path), batch_size, int(train_cfg["gradient_samples"]))
    with (output / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    resolved = dict(config)
    resolved.update({"resolved_protocol": _protocol_record(teacher_protocol), "timing_pool_sha256": metadata["timing_pool_sha256"], "functional_fixture": bool(metadata["functional_fixture"]), "dataset_metadata": metadata, "best_epoch": best_epoch, "best_valid_mse": best})
    with (output / "mlp_config_resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, sort_keys=False)
    return {"model": model, "metrics": metrics, "best_checkpoint": output / "signal_simulator_best.pth"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--dataset-dir", required=True); parser.add_argument("--timing-pool", required=True); parser.add_argument("--protocol", required=True); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("MLP config must be a YAML mapping.")
    result = train_mlp(args.dataset_dir, args.timing_pool, config, args.protocol, args.output_dir)
    print(json.dumps({"best_checkpoint": str(result["best_checkpoint"]), "metrics": result["metrics"]}, sort_keys=True))


if __name__ == "__main__":
    main()
