"""mDM-style Adam training on one-time generated HDF5 HHZ-teacher data."""

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
from torch.utils.data import DataLoader, Dataset
import yaml

from .mlp_model import MdmSignalMLP
from .synthetic_dataset import _protocol, load_timing_pool
from .test_mlp import fidelity_metrics


class H5SignalDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file(): raise FileNotFoundError(f"HDF5 split does not exist: {self.path}")
        with h5py.File(self.path, "r") as handle:
            if set(("input12", "target_signal10")).difference(handle): raise ValueError(f"{self.path} lacks input12/target_signal10.")
            self.length = int(handle["input12"].shape[0])
            if handle["input12"].shape != (self.length, 12) or handle["target_signal10"].shape != (self.length, 10): raise ValueError(f"{self.path} has invalid MLP split shapes.")

    def __len__(self) -> int: return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with h5py.File(self.path, "r") as handle:
            return torch.from_numpy(np.asarray(handle["input12"][index], dtype=np.float32)), torch.from_numpy(np.asarray(handle["target_signal10"][index], dtype=np.float32))


def _require(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping: raise ValueError(f"MLP config lacks required key: {key}")
    return mapping[key]


def train_mlp(dataset_dir: str | Path, timing_pool_path: str | Path, config: Mapping[str, Any], protocol_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    train_cfg = _require(config, "training")
    required = ("device", "seed", "epochs", "batch_size", "learning_rate", "scheduler_step_size", "scheduler_gamma", "gradient_samples")
    missing = [key for key in required if key not in train_cfg]
    if missing: raise ValueError(f"MLP training config lacks: {missing}")
    device = torch.device(train_cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError(f"Requested CUDA unavailable: {device}")
    seed = int(train_cfg["seed"]); torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    dataset_dir, output = Path(dataset_dir), Path(output_dir)
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"MLP output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    train_ds, valid_ds, test_ds = (H5SignalDataset(dataset_dir / f"{split}.h5") for split in ("train", "valid", "test"))
    loaders = {"train": DataLoader(train_ds, batch_size=int(train_cfg["batch_size"]), shuffle=True), "valid": DataLoader(valid_ds, batch_size=int(train_cfg["batch_size"]), shuffle=False)}
    model = MdmSignalMLP().to(device); optimizer = Adam(model.parameters(), lr=float(train_cfg["learning_rate"])); scheduler = StepLR(optimizer, step_size=int(train_cfg["scheduler_step_size"]), gamma=float(train_cfg["scheduler_gamma"]))
    pool = load_timing_pool(timing_pool_path); pool["protocol_path"] = str(protocol_path)
    teacher_protocol = _protocol(pool, protocol_path)
    best = float("inf"); history: list[dict[str, Any]] = []
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train(); train_loss = 0.0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device); loss = (model(x) - y).pow(2).mean()
            if not torch.isfinite(loss): raise FloatingPointError(f"Non-finite MLP train loss at epoch {epoch}.")
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); train_loss += float(loss.detach()) * x.shape[0]
        model.eval(); valid_total = 0.0
        with torch.no_grad():
            for x, y in loaders["valid"]:
                valid_total += float((model(x.to(device)) - y.to(device)).pow(2).mean()) * x.shape[0]
        train_loss /= len(train_ds); valid_loss = valid_total / len(valid_ds); scheduler.step()
        history.append({"epoch": epoch, "train_mse": train_loss, "valid_mse": valid_loss, "lr": optimizer.param_groups[0]["lr"]})
        if valid_loss < best:
            best = valid_loss
            torch.save({"state_dict": model.state_dict(), "architecture": "12-200-200-200-10", "input_normalization": "T1/1000,T2/1000,B1,timing9/1000", "output_normalization": "raw_l2_normalized", "protocol_hhz_v1": {"tr_ms": teacher_protocol.tr_ms, "vps": teacher_protocol.vps, "fa_deg": list(teacher_protocol.fa_deg), "ti_ms": list(teacher_protocol.ti_ms), "t2prep_ms": list(teacher_protocol.t2prep_ms), "n_ramp_up": teacher_protocol.n_ramp_up}, "timing_pool_provenance": str(timing_pool_path), "parameter_ranges": {"t1_ms": [20, 2500], "t2_ms": [5, 200], "b1": [0.1, 1.2], "constraint": "T1>T2"}, "seed": seed}, output / "signal_simulator_best.pth")
    with (output / "train_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_mse", "valid_mse", "lr"]); writer.writeheader(); writer.writerows(history)
    checkpoint = torch.load(output / "signal_simulator_best.pth", map_location=device, weights_only=False); model.load_state_dict(checkpoint["state_dict"]); model.eval()
    metrics = fidelity_metrics(model, dataset_dir / "test.h5", pool, int(train_cfg["batch_size"]), int(train_cfg["gradient_samples"]))
    with (output / "test_metrics.json").open("w", encoding="utf-8") as handle: json.dump(metrics, handle, indent=2)
    with (output / "mlp_config_resolved.yaml").open("w", encoding="utf-8") as handle: yaml.safe_dump(dict(config), handle, sort_keys=False)
    return {"model": model, "metrics": metrics, "best_checkpoint": output / "signal_simulator_best.pth"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--dataset-dir", required=True); parser.add_argument("--timing-pool", required=True); parser.add_argument("--protocol", required=True); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as handle: config = yaml.safe_load(handle)
    if not isinstance(config, dict): raise ValueError("MLP config must be a YAML mapping.")
    result = train_mlp(args.dataset_dir, args.timing_pool, config, args.protocol, args.output_dir)
    print(json.dumps({"best_checkpoint": str(result["best_checkpoint"]), "metrics": result["metrics"]}, sort_keys=True))


if __name__ == "__main__": main()
