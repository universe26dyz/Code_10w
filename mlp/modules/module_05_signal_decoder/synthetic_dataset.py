"""Generate the active rhythm-disjoint HHZ/VPS=87 surrogate dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch
import yaml

from .mlp_model import make_input12
from .synthetic_rr_domain import RRDomainConfig, generate_rhythm_timing9, split_rhythm_ids
from .trad_teacher.trad_signal_simulator import TradProtocol, TradSignalSimulator


def protocol_from_record(record: Mapping[str, Any], protocol_path: str | Path, *, require_hhz_vps87: bool = True) -> TradProtocol:
    """Validate a dataset/checkpoint protocol against authoritative HHZ YAML."""

    with Path(protocol_path).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    protocol = TradProtocol(
        float(record["tr_ms"]), int(record["vps"]),
        tuple(float(x) for x in record["fa_deg"]), tuple(float(x) for x in record["ti_ms"]),
        tuple(float(x) for x in record["t2prep_ms"]), int(record["n_ramp_up"]),
    )
    protocol.validate()
    expected = {
        "tr_ms": 2.61, "vps": 87,
        "fa_deg": tuple(float(x) for x in cfg["flip_angle_degrees"]),
        "ti_ms": tuple(float(x) for x in cfg["inversion_times_ms"]),
        "t2prep_ms": tuple(float(x) for x in cfg["t2prep_ms"]),
        "n_ramp_up": int(cfg["ramp_up_pulses"]),
    }
    actual = {
        "tr_ms": protocol.tr_ms, "vps": protocol.vps, "fa_deg": protocol.fa_deg,
        "ti_ms": protocol.ti_ms, "t2prep_ms": protocol.t2prep_ms, "n_ramp_up": protocol.n_ramp_up,
    }
    if require_hhz_vps87 and actual != expected:
        raise ValueError("Dataset protocol does not equal the authoritative fixed HHZ VPS=87 contract.")
    return protocol


def _teacher_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested teacher_device is unavailable: {device}")
    return device


def _teacher_signal_and_jacobian(teacher: TradSignalSimulator, t1: torch.Tensor, t2: torch.Tensor, b1: torch.Tensor, timing: torch.Tensor, protocol: TradProtocol) -> tuple[torch.Tensor, torch.Tensor]:
    """Offline Bloch targets; Jacobian is d(signal)/d(T1/1000,T2/1000,B1)."""

    t1, t2, b1 = (value.detach().clone().requires_grad_(True) for value in (t1, t2, b1))
    signal = teacher(t1, t2, b1, timing, protocol, normalize=True)
    columns = []
    for weight in range(10):
        gradients = torch.autograd.grad(signal[:, weight].sum(), (t1, t2, b1), retain_graph=True)
        columns.append(torch.stack((gradients[0] * 1000.0, gradients[1] * 1000.0, gradients[2]), dim=-1))
    return signal.detach(), torch.stack(columns, dim=1).detach()


def _teacher_signal(teacher: TradSignalSimulator, t1: torch.Tensor, t2: torch.Tensor, b1: torch.Tensor, timing: torch.Tensor, protocol: TradProtocol) -> torch.Tensor:
    with torch.no_grad():
        return teacher(t1, t2, b1, timing, protocol, normalize=True).detach()


def _fixed_rr_protocol() -> TradProtocol:
    protocol = TradProtocol(2.61, 87, (45.0, 45.0, 45.0), (50.0, 150.0), (35.0, 45.0, 55.0), 10)
    protocol.validate()
    return protocol


def _generate_rr_split(path: Path, rhythm_ids: np.ndarray, rhythms: Mapping[str, np.ndarray], *, tissue_samples_per_rhythm: int, seed: int, chunk_size: int, device: torch.device, store_jacobian: bool) -> int:
    if tissue_samples_per_rhythm < 1 or chunk_size < 1:
        raise ValueError("tissue_samples_per_rhythm and teacher_chunk_size must be positive.")
    n_samples = int(rhythm_ids.size) * tissue_samples_per_rhythm
    rng, teacher, protocol = np.random.default_rng(seed), TradSignalSimulator(), _fixed_rr_protocol()
    with h5py.File(path, "w") as handle:
        input_set = handle.create_dataset("input12", shape=(n_samples, 12), dtype="f4")
        target_set = handle.create_dataset("target_signal10", shape=(n_samples, 10), dtype="f4")
        rhythm_set = handle.create_dataset("rhythm_id", shape=(n_samples,), dtype="i8")
        jacobian_set = handle.create_dataset("target_jacobian10x3", shape=(n_samples, 10, 3), dtype="f4") if store_jacobian else None
        if jacobian_set is not None:
            jacobian_set.attrs["parameter_space"] = "T1/1000,T2/1000,B1"
            jacobian_set.attrs["units"] = "normalized_signal_per_normalized_parameter"
        expanded_ids = np.repeat(np.asarray(rhythm_ids, dtype=np.int64), tissue_samples_per_rhythm)
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            ids = expanded_ids[start:end]
            count = end - start
            t2 = rng.uniform(5.0, 200.0, count).astype(np.float32)
            t1 = rng.uniform(np.maximum(20.0, t2 + np.finfo(np.float32).eps), 2500.0).astype(np.float32)
            b1 = rng.uniform(0.1, 1.2, count).astype(np.float32)
            timing = np.asarray(rhythms["timing9_ms"][ids], dtype=np.float32)
            t1_t, t2_t, b1_t, timing_t = (torch.from_numpy(value) for value in (t1, t2, b1, timing))
            if jacobian_set is None:
                target = _teacher_signal(teacher, t1_t.to(device), t2_t.to(device), b1_t.to(device), timing_t.to(device), protocol)
            else:
                target, jacobian = _teacher_signal_and_jacobian(teacher, t1_t.to(device), t2_t.to(device), b1_t.to(device), timing_t.to(device), protocol)
                jacobian_set[start:end] = jacobian.cpu().numpy().astype(np.float32)
            input_set[start:end] = make_input12(t1_t, t2_t, b1_t, timing_t).numpy().astype(np.float32)
            target_set[start:end] = target.cpu().numpy().astype(np.float32)
            rhythm_set[start:end] = ids
    return n_samples


def generate_rr_synthetic_dataset(output_dir: str | Path, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Generate a non-overwriting, rhythm-disjoint broad-RR surrogate dataset."""

    required = {"n_rhythms", "tissue_samples_per_rhythm", "split_rhythms", "seed", "store_jacobian", "teacher_chunk_size", "teacher_device"}
    missing = required.difference(settings)
    if missing:
        raise ValueError(f"RR synthetic settings lack: {sorted(missing)}")
    protocol = _fixed_rr_protocol()
    split_counts = settings["split_rhythms"]
    if not isinstance(split_counts, Mapping):
        raise ValueError("split_rhythms must be a train/valid/test mapping.")
    config = RRDomainConfig(
        n_rhythms=int(settings["n_rhythms"]), seed=int(settings["seed"]), tr_ms=protocol.tr_ms,
        vps=protocol.vps, ti_ms=tuple(protocol.ti_ms), t2prep_ms=tuple(protocol.t2prep_ms),
    )
    rhythms = generate_rhythm_timing9(config)
    splits = split_rhythm_ids(config.n_rhythms, seed=config.seed, counts={key: int(value) for key, value in split_counts.items()})
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"RR synthetic output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = _teacher_device(str(settings["teacher_device"]))
    sizes = {
        name: _generate_rr_split(
            output / f"{name}.h5", ids, rhythms, tissue_samples_per_rhythm=int(settings["tissue_samples_per_rhythm"]),
            seed=config.seed + offset, chunk_size=int(settings["teacher_chunk_size"]), device=device,
            store_jacobian=bool(settings["store_jacobian"]),
        )
        for offset, (name, ids) in enumerate(splits.items())
    }
    np.savez_compressed(output / "rhythms.npz", **rhythms)
    timing_min, timing_max = rhythms["timing9_ms"].min(0).tolist(), rhythms["timing9_ms"].max(0).tolist()
    metadata = {
        "schema": "mlp_rr_synthetic/v1", "split_mode": "rhythm", "store_jacobian": bool(settings["store_jacobian"]),
        "rhythm_split_ids": {name: ids.tolist() for name, ids in splits.items()}, "sizes": sizes,
        "protocol": {"tr_ms": protocol.tr_ms, "vps": protocol.vps, "fa_deg": list(protocol.fa_deg), "ti_ms": list(protocol.ti_ms), "t2prep_ms": list(protocol.t2prep_ms), "n_ramp_up": protocol.n_ramp_up},
        "rr_domain": {"hr_mean_bpm": [20.0, 120.0], "rr_cv": [0.0, 0.50], "generator": "mDM-inspired coherent normal RR histories; not byte-for-byte upstream mDM"},
        "parameter_ranges": {"t1_ms": [20, 2500], "t2_ms": [5, 200], "b1": [0.1, 1.2], "constraint": "T1>T2"},
        "timing9_min_ms": timing_min, "timing9_max_ms": timing_max,
        "train_timing9_min_ms": timing_min, "train_timing9_max_ms": timing_max,
        "seed": config.seed, "teacher_device": str(device),
    }
    (output / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
