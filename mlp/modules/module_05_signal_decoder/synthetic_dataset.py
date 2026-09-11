"""One-time HDF5 generation of continuous HHZ-teacher samples split by timing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch
import yaml

from .mlp_model import make_input12
from .trad_teacher.trad_signal_simulator import TradProtocol, TradSignalSimulator


def load_timing_pool(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Timing pool does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"timing9_ms", "tr_ms", "vps", "source_id", "group_id"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Timing pool lacks: {sorted(missing)}")
        result = {key: np.asarray(data[key]) for key in required}
    if result["timing9_ms"].ndim != 2 or result["timing9_ms"].shape[1] != 9 or result["timing9_ms"].shape[0] < 3:
        raise ValueError("Timing pool must contain at least three unique [N,9] timing vectors for train/val/test timing splits.")
    if float(result["tr_ms"].reshape(-1)[0]) <= 0 or int(result["vps"].reshape(-1)[0]) < 16:
        raise ValueError("Timing pool has invalid TR/VPS.")
    return result


def make_functional_timing_fixture(source_pool_path: str | Path, output_path: str | Path) -> None:
    """Create an explicitly labeled 3-vector fixture when a one-group smoke pool cannot split.

    This fixture is only a training-code path check, never a formal timing
    augmentation or a cross-timing generalization claim.
    """

    source = Path(source_pool_path)
    with np.load(source, allow_pickle=False) as data:
        timing = np.asarray(data["timing9_ms"], dtype=np.float64)
        tr_ms, vps = float(np.asarray(data["tr_ms"]).reshape(-1)[0]), int(np.asarray(data["vps"]).reshape(-1)[0])
    if timing.shape != (1, 9):
        raise ValueError("Functional fixture is allowed only from one real tiny timing vector.")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Functional timing fixture already exists: {output}")
    perturbations = np.asarray([[0.0] * 9, [2.0, -1.0, 1.0, 0.0, 2.0, -1.0, 1.0, 0.0, 2.0], [-2.0, 1.0, 0.0, 2.0, -1.0, 1.0, 0.0, 2.0, -1.0]])
    np.savez_compressed(output, timing9_ms=timing[0][None] + perturbations, tr_ms=np.asarray(tr_ms), vps=np.asarray(vps, dtype=np.int64), source_id=np.asarray(["FUNCTIONAL_FIXTURE_FROM_" + str(source)] * 3), group_id=np.asarray([0, 1, 2], dtype=np.int64), stack_idx=np.asarray([-1, -1, -1], dtype=np.int64), functional_fixture=np.asarray(True))


def _protocol(pool: Mapping[str, np.ndarray], protocol_path: str | Path) -> TradProtocol:
    with Path(protocol_path).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    required = ("flip_angle_degrees", "inversion_times_ms", "t2prep_ms", "ramp_up_pulses")
    if not isinstance(cfg, dict) or any(key not in cfg for key in required):
        raise ValueError("Protocol YAML lacks fixed HHZ values.")
    protocol = TradProtocol(float(pool["tr_ms"].reshape(-1)[0]), int(pool["vps"].reshape(-1)[0]), tuple(float(x) for x in cfg["flip_angle_degrees"]), tuple(float(x) for x in cfg["inversion_times_ms"]), tuple(float(x) for x in cfg["t2prep_ms"]), int(cfg["ramp_up_pulses"]))
    protocol.validate()
    return protocol


def split_timing_ids(n_timing: int, seed: int) -> dict[str, np.ndarray]:
    if n_timing < 3:
        raise ValueError("At least three unique timing vectors are required for timing-group train/val/test split.")
    order = np.random.default_rng(seed).permutation(n_timing)
    n_train = max(1, int(round(n_timing * 0.6)))
    n_val = max(1, int(round(n_timing * 0.2)))
    if n_train + n_val >= n_timing:
        n_train, n_val = n_timing - 2, 1
    return {"train": order[:n_train], "valid": order[n_train:n_train + n_val], "test": order[n_train + n_val:]}


def _generate_split(path: Path, n_samples: int, timing_ids: np.ndarray, pool: Mapping[str, np.ndarray], protocol: TradProtocol, seed: int, chunk_size: int) -> None:
    if n_samples < 1 or timing_ids.size < 1 or chunk_size < 1:
        raise ValueError("Dataset split size, timing IDs, and chunk_size must be positive.")
    rng = np.random.default_rng(seed)
    teacher = TradSignalSimulator()
    with h5py.File(path, "w") as handle:
        input_set = handle.create_dataset("input12", shape=(n_samples, 12), dtype="f4")
        target_set = handle.create_dataset("target_signal10", shape=(n_samples, 10), dtype="f4")
        timing_set = handle.create_dataset("timing_id", shape=(n_samples,), dtype="i8")
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            count = end - start
            t2 = rng.uniform(5.0, 200.0, count).astype(np.float32)
            t1 = rng.uniform(np.maximum(20.0, t2 + np.finfo(np.float32).eps), 2500.0).astype(np.float32)
            b1 = rng.uniform(0.1, 1.2, count).astype(np.float32)
            chosen = rng.choice(timing_ids, size=count, replace=True)
            timing = np.asarray(pool["timing9_ms"][chosen], dtype=np.float32)
            t1_t, t2_t, b1_t, timing_t = (torch.from_numpy(value) for value in (t1, t2, b1, timing))
            with torch.no_grad():
                target = teacher(t1_t, t2_t, b1_t, timing_t, protocol, normalize=True).numpy().astype(np.float32)
            input_set[start:end] = make_input12(t1_t, t2_t, b1_t, timing_t).numpy().astype(np.float32)
            target_set[start:end] = target
            timing_set[start:end] = chosen


def generate_mlp_dataset(timing_pool_path: str | Path, protocol_path: str | Path, output_dir: str | Path, sizes: Mapping[str, Any], seed: int, chunk_size: int) -> dict[str, Any]:
    pool = load_timing_pool(timing_pool_path)
    required = ("train", "valid", "test")
    if any(key not in sizes for key in required):
        raise ValueError("Dataset sizes must explicitly include train, valid, and test.")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Synthetic dataset output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    protocol = _protocol(pool, protocol_path)
    splits = split_timing_ids(pool["timing9_ms"].shape[0], seed)
    for offset, split in enumerate(required):
        _generate_split(output / f"{split}.h5", int(sizes[split]), splits[split], pool, protocol, seed + offset, chunk_size)
    metadata = {"timing_pool": str(Path(timing_pool_path)), "timing_split_ids": {key: value.tolist() for key, value in splits.items()}, "protocol": {"tr_ms": protocol.tr_ms, "vps": protocol.vps, "fa_deg": list(protocol.fa_deg), "ti_ms": list(protocol.ti_ms), "t2prep_ms": list(protocol.t2prep_ms), "n_ramp_up": protocol.n_ramp_up}, "sizes": {key: int(sizes[key]) for key in required}, "seed": int(seed)}
    with (output / "dataset_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata
