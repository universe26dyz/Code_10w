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
from .provenance import sha256_file
from .trad_teacher.trad_signal_simulator import TradProtocol, TradSignalSimulator


FORMAL_MLP_SUBJECTS = ("CYJ", "DYZ", "HHZ", "HJL")
FORMAL_MLP_SPLIT = {"train_subjects": ["CYJ", "DYZ"], "valid_subjects": ["HHZ"], "test_subjects": ["HJL"]}


def validate_formal_mlp_subject_split(subject_ids: np.ndarray, split: Mapping[str, Any]) -> dict[str, list[str]]:
    """Enforce the fixed VPS=87 formal MLP pool; DYL is Trad-only here."""

    observed = set(np.asarray(subject_ids).astype(str).tolist())
    if "DYL" in observed:
        raise ValueError("DYL is excluded from the formal MLP pool (VPS=85; Trad/Bloch only).")
    expected = set(FORMAL_MLP_SUBJECTS)
    if observed != expected:
        raise ValueError(f"Formal MLP pool must be exactly {sorted(expected)}, got {sorted(observed)}.")
    actual = {key: [str(value) for value in split.get(key, [])] for key in FORMAL_MLP_SPLIT}
    if actual != FORMAL_MLP_SPLIT:
        raise ValueError("Formal MLP split is fixed: CYJ/DYZ -> HHZ -> HJL.")
    return {key.replace("_subjects", ""): list(value) for key, value in FORMAL_MLP_SPLIT.items()}


def load_timing_pool(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Timing pool does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"timing9_ms", "tr_ms", "vps", "source_id", "group_id", "functional_fixture"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Timing pool lacks: {sorted(missing)}")
        result = {key: np.asarray(data[key]) for key in required}
        for key in ("subject_id", "stack", "stack_idx"):
            if key in data.files: result[key] = np.asarray(data[key])
    if result["timing9_ms"].ndim != 2 or result["timing9_ms"].shape[1] != 9 or result["timing9_ms"].shape[0] < 3:
        raise ValueError("Timing pool must contain at least three unique [N,9] timing vectors for train/val/test timing splits.")
    if result["functional_fixture"].size != 1:
        raise ValueError("Timing pool functional_fixture must be one explicit boolean.")
    result["functional_fixture"] = np.asarray(bool(result["functional_fixture"].reshape(-1)[0]))
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


def split_subject_ids(subject_ids: np.ndarray, split: Mapping[str, Any]) -> dict[str, list[str]]:
    if split.get("mode") != "subject": raise ValueError("Formal split.mode must be 'subject'.")
    available = set(np.asarray(subject_ids).astype(str).tolist())
    result = {name: [str(value) for value in split.get(f"{name}_subjects", [])] for name in ("train", "valid", "test")}
    if any(not result[name] for name in result) or any(len(values) != len(set(values)) for values in result.values()) or set().union(*map(set, result.values())) != available or any(set(result[a]).intersection(result[b]) for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))):
        raise ValueError("Formal subject split must explicitly partition every pool subject into non-empty disjoint train/valid/test sets.")
    return result


def timing_domain_metadata(timing: np.ndarray, subject_ids: np.ndarray, split_subjects: Mapping[str, list[str]]) -> dict[str, Any]:
    timing, subject_ids = np.asarray(timing, dtype=np.float64), np.asarray(subject_ids).astype(str)
    train = timing[np.isin(subject_ids, split_subjects["train"])]
    test = timing[np.isin(subject_ids, split_subjects["test"])]
    lower, upper = train.min(0), train.max(0)
    relation = "interpolation_to_train_domain" if np.all((test >= lower) & (test <= upper)) else "extrapolation_from_train_domain"
    return {"train_timing9_min_ms": lower.tolist(), "train_timing9_max_ms": upper.tolist(), "pool_timing9_min_ms": timing.min(0).tolist(), "pool_timing9_max_ms": timing.max(0).tolist(), "test_timing_relation_to_train": relation}


def _teacher_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested teacher_device is unavailable: {device}")
    return device


def _teacher_signal_and_jacobian(teacher: TradSignalSimulator, t1: torch.Tensor, t2: torch.Tensor, b1: torch.Tensor, timing: torch.Tensor, protocol: TradProtocol) -> tuple[torch.Tensor, torch.Tensor]:
    """Offline Bloch targets; Jacobian is d(normalized signal)/d(T1/1000,T2/1000,B1)."""

    t1, t2, b1 = (value.detach().clone().requires_grad_(True) for value in (t1, t2, b1))
    signal = teacher(t1, t2, b1, timing, protocol, normalize=True)
    columns = []
    for weight in range(10):
        gradients = torch.autograd.grad(signal[:, weight].sum(), (t1, t2, b1), retain_graph=True)[0:3]
        columns.append(torch.stack((gradients[0] * 1000.0, gradients[1] * 1000.0, gradients[2]), dim=-1))
    return signal.detach(), torch.stack(columns, dim=1).detach()


def _generate_split(path: Path, n_samples: int, timing_ids: np.ndarray, pool: Mapping[str, np.ndarray], protocol: TradProtocol, seed: int, chunk_size: int, teacher_device: torch.device) -> None:
    if n_samples < 1 or timing_ids.size < 1 or chunk_size < 1:
        raise ValueError("Dataset split size, timing IDs, and chunk_size must be positive.")
    rng = np.random.default_rng(seed)
    teacher = TradSignalSimulator()
    with h5py.File(path, "w") as handle:
        input_set = handle.create_dataset("input12", shape=(n_samples, 12), dtype="f4")
        target_set = handle.create_dataset("target_signal10", shape=(n_samples, 10), dtype="f4")
        jacobian_set = handle.create_dataset("target_jacobian10x3", shape=(n_samples, 10, 3), dtype="f4")
        jacobian_set.attrs["parameter_space"] = "T1/1000,T2/1000,B1"
        jacobian_set.attrs["units"] = "normalized_signal_per_normalized_parameter"
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
            target, jacobian = _teacher_signal_and_jacobian(teacher, t1_t.to(teacher_device), t2_t.to(teacher_device), b1_t.to(teacher_device), timing_t.to(teacher_device), protocol)
            input_set[start:end] = make_input12(t1_t, t2_t, b1_t, timing_t).numpy().astype(np.float32)
            target_set[start:end] = target.cpu().numpy().astype(np.float32)
            jacobian_set[start:end] = jacobian.cpu().numpy().astype(np.float32)
            timing_set[start:end] = chosen


def generate_mlp_dataset(timing_pool_path: str | Path, protocol_path: str | Path, output_dir: str | Path, sizes: Mapping[str, Any], seed: int, chunk_size: int, teacher_device: str = "cpu") -> dict[str, Any]:
    pool = load_timing_pool(timing_pool_path)
    required = ("train", "valid", "test")
    if any(key not in sizes for key in required):
        raise ValueError("Dataset sizes must explicitly include train, valid, and test.")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Synthetic dataset output must be absent or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if not isinstance(teacher_device, str):
        raise ValueError("teacher_device must be an explicit device string.")
    protocol = _protocol(pool, protocol_path)
    device = _teacher_device(teacher_device)
    split_cfg = sizes.get("split")
    if not isinstance(split_cfg, Mapping) or "mode" not in split_cfg: raise ValueError("dataset.split with explicit mode is required.")
    if split_cfg["mode"] == "subject":
        if "subject_id" not in pool: raise ValueError("Formal subject split requires timing-pool subject_id provenance.")
        subject_splits = validate_formal_mlp_subject_split(pool["subject_id"], split_cfg) if bool(split_cfg.get("formal_mlp", False)) else split_subject_ids(pool["subject_id"], split_cfg)
        splits = {name: np.flatnonzero(np.isin(pool["subject_id"].astype(str), subjects)) for name, subjects in subject_splits.items()}
        domain = timing_domain_metadata(pool["timing9_ms"], pool["subject_id"], subject_splits)
    elif split_cfg["mode"] == "timing":
        splits, subject_splits, domain = split_timing_ids(pool["timing9_ms"].shape[0], seed), {"train": [], "valid": [], "test": []}, {"train_timing9_min_ms": [], "train_timing9_max_ms": [], "pool_timing9_min_ms": np.asarray(pool["timing9_ms"]).min(0).tolist(), "pool_timing9_max_ms": np.asarray(pool["timing9_ms"]).max(0).tolist(), "test_timing_relation_to_train": "not_subject_validation"}
    else: raise ValueError("dataset.split.mode must be 'timing' or 'subject'.")
    for offset, split in enumerate(required):
        _generate_split(output / f"{split}.h5", int(sizes[split]), splits[split], pool, protocol, seed + offset, chunk_size, device)
    timing = np.asarray(pool["timing9_ms"], dtype=np.float64)
    metadata = {"timing_pool": str(Path(timing_pool_path)), "timing_pool_sha256": sha256_file(timing_pool_path), "functional_fixture": bool(pool["functional_fixture"]), "timing9_min_ms": timing.min(axis=0).tolist(), "timing9_max_ms": timing.max(axis=0).tolist(), "tr_ms": protocol.tr_ms, "vps": protocol.vps, "split_mode": split_cfg["mode"], "formal_mlp_split": bool(split_cfg.get("formal_mlp", False)), "train_subject_ids": subject_splits["train"], "valid_subject_ids": subject_splits["valid"], "test_subject_ids": subject_splits["test"], **domain, "timing_split_ids": {key: value.tolist() for key, value in splits.items()}, "protocol": {"tr_ms": protocol.tr_ms, "vps": protocol.vps, "fa_deg": list(protocol.fa_deg), "ti_ms": list(protocol.ti_ms), "t2prep_ms": list(protocol.t2prep_ms), "n_ramp_up": protocol.n_ramp_up}, "sizes": {key: int(sizes[key]) for key in required}, "seed": int(seed), "teacher_device": str(device), "jacobian_target": "target_jacobian10x3", "jacobian_target_units": "normalized_signal_per_normalized_T1_T2_B1"}
    with (output / "dataset_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata
