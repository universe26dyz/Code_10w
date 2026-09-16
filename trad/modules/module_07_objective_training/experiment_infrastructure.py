"""Reproducibility, monitoring, sampling, and profiling utilities for reconstruction."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch


PROFILE_SECTIONS = (
    "batch_sampling", "coordinate_conversion", "psf_and_rigid", "inr_forward",
    "signal_decoder_forward", "data_loss", "spatial_regularization", "backward",
    "optimizer_step", "whole_iteration",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(("git", *args), cwd=repo_root, check=True, capture_output=True, text=True).stdout.strip()


def git_provenance(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    status = _git(root, "status", "--porcelain")
    dirty = bool(status)
    result: dict[str, Any] = {
        "git_branch": _git(root, "branch", "--show-current") or None,
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_dirty": dirty,
        "git_status_porcelain": status or None,
        "git_diff_sha256": None,
        "git_diff_patch": None,
    }
    if dirty:
        patch = _git(root, "diff", "--binary", "HEAD")
        result["_diff_bytes"] = patch.encode("utf-8")
        result["git_diff_sha256"] = hashlib.sha256(result["_diff_bytes"]).hexdigest()
        result["git_diff_patch"] = "code_diff.patch"
    return result


def _vendored_commit(method_root: Path) -> str | None:
    candidate = method_root / "third_party" / "nesvor"
    if not candidate.is_dir():
        return None
    try:
        return _git(candidate, "rev-parse", "HEAD")
    except subprocess.CalledProcessError:
        return None


def normalize_step1_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Add explicit disabled/default Step-1 controls and reject unavailable modes."""

    resolved = copy.deepcopy(dict(config))
    training = resolved.setdefault("training", {})
    if not isinstance(training, dict):
        raise ValueError("training must be a mapping.")
    samples = int(training.get("psf_samples", 8))
    if samples < 1:
        raise ValueError("training.psf_samples must be positive.")
    training.setdefault("checkpoint_every", 0)
    training.setdefault("require_clean_git", False)
    training.setdefault("profile_every", 100)
    training.setdefault("profile_warmup_samples", 2)
    training.setdefault("monitor_every", 500)
    training.setdefault("monitor_samples_per_weight_per_stack", 8)
    if int(training["checkpoint_every"]) < 0 or int(training["profile_every"]) < 0 or int(training["monitor_every"]) < 0:
        raise ValueError("checkpoint_every, profile_every, and monitor_every must be non-negative.")
    psf = resolved.setdefault("psf", {})
    if not isinstance(psf, dict):
        raise ValueError("psf must be a mapping.")
    psf.setdefault("training", {"enabled": True, "type": "gaussian", "n_samples": samples, "sampling_mode": "random"})
    psf.setdefault("export", {"enabled": False, "n_samples": 128, "output_psf_factor": 1.0})
    train_psf, export_psf = psf["training"], psf["export"]
    if not isinstance(train_psf, dict) or not isinstance(export_psf, dict):
        raise ValueError("psf.training and psf.export must be mappings.")
    expected_train = {"enabled": True, "type": "gaussian", "n_samples": samples, "sampling_mode": "random"}
    for key, expected in expected_train.items():
        if train_psf.get(key, expected) != expected:
            raise ValueError(f"Unsupported Step-1 psf.training.{key}={train_psf.get(key)!r}; baseline requires {expected!r}.")
        train_psf.setdefault(key, expected)
    if export_psf.get("enabled", False):
        raise ValueError("PSF export is not implemented in Step 1; set psf.export.enabled=false.")
    export_psf.setdefault("enabled", False); export_psf.setdefault("n_samples", 128); export_psf.setdefault("output_psf_factor", 1.0)
    variance = resolved.setdefault("variance", {"enabled": False, "pixel": False, "slice": False})
    if not isinstance(variance, dict):
        raise ValueError("variance must be a mapping.")
    for key in ("enabled", "pixel", "slice"):
        variance.setdefault(key, False)
    if any(bool(variance[key]) for key in ("enabled", "pixel", "slice")):
        raise ValueError("variance is not implemented in Step 1; enabled, pixel, and slice must all be false.")
    regularization = resolved.setdefault("spatial_regularization", {"mode": "current_l2"})
    if not isinstance(regularization, dict) or regularization.get("mode", "current_l2") != "current_l2":
        raise ValueError("Only spatial_regularization.mode=current_l2 is available in Step 1.")
    regularization.setdefault("mode", "current_l2")
    return resolved


class CachedBalancedSampler:
    """Cache candidate pixels for the existing 10-weight balanced sampling law."""

    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset
        self.candidate_indices_by_weight = tuple(torch.nonzero(dataset.weight_idx == weight, as_tuple=False).flatten() for weight in range(10))
        if any(indices.numel() == 0 for indices in self.candidate_indices_by_weight):
            raise ValueError("Every observed weight 0..9 must have at least one candidate pixel.")

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if batch_size < 10 or batch_size % 10:
            raise ValueError("batch_size must be a positive multiple of 10 for explicit 10-weight balance.")
        per_weight = batch_size // 10
        index = torch.cat([candidates[torch.randint(candidates.numel(), (per_weight,), device=candidates.device)] for candidates in self.candidate_indices_by_weight])
        index = index[torch.randperm(index.numel(), device=index.device)]
        return {name: getattr(self.dataset, name)[index] for name in ("xyz", "v", "group_idx", "weight_idx", "stack_idx", "timing")}


class FixedMonitorSet:
    """A seed-stable, 10-weight balanced evaluation set covering each present stack."""

    def __init__(self, batch: Mapping[str, torch.Tensor]) -> None:
        self.batch = dict(batch)

    @classmethod
    def from_dataset(cls, dataset: Any, *, seed: int, samples_per_weight_per_stack: int) -> "FixedMonitorSet":
        if samples_per_weight_per_stack < 1:
            raise ValueError("monitor_samples_per_weight_per_stack must be positive.")
        generator = np.random.default_rng(seed)
        indices: list[torch.Tensor] = []
        stacks = sorted(int(value) for value in dataset.stack_idx.unique().detach().cpu().tolist())
        for weight in range(10):
            for stack in stacks:
                candidates = torch.nonzero((dataset.weight_idx == weight) & (dataset.stack_idx == stack), as_tuple=False).flatten()
                if candidates.numel() == 0:
                    raise ValueError(f"Monitor set cannot balance absent weight={weight}, stack={stack}.")
                chosen = generator.choice(candidates.detach().cpu().numpy(), size=samples_per_weight_per_stack, replace=candidates.numel() < samples_per_weight_per_stack)
                indices.append(torch.as_tensor(chosen, dtype=torch.long, device=candidates.device))
        index = torch.cat(indices)
        return cls({name: getattr(dataset, name)[index] for name in ("xyz", "v", "group_idx", "weight_idx", "stack_idx", "timing")})


class IterationProfiler:
    """Low-overhead sampled timer; CUDA events are resolved once when CSV is written."""

    def __init__(self, device: torch.device, *, every: int, warmup_samples: int) -> None:
        self.device, self.every, self.warmup_samples = device, every, warmup_samples
        self._active = False; self._sample_count = 0; self._iteration = 0; self._stage = ""; self._records: list[tuple[int, int, str, str, Any, Any]] = []

    def begin(self, iteration: int, stage: str) -> None:
        self._active = self.every > 0 and iteration % self.every == 0
        self._iteration, self._stage = iteration, stage
        if self._active:
            self._sample_count += 1
            self._whole_start = self._now()

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        if not self._active:
            with nullcontext():
                yield
            return
        start = self._now()
        yield
        self._records.append((self._sample_count, self._iteration, self._stage, name, start, self._now()))

    def end(self) -> None:
        if self._active:
            self._records.append((self._sample_count, self._iteration, self._stage, "whole_iteration", self._whole_start, self._now()))
        self._active = False

    def _now(self) -> Any:
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True); event.record(); return event
        return time.perf_counter()

    def write_csv(self, path: str | Path) -> None:
        if self.device.type == "cuda" and self._records:
            torch.cuda.synchronize(self.device)
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("iteration", "stage", "section", "milliseconds")); writer.writeheader()
            for sample_number, iteration, stage, section, start, end in self._records:
                if sample_number <= self.warmup_samples:
                    continue
                milliseconds = start.elapsed_time(end) if self.device.type == "cuda" else (end - start) * 1000.0
                writer.writerow({"iteration": iteration, "stage": stage, "section": section, "milliseconds": milliseconds})


def write_experiment_manifest(output_dir: str | Path, *, route: str, subject_id: str | None, repo_root: str | Path, config_resolved: str | Path, prepared_inputs: list[str | Path], protocol: Mapping[str, Any], stack_group_counts: Mapping[str, int], seed: int, command: str, method_root: str | Path | None = None) -> dict[str, Any]:
    """Write non-sensitive provenance, including a recoverable patch for a dirty tree."""

    output, root = Path(output_dir), Path(repo_root)
    git = git_provenance(root)
    diff = git.pop("_diff_bytes", None)
    if diff is not None:
        (output / "code_diff.patch").write_bytes(diff)
    method = Path(method_root) if method_root is not None else root
    prepared = [Path(value) for value in prepared_inputs]
    cuda = torch.version.cuda
    try:
        import tinycudann as tcnn
        tinycudann_version: str | None = getattr(tcnn, "__version__", "installed")
    except ImportError:
        tinycudann_version = None
    manifest: dict[str, Any] = {
        "experiment_id": f"{route}_{subject_id or 'unknown'}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}", "timestamp": datetime.now(timezone.utc).isoformat(),
        "route": route, "subject_id": subject_id, "repository": "universe26dyz/Code_10w", **git,
        "vendored_nesvor_commit": _vendored_commit(method), "config_resolved": str(config_resolved), "command": command,
        "prepared_inputs": [str(value) for value in prepared], "prepared_manifest_sha256": hashlib.sha256("".join(_sha256_file(value) for value in prepared).encode("ascii")).hexdigest(),
        "tr_ms": protocol.get("tr_ms"), "vps": protocol.get("vps"), "stack_group_counts": dict(stack_group_counts), "seed": int(seed),
        "python": sys.version, "torch": torch.__version__, "cuda": cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tinycudann": tinycudann_version, "hostname": platform.node(),
    }
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
