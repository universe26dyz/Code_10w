"""Transparent mDM-inspired coherent-RR synthetic timing domain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from modules.module_02_data_bridge.timing import compute_duration_before_acq


@dataclass(frozen=True)
class RRDomainConfig:
    n_rhythms: int
    seed: int
    tr_ms: float
    vps: int
    ti_ms: tuple[float, float]
    t2prep_ms: tuple[float, float, float]
    hr_range_bpm: tuple[float, float] = (20.0, 120.0)
    cv_range: tuple[float, float] = (0.0, 0.50)
    max_attempts_per_rhythm: int = 1000

    def validate(self) -> None:
        if self.n_rhythms < 1 or self.vps < 1 or self.tr_ms <= 0 or self.max_attempts_per_rhythm < 1:
            raise ValueError("RR domain counts and sequence timing must be positive.")
        if self.hr_range_bpm[0] <= 0 or self.hr_range_bpm[0] > self.hr_range_bpm[1] or self.cv_range[0] < 0 or self.cv_range[0] > self.cv_range[1]:
            raise ValueError("RR HR/CV ranges are invalid.")


def _timing_from_rr(rr_ms: np.ndarray, config: RRDomainConfig) -> np.ndarray:
    acquisition = np.concatenate(([0.0], np.cumsum(np.asarray(rr_ms, dtype=np.float64))))
    duration = compute_duration_before_acq(acquisition, config.tr_ms, config.vps, np.asarray(config.ti_ms), np.asarray(config.t2prep_ms))
    timing9 = duration[1:]
    if not np.all(np.isfinite(timing9)) or np.any(timing9 <= 0):
        raise ValueError("RR history produces non-positive or non-finite duration_before_acq.")
    return timing9


def generate_rhythm_timing9(config: RRDomainConfig) -> dict[str, np.ndarray]:
    """Generate coherent 9-RR histories; invalid sequence intervals are rejected."""

    config.validate(); rng = np.random.default_rng(config.seed)
    rr_rows, timing_rows, mean_hr, cv_rows = [], [], [], []
    for _ in range(config.n_rhythms):
        for _attempt in range(config.max_attempts_per_rhythm):
            hr = float(rng.uniform(*config.hr_range_bpm)); cv = float(rng.uniform(*config.cv_range))
            mean_rr = 60000.0 / hr
            rr = rng.normal(mean_rr, cv * mean_rr, size=9)
            if np.any(rr <= 0):
                continue
            try:
                timing = _timing_from_rr(rr, config)
            except ValueError:
                continue
            rr_rows.append(rr); timing_rows.append(timing); mean_hr.append(hr); cv_rows.append(cv)
            break
        else:
            raise RuntimeError("Could not sample a sequence-valid RR history within max_attempts_per_rhythm.")
    return {"rhythm_id": np.arange(config.n_rhythms, dtype=np.int64), "rr_ms": np.asarray(rr_rows, dtype=np.float32), "timing9_ms": np.asarray(timing_rows, dtype=np.float32), "mean_hr_bpm": np.asarray(mean_hr, dtype=np.float32), "rr_cv": np.asarray(cv_rows, dtype=np.float32)}


def split_rhythm_ids(n_rhythms: int, *, seed: int, counts: Mapping[str, int]) -> dict[str, np.ndarray]:
    if set(counts) != {"train", "valid", "test"} or sum(int(value) for value in counts.values()) != n_rhythms or any(int(value) < 1 for value in counts.values()):
        raise ValueError("rhythm split must be a non-empty exhaustive train/valid/test partition.")
    order = np.random.default_rng(seed).permutation(n_rhythms); start = 0; result = {}
    for name in ("train", "valid", "test"):
        end = start + int(counts[name]); result[name] = order[start:end]; start = end
    return result
