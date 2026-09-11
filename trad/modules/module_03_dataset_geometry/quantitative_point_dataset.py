"""Project-local group dataset adapted from NeSVoR ``PointDataset`` semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch

from modules.module_02_data_bridge.geometry import apply_affine_rc


class QuantPointDataset:
    """Flatten masked prepared pixels while preserving one rigid-pose index per group.

    This retains NeSVoR PointDataset's central representation (concatenated
    untransformed ``xyz`` and intensity ``v`` plus batched sampling) but replaces
    its per-slice index with ``group_idx``. All ten weights of a native spatial
    slice therefore resolve to exactly one future rigid pose.
    """

    def __init__(self, observations_npz: str | Path, device: str | torch.device = "cpu") -> None:
        observations_npz = Path(observations_npz)
        if not observations_npz.is_file():
            raise FileNotFoundError(f"Prepared observations NPZ does not exist: {observations_npz}")
        with np.load(observations_npz, allow_pickle=False) as data:
            required = {
                "images",
                "masks",
                "group_idx",
                "weight_idx",
                "stack_idx",
                "acquisition_time_ms",
                "timing9_ms",
                "affine_lps_rc",
            }
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"Prepared observations NPZ lacks: {sorted(missing)}")
            images = np.asarray(data["images"], dtype=np.float32)
            masks = np.asarray(data["masks"], dtype=bool)
            group_idx = np.asarray(data["group_idx"], dtype=np.int64)
            weight_idx = np.asarray(data["weight_idx"], dtype=np.int64)
            stack_idx = np.asarray(data["stack_idx"], dtype=np.int64)
            acquisition_time_ms = np.asarray(data["acquisition_time_ms"], dtype=np.float64)
            timing9_ms = np.asarray(data["timing9_ms"], dtype=np.float64)
            affine_lps_rc = np.asarray(data["affine_lps_rc"], dtype=np.float64)
        self._validate_observations(
            images, masks, group_idx, weight_idx, stack_idx, acquisition_time_ms, timing9_ms, affine_lps_rc
        )
        self.coordinate_system = "DICOM_LPS_mm; xyz derived from affine_lps_rc [row,col,slice,1]"
        self.group_ids = np.unique(group_idx)
        self.group_pose_count = int(self.group_ids.size)
        self.group_affine_lps_rc = np.stack(
            [affine_lps_rc[np.flatnonzero(group_idx == group_id)[0]] for group_id in self.group_ids]
        )
        xyz_all: list[np.ndarray] = []
        v_all: list[np.ndarray] = []
        group_all: list[np.ndarray] = []
        weight_all: list[np.ndarray] = []
        stack_all: list[np.ndarray] = []
        timing_all: list[np.ndarray] = []
        acq_all: list[np.ndarray] = []
        for observation_idx in range(images.shape[0]):
            rows, cols = np.nonzero(masks[observation_idx])
            if rows.size == 0:
                raise ValueError(f"Observation {observation_idx} has an empty foreground mask.")
            xyz_all.append(apply_affine_rc(affine_lps_rc[observation_idx], rows, cols))
            v_all.append(images[observation_idx, rows, cols])
            group_all.append(np.full(rows.size, group_idx[observation_idx], dtype=np.int64))
            weight_all.append(np.full(rows.size, weight_idx[observation_idx], dtype=np.int64))
            stack_all.append(np.full(rows.size, stack_idx[observation_idx], dtype=np.int64))
            timing_all.append(np.repeat(timing9_ms[observation_idx][None, :], rows.size, axis=0))
            acq_all.append(np.full(rows.size, acquisition_time_ms[observation_idx], dtype=np.float64))
        self.xyz = torch.as_tensor(np.concatenate(xyz_all), dtype=torch.float32, device=device)
        self.v = torch.as_tensor(np.concatenate(v_all), dtype=torch.float32, device=device)
        self.group_idx = torch.as_tensor(np.concatenate(group_all), dtype=torch.long, device=device)
        self.weight_idx = torch.as_tensor(np.concatenate(weight_all), dtype=torch.long, device=device)
        self.stack_idx = torch.as_tensor(np.concatenate(stack_all), dtype=torch.long, device=device)
        self.timing = torch.as_tensor(np.concatenate(timing_all), dtype=torch.float32, device=device)
        self.acquisition_time_ms = torch.as_tensor(np.concatenate(acq_all), dtype=torch.float32, device=device)
        self.count = 0
        self.epoch = 0

    @staticmethod
    def _validate_observations(
        images: np.ndarray,
        masks: np.ndarray,
        group_idx: np.ndarray,
        weight_idx: np.ndarray,
        stack_idx: np.ndarray,
        acquisition_time_ms: np.ndarray,
        timing9_ms: np.ndarray,
        affine_lps_rc: np.ndarray,
    ) -> None:
        n_obs = images.shape[0]
        if images.ndim != 3 or masks.shape != images.shape:
            raise ValueError(f"images/masks must be equally shaped [N,H,W], got {images.shape}/{masks.shape}.")
        if any(value.shape != (n_obs,) for value in (group_idx, weight_idx, stack_idx, acquisition_time_ms)):
            raise ValueError("Observation metadata vectors must have shape [N].")
        if timing9_ms.shape != (n_obs, 9):
            raise ValueError(f"timing9_ms must be [N,9], got {timing9_ms.shape}.")
        if affine_lps_rc.shape != (n_obs, 4, 4):
            raise ValueError(f"affine_lps_rc must be [N,4,4], got {affine_lps_rc.shape}.")
        if not np.all(np.isfinite(images)) or not np.all(np.isfinite(timing9_ms)):
            raise ValueError("Prepared images and timing must be finite.")
        unique_groups = np.unique(group_idx)
        if not np.array_equal(unique_groups, np.arange(unique_groups.size)):
            raise ValueError(f"group_idx must be contiguous zero-based, got {unique_groups.tolist()}.")
        for group in unique_groups:
            indices = np.flatnonzero(group_idx == group)
            if indices.size != 10 or not np.array_equal(np.sort(weight_idx[indices]), np.arange(10)):
                raise ValueError(f"Group {group} must contain exactly one observation for weights 0..9.")
            if np.unique(stack_idx[indices]).size != 1:
                raise ValueError(f"Group {group} must have one stack_idx.")
            if not np.allclose(affine_lps_rc[indices], affine_lps_rc[indices[0]], rtol=0, atol=1e-10):
                raise ValueError(f"Group {group} weights must share one HB1 cropped affine.")
            if not np.allclose(timing9_ms[indices], timing9_ms[indices[0]], rtol=0, atol=1e-10):
                raise ValueError(f"Group {group} weights must share one timing vector.")

    @property
    def bounding_box(self) -> torch.Tensor:
        """NeSVoR-style untransformed LPS bounding box, without rigid poses yet."""

        return torch.stack((self.xyz.amin(dim=0), self.xyz.amax(dim=0)), dim=0)

    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Return PointDataset-like tensors augmented with group/weight/timing metadata."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.count + batch_size > self.xyz.shape[0]:
            self.count = 0
            self.epoch += 1
            permutation = torch.randperm(self.xyz.shape[0], device=self.xyz.device)
            for name in ("xyz", "v", "group_idx", "weight_idx", "stack_idx", "timing", "acquisition_time_ms"):
                setattr(self, name, getattr(self, name)[permutation])
        batch_slice = slice(self.count, self.count + batch_size)
        self.count += batch_size
        return {
            "xyz": self.xyz[batch_slice],
            "v": self.v[batch_slice],
            "group_idx": self.group_idx[batch_slice],
            "weight_idx": self.weight_idx[batch_slice],
            "stack_idx": self.stack_idx[batch_slice],
            "timing": self.timing[batch_slice],
            "acquisition_time_ms": self.acquisition_time_ms[batch_slice],
        }
