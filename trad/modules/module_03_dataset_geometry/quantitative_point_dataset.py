"""NeSVoR-style local-coordinate dataset for one or more 10-weight prepared stacks."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch

from trad.modules.module_02_data_bridge.geometry import cropped_affine_lps_rc_to_initial_rigid
from trad.third_party.nesvor.nesvor.transform import RigidTransform, ax_transform_points


def robust_trimmed_mean_intensity(
    values: torch.Tensor, *, lower_quantile: float, upper_quantile: float
) -> torch.Tensor:
    """Return NeSVoR-style robust scale from all subject masked intensities."""

    if values.numel() == 0:
        raise ValueError("Subject intensity values must not be empty.")
    if not torch.isfinite(values).all():
        raise ValueError("Subject intensity values must be finite.")
    if not 0.0 < lower_quantile < upper_quantile < 1.0:
        raise ValueError("Intensity normalization quantiles must satisfy 0 < lower < upper < 1.")
    flattened = values.reshape(-1)
    q10, q90 = torch.quantile(flattened, lower_quantile), torch.quantile(flattened, upper_quantile)
    trimmed = flattened[(flattened > q10) & (flattened < q90)]
    if trimmed.numel() == 0:
        raise ValueError("Subject trimmed intensity set is empty.")
    scale = trimmed.mean()
    if not torch.isfinite(scale) or scale <= 0:
        raise ValueError(f"Subject trimmed intensity scale must be finite and positive, got {float(scale)}.")
    return scale


class QuantPointDataset:
    """Flatten complete 10-weight groups while retaining NeSVoR local/world semantics.

    ``observation_npz_paths`` is a non-empty sequence of prepared stack files.
    Source-local groups are remapped consecutively. ``xyz`` is centered
    slice-local `[column,row,slice]` mm; ``xyz_transformed`` applies exactly
    one initial RigidTransform per global group into RAS-mm world space.
    """

    def __init__(
        self, observation_npz_paths: Sequence[str | Path], device: str | torch.device = "cpu"
    ) -> None:
        if isinstance(observation_npz_paths, (str, Path)) or not observation_npz_paths:
            raise ValueError("QuantPointDataset requires a non-empty sequence of observations.npz paths.")
        payloads = [self._load_stack(Path(path)) for path in observation_npz_paths]
        self.coordinate_system = "local xyz [column,row,slice] mm; transformed world RAS mm"
        self._build_from_stacks(payloads, device)

    @staticmethod
    def _load_stack(path: Path) -> dict[str, np.ndarray]:
        if not path.is_file():
            raise FileNotFoundError(f"Prepared observations NPZ does not exist: {path}")
        with np.load(path, allow_pickle=False) as source:
            required = {
                "images", "masks", "group_idx", "weight_idx", "stack_idx",
                "acquisition_time_ms", "timing9_ms", "affine_lps_rc",
                "pixel_spacing_rc_mm", "slice_thickness_mm", "tr_ms", "vps",
            }
            missing = required.difference(source.files)
            if missing:
                raise ValueError(f"Prepared observations NPZ lacks: {sorted(missing)}")
            return {name: np.asarray(source[name]) for name in required}

    @staticmethod
    def _validate_stack(payload: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        images = np.asarray(payload["images"], dtype=np.float32)
        masks = np.asarray(payload["masks"], dtype=bool)
        group_idx = np.asarray(payload["group_idx"], dtype=np.int64)
        weight_idx = np.asarray(payload["weight_idx"], dtype=np.int64)
        stack_idx = np.asarray(payload["stack_idx"], dtype=np.int64)
        acquisition_time_ms = np.asarray(payload["acquisition_time_ms"], dtype=np.float64)
        timing9_ms = np.asarray(payload["timing9_ms"], dtype=np.float64)
        affine_lps_rc = np.asarray(payload["affine_lps_rc"], dtype=np.float64)
        pixel_spacing_rc_mm = np.asarray(payload["pixel_spacing_rc_mm"], dtype=np.float64)
        slice_thickness_mm = np.asarray(payload["slice_thickness_mm"], dtype=np.float64)
        tr_ms = np.asarray(payload["tr_ms"], dtype=np.float64)
        vps = np.asarray(payload["vps"], dtype=np.int64)
        n_obs = images.shape[0]
        if images.ndim != 3 or masks.shape != images.shape:
            raise ValueError(f"images/masks must be equal [N,H,W], got {images.shape}/{masks.shape}.")
        if any(v.shape != (n_obs,) for v in (group_idx, weight_idx, stack_idx, acquisition_time_ms, slice_thickness_mm, tr_ms, vps)):
            raise ValueError("Observation vectors must have shape [N].")
        if timing9_ms.shape != (n_obs, 9) or affine_lps_rc.shape != (n_obs, 4, 4):
            raise ValueError("timing9_ms and affine_lps_rc must be [N,9] and [N,4,4].")
        if pixel_spacing_rc_mm.shape != (n_obs, 2):
            raise ValueError(f"pixel_spacing_rc_mm must be [N,2], got {pixel_spacing_rc_mm.shape}.")
        local_groups = np.unique(group_idx)
        if not np.array_equal(local_groups, np.arange(local_groups.size)):
            raise ValueError(f"Source group_idx must be contiguous zero-based, got {local_groups.tolist()}.")
        resolutions: list[np.ndarray] = []
        group_tr_ms: list[float] = []
        group_vps: list[int] = []
        for group in local_groups:
            indices = np.flatnonzero(group_idx == group)
            if indices.size != 10 or not np.array_equal(np.sort(weight_idx[indices]), np.arange(10)):
                raise ValueError(f"Source group {group} must contain exactly one weight 0..9.")
            if np.unique(stack_idx[indices]).size != 1:
                raise ValueError(f"Source group {group} must have one stack_idx.")
            if not np.allclose(affine_lps_rc[indices], affine_lps_rc[indices[0]], rtol=0, atol=1e-10):
                raise ValueError(f"Source group {group} weights must share one cropped HB1 affine.")
            if not np.allclose(timing9_ms[indices], timing9_ms[indices[0]], rtol=0, atol=1e-10):
                raise ValueError(f"Source group {group} weights must share one timing vector.")
            if not np.allclose(pixel_spacing_rc_mm[indices], pixel_spacing_rc_mm[indices[0]], rtol=0, atol=1e-10):
                raise ValueError(f"Source group {group} weights must share PixelSpacing.")
            if not np.allclose(slice_thickness_mm[indices], slice_thickness_mm[indices[0]], rtol=0, atol=1e-10):
                raise ValueError(f"Source group {group} weights must share SliceThickness.")
            if not np.allclose(tr_ms[indices], tr_ms[indices[0]], rtol=1e-6, atol=1e-6):
                raise ValueError(f"Source group {group} weights must share DICOM TR.")
            if not np.array_equal(vps[indices], np.full(indices.size, vps[indices[0]])):
                raise ValueError(f"Source group {group} weights must share DICOM VPS.")
            row_spacing, col_spacing = pixel_spacing_rc_mm[indices[0]]
            resolution_xyz = np.asarray([col_spacing, row_spacing, slice_thickness_mm[indices[0]]])
            if np.any(~np.isfinite(resolution_xyz)) or np.any(resolution_xyz <= 0):
                raise ValueError(f"Source group {group} has invalid [col,row,thickness] resolution.")
            resolutions.append(resolution_xyz)
            if not np.isfinite(tr_ms[indices[0]]) or tr_ms[indices[0]] <= 0 or vps[indices[0]] <= 0:
                raise ValueError(f"Source group {group} has invalid DICOM TR/VPS.")
            group_tr_ms.append(float(tr_ms[indices[0]]))
            group_vps.append(int(vps[indices[0]]))
        return np.stack(resolutions), np.asarray(group_tr_ms), np.asarray(group_vps)

    def _build_from_stacks(self, payloads: list[dict[str, np.ndarray]], device: str | torch.device) -> None:
        local_xyz, intensity, groups, weights, stacks, timings, acq = [], [], [], [], [], [], []
        initial_matrices, resolutions, group_tr_ms, group_vps = [], [], [], []
        offset = 0
        for payload in payloads:
            group_resolutions, source_group_tr, source_group_vps = self._validate_stack(payload)
            images, masks = payload["images"].astype(np.float32), payload["masks"].astype(bool)
            source_groups = payload["group_idx"].astype(np.int64)
            for local_group, resolution_xyz in enumerate(group_resolutions):
                indices = np.flatnonzero(source_groups == local_group)
                h, w = images.shape[1:]
                initial = cropped_affine_lps_rc_to_initial_rigid(
                    payload["affine_lps_rc"][indices[0]], (h, w), resolution_xyz, device=device
                )
                initial_matrices.append(initial.matrix(trans_first=True))
                resolutions.append(resolution_xyz)
                group_tr_ms.append(source_group_tr[local_group])
                group_vps.append(source_group_vps[local_group])
                global_group = offset + local_group
                for observation_idx in indices:
                    rows, cols = np.nonzero(masks[observation_idx])
                    if rows.size == 0:
                        raise ValueError(f"Group {global_group} has an empty foreground mask.")
                    local_xyz.append(np.column_stack(((cols - (w - 1) / 2.0) * resolution_xyz[0], (rows - (h - 1) / 2.0) * resolution_xyz[1], np.zeros(rows.size))))
                    intensity.append(images[observation_idx, rows, cols])
                    groups.append(np.full(rows.size, global_group, dtype=np.int64))
                    weights.append(np.full(rows.size, payload["weight_idx"][observation_idx], dtype=np.int64))
                    stacks.append(np.full(rows.size, payload["stack_idx"][observation_idx], dtype=np.int64))
                    timings.append(np.repeat(payload["timing9_ms"][observation_idx][None], rows.size, axis=0))
                    acq.append(np.full(rows.size, payload["acquisition_time_ms"][observation_idx], dtype=np.float64))
            offset += group_resolutions.shape[0]
        self.group_pose_count = offset
        self.group_resolution_xyz_mm = torch.as_tensor(np.stack(resolutions), dtype=torch.float32, device=device)
        # Keep DICOM timing provenance in float64; unlike pixel tensors it is
        # protocol metadata, and training must not silently round it to float32.
        self.group_tr_ms = torch.as_tensor(np.asarray(group_tr_ms), dtype=torch.float64, device=device)
        self.group_vps = torch.as_tensor(np.asarray(group_vps), dtype=torch.long, device=device)
        self.initial_transformation = RigidTransform(torch.cat(initial_matrices), trans_first=True)
        self.group_axisangle_init = self.initial_transformation.axisangle(trans_first=True).detach().clone()
        self.xyz = torch.as_tensor(np.concatenate(local_xyz), dtype=torch.float32, device=device)
        self.v = torch.as_tensor(np.concatenate(intensity), dtype=torch.float32, device=device)
        self.group_idx = torch.as_tensor(np.concatenate(groups), dtype=torch.long, device=device)
        self.weight_idx = torch.as_tensor(np.concatenate(weights), dtype=torch.long, device=device)
        self.stack_idx = torch.as_tensor(np.concatenate(stacks), dtype=torch.long, device=device)
        self.timing = torch.as_tensor(np.concatenate(timings), dtype=torch.float32, device=device)
        self.acquisition_time_ms = torch.as_tensor(np.concatenate(acq), dtype=torch.float32, device=device)
        self.count, self.epoch = 0, 0

    @property
    def xyz_transformed(self) -> torch.Tensor:
        return ax_transform_points(self.group_axisangle_init[self.group_idx], self.xyz, trans_first=True)

    @property
    def bounding_box(self) -> torch.Tensor:
        transformed = self.xyz_transformed
        max_r = self.group_resolution_xyz_mm.max()
        return torch.stack((transformed.amin(dim=0) - 2 * max_r, transformed.amax(dim=0) + 2 * max_r), dim=0)

    def validated_tr_vps(self) -> tuple[float, int]:
        """Return the only protocol allowed for one online Trad reconstruction."""

        tr_values = self.group_tr_ms.detach().cpu().numpy()
        vps_values = self.group_vps.detach().cpu().numpy()
        if not np.allclose(tr_values, tr_values[0], rtol=1e-6, atol=1e-6) or not np.all(vps_values == vps_values[0]):
            records = [f"group={i}:TR={tr_values[i]:.9g},VPS={int(vps_values[i])}" for i in range(self.group_pose_count)]
            raise ValueError("Trad requires one DICOM TR/VPS across all input groups/stacks; " + "; ".join(records))
        return float(tr_values[0]), int(vps_values[0])

    def validate_balanced_samples(self) -> dict[int, float]:
        """Validate exact 10-weight counts and return static inverse-frequency stack weights."""

        counts = torch.bincount(self.weight_idx, minlength=10)
        if counts.numel() != 10 or torch.any(counts == 0) or not torch.all(counts == counts[0]):
            raise ValueError(f"Dataset must have equal samples for weights 0..9, got {counts.tolist()}.")
        for group in range(self.group_pose_count):
            current = torch.bincount(self.weight_idx[self.group_idx == group], minlength=10)
            if not torch.all(current == current[0]):
                raise ValueError(f"Group {group} must have equal samples for its 10 weights, got {current.tolist()}.")
        stack_counts = {int(stack): int((self.stack_idx == stack).sum().item()) for stack in self.stack_idx.unique().tolist()}
        inverse = {stack: 1.0 / count for stack, count in stack_counts.items()}
        normalizer = sum(stack_counts[stack] * inverse[stack] for stack in stack_counts) / self.v.numel()
        return {stack: value / normalizer for stack, value in inverse.items()}

    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.count + batch_size > self.xyz.shape[0]:
            self.count, self.epoch = 0, self.epoch + 1
            permutation = torch.randperm(self.xyz.shape[0], device=self.xyz.device)
            for name in ("xyz", "v", "group_idx", "weight_idx", "stack_idx", "timing", "acquisition_time_ms"):
                setattr(self, name, getattr(self, name)[permutation])
        selected = slice(self.count, self.count + batch_size)
        self.count += batch_size
        return {"xyz": self.xyz[selected], "v": self.v[selected], "group_idx": self.group_idx[selected], "weight_idx": self.weight_idx[selected], "stack_idx": self.stack_idx[selected], "timing": self.timing[selected], "acquisition_time_ms": self.acquisition_time_ms[selected]}
