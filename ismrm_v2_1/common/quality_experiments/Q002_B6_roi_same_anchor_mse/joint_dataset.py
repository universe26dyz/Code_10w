"""Native integer-pixel joint anchors for Q002's 10-weight fingerprint loss."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


_REQUIRED = {
    "images", "masks", "group_idx", "weight_idx", "stack_idx", "timing9_ms", "affine_lps_rc",
    "pixel_spacing_rc_mm", "slice_thickness_mm", "tr_ms", "vps",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _equal(name: str, value: np.ndarray, indices: np.ndarray, *, atol: float = 1e-10) -> None:
    selected = value[indices]
    if not np.allclose(selected, selected[0], rtol=0.0, atol=atol):
        raise ValueError(f"Q002 group weights must share {name}.")


class JointAnchorDataset:
    """A Q002-only view with one complete measured fingerprint per native pixel."""

    coordinate_system = "local xyz [column,row,slice] mm"

    def __init__(self, observation_npz_paths: Sequence[str | Path], scalar_dataset: Any) -> None:
        if isinstance(observation_npz_paths, (str, Path)) or not observation_npz_paths:
            raise ValueError("JointAnchorDataset requires non-empty cropped observations paths.")
        self.source_paths = tuple(Path(item) for item in observation_npz_paths)
        self.scalar_dataset = scalar_dataset
        self.stack_weights = dict(scalar_dataset.validate_balanced_samples())
        self._build()

    def _load(self, path: Path) -> Mapping[str, np.ndarray]:
        if not path.is_file():
            raise FileNotFoundError(f"Q002 observations NPZ does not exist: {path}")
        if "full_fov" in str(path).lower():
            raise ValueError("Q002 joint dataset forbids full-FOV sources.")
        with np.load(path, allow_pickle=False) as data:
            missing = _REQUIRED.difference(data.files)
            if missing:
                raise ValueError(f"Q002 observations NPZ lacks: {sorted(missing)}")
            return {name: np.asarray(data[name]) for name in _REQUIRED}

    @staticmethod
    def _validate_payload(payload: Mapping[str, np.ndarray], path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        images, masks = np.asarray(payload["images"], np.float32), np.asarray(payload["masks"], bool)
        groups, weights, stacks = (np.asarray(payload[name], np.int64) for name in ("group_idx", "weight_idx", "stack_idx"))
        n_observations = images.shape[0]
        if images.ndim != 3 or masks.shape != images.shape:
            raise ValueError(f"Q002 images/masks must match [N,H,W] in {path}.")
        if any(array.shape != (n_observations,) for array in (groups, weights, stacks, np.asarray(payload["slice_thickness_mm"]), np.asarray(payload["tr_ms"]), np.asarray(payload["vps"]))):
            raise ValueError(f"Q002 observation metadata must be [N] in {path}.")
        if np.asarray(payload["timing9_ms"]).shape != (n_observations, 9) or np.asarray(payload["affine_lps_rc"]).shape != (n_observations, 4, 4):
            raise ValueError(f"Q002 timing/affine shape is invalid in {path}.")
        if np.asarray(payload["pixel_spacing_rc_mm"]).shape != (n_observations, 2):
            raise ValueError(f"Q002 pixel spacing shape is invalid in {path}.")
        unique = np.unique(groups)
        if not np.array_equal(unique, np.arange(unique.size)):
            raise ValueError(f"Q002 source group_idx must be contiguous zero-based in {path}.")
        return images, masks, groups, weights

    def _build(self) -> None:
        xyz_rows: list[np.ndarray] = []
        observed_rows: list[np.ndarray] = []
        group_rows: list[np.ndarray] = []
        stack_rows: list[np.ndarray] = []
        timing_rows: list[np.ndarray] = []
        row_col_rows: list[np.ndarray] = []
        group_qc: dict[str, dict[str, Any]] = {}
        scalar_weight_counts = {str(weight): 0 for weight in range(10)}
        source_hashes: dict[str, str] = {}
        global_offset = 0
        for path in self.source_paths:
            payload = self._load(path)
            source_hashes[str(path)] = _sha256(path)
            images, masks, groups, weights = self._validate_payload(payload, path)
            stacks = np.asarray(payload["stack_idx"], np.int64)
            timing = np.asarray(payload["timing9_ms"], np.float64)
            affine = np.asarray(payload["affine_lps_rc"], np.float64)
            spacing = np.asarray(payload["pixel_spacing_rc_mm"], np.float64)
            thickness = np.asarray(payload["slice_thickness_mm"], np.float64)
            tr_ms, vps = np.asarray(payload["tr_ms"], np.float64), np.asarray(payload["vps"], np.int64)
            for local_group in np.unique(groups):
                member = np.flatnonzero(groups == local_group)
                ordered = member[np.argsort(weights[member], kind="stable")]
                if ordered.size != 10 or not np.array_equal(weights[ordered], np.arange(10)):
                    raise ValueError(f"Q002 group {local_group} in {path} must contain exactly one weight_idx 0..9.")
                if np.unique(stacks[ordered]).size != 1:
                    raise ValueError("Q002 group weights must share stack_idx.")
                _equal("HB1 affine", affine, ordered)
                _equal("timing9_ms", timing, ordered)
                _equal("pixel spacing", spacing, ordered)
                _equal("slice thickness", thickness, ordered)
                _equal("TR", tr_ms, ordered, atol=1e-6)
                if not np.array_equal(vps[ordered], np.full(10, vps[ordered[0]])):
                    raise ValueError("Q002 group weights must share VPS.")
                for weight, index in enumerate(ordered):
                    scalar_weight_counts[str(weight)] += int(masks[index].sum())
                joint_mask = masks[ordered].all(axis=0) & np.isfinite(images[ordered]).all(axis=0)
                rows, cols = np.nonzero(joint_mask)
                global_group = global_offset + int(local_group)
                if rows.size == 0:
                    raise ValueError(f"Q002 group {global_group} has no joint-valid anchor after ten-way AND support.")
                height, width = images.shape[1:]
                row_spacing, col_spacing = spacing[ordered[0]]
                xyz_rows.append(np.column_stack(((cols - (width - 1) / 2.0) * col_spacing, (rows - (height - 1) / 2.0) * row_spacing, np.zeros(rows.size))))
                observed_rows.append(images[ordered][:, rows, cols].T)
                group_rows.append(np.full(rows.size, global_group, np.int64))
                stack_rows.append(np.full(rows.size, int(stacks[ordered[0]]), np.int64))
                timing_rows.append(np.repeat(timing[ordered[0]][None], rows.size, axis=0))
                row_col_rows.append(np.column_stack((rows, cols)).astype(np.int64, copy=False))
                union = masks[ordered].any(axis=0)
                scalar_total = int(masks[ordered].sum())
                group_qc[str(global_group)] = {
                    "source_path": str(path), "source_local_group_idx": int(local_group), "stack_idx": int(stacks[ordered[0]]),
                    "roi_shape_rows_cols": [int(height), int(width)], "joint_valid_count": int(rows.size),
                    "union_valid_count": int(union.sum()), "scalar_valid_count_all_weights": scalar_total,
                    "joint_support_fraction_of_union": float(rows.size / union.sum()) if union.any() else 0.0,
                    "dropped_scalar_fraction_vs_per_weight_union": float(1.0 - (10 * rows.size) / scalar_total) if scalar_total else 1.0,
                    "masks_identical": bool(np.all(masks[ordered] == masks[ordered[0]])),
                }
            global_offset += int(np.unique(groups).size)
        if global_offset != int(self.scalar_dataset.group_pose_count):
            raise ValueError("Q002 joint/scalar global group counts differ; observations paths must match exactly.")
        self.xyz = torch.as_tensor(np.concatenate(xyz_rows), dtype=torch.float32)
        self.observed = torch.as_tensor(np.concatenate(observed_rows), dtype=torch.float32)
        self.group_idx = torch.as_tensor(np.concatenate(group_rows), dtype=torch.long)
        self.stack_idx = torch.as_tensor(np.concatenate(stack_rows), dtype=torch.long)
        self.timing = torch.as_tensor(np.concatenate(timing_rows), dtype=torch.float32)
        self.row_col = torch.as_tensor(np.concatenate(row_col_rows), dtype=torch.long)
        joint_stack_counts = {str(int(stack)): int((self.stack_idx == stack).sum()) for stack in self.stack_idx.unique().tolist()}
        self.qc = {
            "coordinate_system": self.coordinate_system,
            "source_sha256": source_hashes,
            "groups": group_qc,
            "scalar_valid_pixels_per_weight": scalar_weight_counts,
            "joint_anchor_count": int(self.xyz.shape[0]),
            "joint_anchor_count_by_stack": joint_stack_counts,
            "scalar_stack_weights": {str(key): float(value) for key, value in self.stack_weights.items()},
            "joint_support_changes_sampling_support": any(not value["masks_identical"] for value in group_qc.values()),
        }

    def fixed_monitor(self, *, seed: int, samples_per_stack: int) -> dict[str, Any]:
        if samples_per_stack < 1:
            raise ValueError("samples_per_stack must be positive.")
        generator = np.random.default_rng(seed)
        chosen: list[np.ndarray] = []
        for stack in sorted(int(item) for item in self.stack_idx.unique().tolist()):
            candidates = torch.nonzero(self.stack_idx == stack, as_tuple=False).flatten().cpu().numpy()
            chosen.append(generator.choice(candidates, size=samples_per_stack, replace=candidates.size < samples_per_stack))
        indices = torch.as_tensor(np.concatenate(chosen), dtype=torch.long)
        identity = np.column_stack((self.group_idx[indices].numpy(), self.row_col[indices].numpy())).astype("<i8", copy=False)
        return {
            "xyz": self.xyz[indices], "observed": self.observed[indices], "group_idx": self.group_idx[indices],
            "stack_idx": self.stack_idx[indices], "timing": self.timing[indices], "row_col": self.row_col[indices],
            "seed": int(seed), "samples_per_stack": int(samples_per_stack),
            "anchor_identity_sha256": hashlib.sha256(identity.tobytes()).hexdigest(),
        }
