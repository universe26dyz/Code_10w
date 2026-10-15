"""Finite-support, finite-thickness signal-domain rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates


def lps_to_ras_affine(affine_lps: np.ndarray) -> np.ndarray:
    flip = np.diag([-1.0, -1.0, 1.0, 1.0])
    return flip @ np.asarray(affine_lps, dtype=float)


def ras_to_lps_affine(affine_ras: np.ndarray) -> np.ndarray:
    flip = np.diag([-1.0, -1.0, 1.0, 1.0])
    return flip @ np.asarray(affine_ras, dtype=float)


def _sample(volume: np.ndarray, points_world: np.ndarray, origin: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    coords = ((points_world - origin[None, :]) / spacing[None, :]).T
    samples = []
    for weight in range(volume.shape[-1]):
        samples.append(map_coordinates(volume[..., weight], coords, order=1, mode="constant", cval=0.0, prefilter=False))
    return np.stack(samples, axis=-1)


def integrate_signal_volume(
    signal_volume: np.ndarray,
    volume_origin_mm: np.ndarray,
    volume_spacing_mm: np.ndarray,
    affine_lps_rc: np.ndarray,
    rows: int,
    cols: int,
    inplane_samples: int = 2,
    thickness_samples: int = 3,
) -> np.ndarray:
    """Average sampled signal values over each pixel footprint and slice.

    ``signal_volume`` is ``(x, y, z, 10)``.  The third affine column carries
    the acquired slice-thickness vector.  Sampling is midpoint quadrature with
    linear signal interpolation and a fixed deterministic grid.
    """

    signal_volume = np.asarray(signal_volume, dtype=np.float32)
    if signal_volume.ndim != 4 or signal_volume.shape[-1] != 10 and signal_volume.shape[-1] != 1:
        raise ValueError("signal_volume must have shape (x,y,z,weights)")
    affine = np.asarray(affine_lps_rc, dtype=float)
    if affine.shape != (4, 4):
        raise ValueError("affine_lps_rc must be 4x4")
    origin = np.asarray(volume_origin_mm, dtype=float)
    spacing = np.asarray(volume_spacing_mm, dtype=float)
    row_vec, col_vec, slice_vec = affine[:3, 0], affine[:3, 1], affine[:3, 2]
    row_offsets = ((np.arange(inplane_samples, dtype=float) + 0.5) / inplane_samples - 0.5)
    col_offsets = ((np.arange(inplane_samples, dtype=float) + 0.5) / inplane_samples - 0.5)
    slice_offsets = ((np.arange(thickness_samples, dtype=float) + 0.5) / thickness_samples - 0.5)
    rr, cc = np.meshgrid(np.arange(rows, dtype=float), np.arange(cols, dtype=float), indexing="ij")
    base = affine[:3, 3][None, None, :] + rr[..., None] * row_vec + cc[..., None] * col_vec
    offsets = np.asarray([[ro, co, so] for ro in row_offsets for co in col_offsets for so in slice_offsets], dtype=float)
    base = base.reshape(-1, 3)
    accum = np.zeros((base.shape[0], signal_volume.shape[-1]), dtype=np.float64)
    # Batching offsets avoids thousands of tiny interpolation calls for the
    # required 6x6x9 convergence reference without changing quadrature nodes.
    for start in range(0, len(offsets), 12):
        current = offsets[start : start + 12]
        displacements = current[:, 0, None] * row_vec + current[:, 1, None] * col_vec + current[:, 2, None] * slice_vec
        points = (base[None, :, :] + displacements[:, None, :]).reshape(-1, 3)
        coords = ((points - origin[None, :]) / spacing[None, :]).T
        n_offsets = len(current)
        for weight in range(signal_volume.shape[-1]):
            sampled = map_coordinates(signal_volume[..., weight], coords, order=1, mode="constant", cval=0.0, prefilter=False)
            accum[:, weight] += sampled.reshape(n_offsets, -1).sum(axis=0)
    count = len(offsets)
    return (accum.reshape(rows, cols, -1) / count).astype(np.float32)


def foreground_mask(images: np.ndarray) -> np.ndarray:
    """Match ``prepare_observations._group_mask`` semantics."""

    images = np.asarray(images)
    mean_image = images.mean(axis=0)
    mask = np.isfinite(mean_image) & (mean_image > float(mean_image.mean()) / 5.0)
    if not np.any(mask):
        raise ValueError("foreground mask is empty")
    return mask.astype(bool)


def _footprint_bounds_extrema(shape, origin, spacing, group, inplane_samples, thickness_samples) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact voxel-coordinate extrema for renderer midpoint nodes, in O(1) memory.

    The renderer samples an affine Cartesian product.  Each coordinate is
    linear in row, column, and slice offsets, so all extrema are attained at
    the two outermost *midpoint* nodes of each dimension (eight points).
    """
    rows, cols = group["image_shape_rows_cols"]
    affine = np.asarray(group["affine_lps_rc"], float)
    ro = ((np.arange(inplane_samples) + 0.5) / inplane_samples - 0.5)
    co = ((np.arange(inplane_samples) + 0.5) / inplane_samples - 0.5)
    so = ((np.arange(thickness_samples) + 0.5) / thickness_samples - 0.5)
    values = ((ro[0], rows - 1 + ro[-1]), (co[0], cols - 1 + co[-1]), (so[0], so[-1]))
    points = np.asarray([
        affine[:3, 3] + row * affine[:3, 0] + col * affine[:3, 1] + slc * affine[:3, 2]
        for row in values[0] for col in values[1] for slc in values[2]
    ])
    coords = (points - np.asarray(origin, float)) / np.asarray(spacing, float)
    minimum, maximum = coords.min(axis=0), coords.max(axis=0)
    margin = float(np.minimum(minimum, (np.asarray(shape[:3], float) - 1) - maximum).min())
    return minimum, maximum, margin


def _assert_footprint_in_bounds(shape, origin, spacing, group, inplane_samples, thickness_samples) -> float:
    minimum, maximum, margin = _footprint_bounds_extrema(shape, origin, spacing, group, inplane_samples, thickness_samples)
    upper = np.asarray(shape[:3]) - 1
    if np.any(minimum < 0) or np.any(maximum > upper):
        raise ValueError("out-of-volume quadrature footprint")
    return margin


def coverage_preflight(shape, origin, spacing, groups, inplane_samples=6, thickness_samples=9) -> dict:
    records = []
    for group in groups:
        margin = _assert_footprint_in_bounds(shape, origin, spacing, group, inplane_samples, thickness_samples)
        records.append({"group_idx": int(group["group_idx"]), "coverage_fraction": 1.0, "out_of_volume_quadrature_points": 0, "minimum_boundary_margin_voxel": margin})
    return {"gate": True, "groups": records}


def render_groups(signal_volume: np.ndarray, volume_origin_mm: np.ndarray, volume_spacing_mm: np.ndarray, groups: list[dict], inplane_samples: int = 2, thickness_samples: int = 3, require_in_bounds: bool = False) -> tuple[np.ndarray, np.ndarray]:
    images, masks = [], []
    for group in groups:
        if require_in_bounds:
            _assert_footprint_in_bounds(signal_volume.shape, volume_origin_mm, volume_spacing_mm, group, inplane_samples, thickness_samples)
        rows, cols = group["image_shape_rows_cols"]
        image = integrate_signal_volume(signal_volume, volume_origin_mm, volume_spacing_mm, np.asarray(group["affine_lps_rc"]), rows, cols, inplane_samples, thickness_samples)
        images.append(image)
        masks.append(np.repeat(foreground_mask(np.moveaxis(image, -1, 0))[None, ...], image.shape[-1], axis=0))
    # Project contract is (N, rows, cols), one ten-weight block per group.
    return np.concatenate([np.moveaxis(x, -1, 0) for x in images], axis=0), np.concatenate(masks, axis=0)
