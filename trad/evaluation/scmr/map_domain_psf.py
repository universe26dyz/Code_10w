"""Shared map-domain PSF-matched reprojection of exported quantitative maps.

This module is deliberately spatial-only: it samples already exported T1/T2
NIfTI maps in physical RAS millimetres.  It must not be used as a scanner or
signal simulation.  It calls vendored NeSVoR ``get_PSF`` (the same kernel used
by corrected evaluation-v2 ``simulate_slices``) and uses K deterministic
draws from that discrete kernel.  K is an integration count, not a physical
kernel-width parameter.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import map_coordinates


# Exact constants from vendored ``third_party/nesvor/nesvor/utils/psf.py``.
_GAUSSIAN_FWHM = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
_SINC_FWHM = 1.206709128803223 * _GAUSSIAN_FWHM


@dataclass(frozen=True)
class NativeGrid:
    """One native map plane in RAS-mm.

    ``affine_ras_rc`` maps zero-based ``[row, col, slice, 1]`` to RAS-mm.
    Its third column gives the local slice-normal direction and thickness.
    """

    shape_rc: tuple[int, int]
    affine_ras_rc: np.ndarray
    resolution_xyz_mm: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        affine = np.asarray(self.affine_ras_rc, dtype=np.float64)
        resolution = np.asarray(self.resolution_xyz_mm, dtype=np.float64)
        if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
            raise ValueError("affine_ras_rc must be a finite 4x4 matrix.")
        if len(self.shape_rc) != 2 or min(self.shape_rc) < 1:
            raise ValueError("shape_rc must contain two positive dimensions.")
        if resolution.shape != (3,) or np.any(~np.isfinite(resolution)) or np.any(resolution <= 0):
            raise ValueError("resolution_xyz_mm must be three positive finite values.")
        # Local x/y/z are col/row/slice, while affine columns are row/col/slice.
        expected = np.asarray([resolution[1], resolution[0], resolution[2]])
        actual = np.linalg.norm(affine[:3, :3], axis=0)
        if not np.allclose(actual, expected, rtol=1e-4, atol=1e-4):
            raise ValueError(f"Native affine spacing {actual} disagrees with [row,col,thickness] {expected}.")
        axes = affine[:3, :3] / actual
        if not np.allclose(axes.T @ axes, np.eye(3), rtol=1e-4, atol=1e-4):
            raise ValueError("Native affine axes must be orthonormal in physical space.")
        object.__setattr__(self, "affine_ras_rc", affine)
        object.__setattr__(self, "resolution_xyz_mm", resolution)


@dataclass(frozen=True)
class ReprojectionResult:
    values: np.ndarray
    support: np.ndarray


def nesvor_sigma_mm(resolution_xyz_mm: np.ndarray) -> np.ndarray:
    """Return NeSVoR Gaussian standard deviations for local [x,y,z] mm."""

    resolution = np.asarray(resolution_xyz_mm, dtype=np.float64)
    if resolution.shape != (3,) or np.any(resolution <= 0) or not np.all(np.isfinite(resolution)):
        raise ValueError("resolution_xyz_mm must be three positive finite values.")
    return resolution * np.asarray([_SINC_FWHM, _SINC_FWHM, _GAUSSIAN_FWHM])


def _nesvor_psf_offsets_mm(
    native_resolution_xyz_mm: np.ndarray, volume_resolution_mm: float, n_pixels: int, n_samples: int, seed: int
) -> np.ndarray:
    """Draw local [x,y,z] offsets from NeSVoR's actual discrete ``get_PSF``."""

    try:
        from third_party.nesvor.nesvor.utils.psf import get_PSF
    except ImportError:
        try:
            from trad.third_party.nesvor.nesvor.utils.psf import get_PSF
        except ImportError as exc:
            raise RuntimeError("Vendored NeSVoR/Torch is required for common map-domain PSF reprojection.") from exc
    kernel = get_PSF(
        res_ratio=tuple((np.asarray(native_resolution_xyz_mm, dtype=np.float64) / volume_resolution_mm).tolist()),
        device="cpu",
    ).detach().cpu().numpy()
    weights = kernel.reshape(-1).astype(np.float64)
    rng = np.random.default_rng(int(seed))
    selected = rng.choice(weights.size, size=(n_pixels, int(n_samples)), p=weights / weights.sum())
    z, y, x = np.unravel_index(selected, kernel.shape)
    centre = (np.asarray(kernel.shape, dtype=np.float64) - 1.0) / 2.0
    return np.stack((x - centre[2], y - centre[1], z - centre[0]), axis=-1) * volume_resolution_mm


def _pixel_centres_and_axes(grid: NativeGrid) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.mgrid[0 : grid.shape_rc[0], 0 : grid.shape_rc[1]]
    affine = grid.affine_ras_rc
    centres = affine[:3, 3] + rows[..., None] * affine[:3, 0] + cols[..., None] * affine[:3, 1]
    # local sample axes [x=column, y=row, z=through-plane]
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    axes = np.column_stack((affine[:3, 1] / spacing[1], affine[:3, 0] / spacing[0], affine[:3, 2] / spacing[2]))
    return centres, axes


def reproject_volume_to_native_map_psf(
    volume: np.ndarray,
    volume_affine_ras: np.ndarray,
    native_grid: NativeGrid,
    *,
    n_samples: int,
    seed: int,
) -> ReprojectionResult:
    """Integrate an exported 3-D map over a final-pose native pixel PSF.

    The same function is used for Trad and 2D-fit-first.  Each pixel gets K
    independent draws from the vendored NeSVoR ``get_PSF`` discrete kernel in
    native local axes; invalid samples are excluded from that pixel's mean and
    reported by ``support``.
    """

    image = np.asarray(volume, dtype=np.float64)
    affine = np.asarray(volume_affine_ras, dtype=np.float64)
    if image.ndim != 3 or min(image.shape) < 1:
        raise ValueError("volume must be a non-empty 3-D array.")
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)) or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ValueError("volume_affine_ras must be an invertible finite 4x4 matrix.")
    if n_samples < 1:
        raise ValueError("n_samples must be at least one.")

    volume_spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if not np.allclose(volume_spacing, volume_spacing[0], rtol=1e-4, atol=1e-4):
        raise ValueError("Common NeSVoR PSF reprojection requires an isotropic exported volume grid.")
    centres, axes = _pixel_centres_and_axes(native_grid)
    offsets = _nesvor_psf_offsets_mm(
        native_grid.resolution_xyz_mm, float(volume_spacing[0]), int(np.prod(native_grid.shape_rc)), n_samples, seed
    ).reshape(*native_grid.shape_rc, int(n_samples), 3)
    world = centres[..., None, :] + np.einsum("...ki,ji->...kj", offsets, axes)
    world_flat = world.reshape(-1, 3)
    voxel = (np.linalg.inv(affine) @ np.c_[world_flat, np.ones(world_flat.shape[0])].T).T[:, :3]
    sampled = map_coordinates(image, voxel.T, order=1, mode="constant", cval=np.nan, prefilter=False).reshape(*native_grid.shape_rc, n_samples)
    valid = np.isfinite(sampled)
    counts = valid.sum(axis=-1)
    total = np.where(valid, sampled, 0.0).sum(axis=-1)
    values = np.full(native_grid.shape_rc, np.nan, dtype=np.float32)
    supported = counts > 0
    values[supported] = (total[supported] / counts[supported]).astype(np.float32)
    return ReprojectionResult(values=values, support=supported)


def _reproject_method(
    volumes: dict[str, tuple[np.ndarray, np.ndarray]],
    grids_by_parameter: dict[str, list[NativeGrid]],
    *,
    n_samples: int,
    seed: int,
) -> dict[str, list[ReprojectionResult]]:
    """Method-neutral wrapper; both public method wrappers call this core."""

    output: dict[str, list[ReprojectionResult]] = {}
    for parameter in ("T1", "T2"):
        image, affine = volumes[parameter]
        # Parameter-specific seed prevents T1/T2 from accidentally sharing a
        # random stream while remaining repeatable across methods.
        parameter_seed = int(seed) + (0 if parameter == "T1" else 1_000_003)
        output[parameter] = [
            reproject_volume_to_native_map_psf(image, affine, grid, n_samples=n_samples, seed=parameter_seed + index)
            for index, grid in enumerate(grids_by_parameter[parameter])
        ]
    return output


def reproject_trad_exported_maps(
    volumes: dict[str, tuple[np.ndarray, np.ndarray]], grids: Iterable[NativeGrid], *, n_samples: int, seed: int
) -> dict[str, list[ReprojectionResult]]:
    """Trad wrapper: exported maps plus prepared geometry/final group poses only."""

    native = list(grids)
    return _reproject_method(volumes, {"T1": native, "T2": native}, n_samples=n_samples, seed=seed)


def reproject_2dfit_exported_maps(
    volumes: dict[str, tuple[np.ndarray, np.ndarray]], grids_by_parameter: dict[str, list[NativeGrid]], *, n_samples: int, seed: int
) -> dict[str, list[ReprojectionResult]]:
    """2D-fit-first wrapper: exported maps plus its parameter-specific final grids."""

    if set(grids_by_parameter) != {"T1", "T2"}:
        raise ValueError("2D-fit-first grids must be supplied separately for T1 and T2.")
    return _reproject_method(volumes, grids_by_parameter, n_samples=n_samples, seed=seed)


def load_nifti_map(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("nibabel is required to read exported quantitative NIfTI maps.") from exc
    image = nib.load(str(path))
    data = np.asarray(image.get_fdata(dtype=np.float32))
    if data.ndim != 3:
        raise ValueError(f"{path} must be a 3-D NIfTI image, got {data.shape}.")
    return data, np.asarray(image.affine, dtype=np.float64)


def grid_from_registered_nifti(path: str | Path) -> NativeGrid:
    """Read one final registered 2D-fit-first slice as an RAS native grid."""

    data, affine = load_nifti_map(path)
    if data.shape[2] != 1:
        raise ValueError(f"Registered native slice must have singleton z dimension: {path}")
    # The corrected evaluation-v2 registered-slice NIfTIs are already final
    # geometry.  Their first two NIfTI axes are retained as [row, col] here.
    resolution_rcz = np.linalg.norm(affine[:3, :3], axis=0)
    return NativeGrid((int(data.shape[0]), int(data.shape[1])), affine, np.asarray([resolution_rcz[1], resolution_rcz[0], resolution_rcz[2]]), label=str(path))


def grids_from_trad_prepared(prepared_paths: Iterable[str | Path], final_poses_path: str | Path) -> list[NativeGrid]:
    """Create one final RAS grid per Trad group from prepared geometry/poses."""

    payload = json.loads(Path(final_poses_path).read_text(encoding="utf-8"))
    if payload.get("coordinate_convention") != "physical RAS mm; trans_first=true; world=R@(local+T)":
        raise ValueError("final poses do not declare the required physical RAS trans_first convention.")
    poses = np.asarray(payload.get("matrix_physical"), dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (3, 4):
        raise ValueError("final_rigid_poses.json matrix_physical must be [group,3,4].")
    grids: list[NativeGrid] = []
    offset = 0
    for source_text in prepared_paths:
        source = Path(source_text)
        with np.load(source, allow_pickle=False) as data:
            required = {"images", "group_idx", "weight_idx", "affine_lps_rc", "pixel_spacing_rc_mm", "slice_thickness_mm"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"{source} lacks prepared geometry keys {sorted(missing)}.")
            images = np.asarray(data["images"])
            groups = np.asarray(data["group_idx"], dtype=np.int64)
            weights = np.asarray(data["weight_idx"], dtype=np.int64)
            affine_lps = np.asarray(data["affine_lps_rc"], dtype=np.float64)
            spacing_rc = np.asarray(data["pixel_spacing_rc_mm"], dtype=np.float64)
            thickness = np.asarray(data["slice_thickness_mm"], dtype=np.float64)
        if images.ndim != 3 or groups.shape != (images.shape[0],) or weights.shape != groups.shape:
            raise ValueError(f"{source} prepared image/group shapes are inconsistent.")
        for group in np.unique(groups):
            indices = np.flatnonzero(groups == group)
            if indices.size != 10 or not np.array_equal(np.sort(weights[indices]), np.arange(10)):
                raise ValueError(f"{source} group {group} must contain exactly weights 0..9.")
            first = int(indices[0])
            if not (np.allclose(affine_lps[indices], affine_lps[first]) and np.allclose(spacing_rc[indices], spacing_rc[first]) and np.allclose(thickness[indices], thickness[first])):
                raise ValueError(f"{source} group {group} does not share one native geometry.")
            pose_index = offset + int(group)
            if pose_index >= len(poses):
                raise ValueError("Prepared groups exceed final physical pose count.")
            row_spacing, col_spacing = spacing_rc[first]
            resolution = np.asarray([col_spacing, row_spacing, thickness[first]], dtype=np.float64)
            # Mirror the validated geometry adapter's local-frame guard.  The
            # prepared DICOM affine is [row,col,slice] in LPS; its axes must
            # agree with recorded [row,col,thickness] physical spacings.
            prepared_spacing = np.linalg.norm(affine_lps[first, :3, :3], axis=0)
            if not np.allclose(prepared_spacing, [row_spacing, col_spacing, thickness[first]], rtol=1e-4, atol=1e-4):
                raise ValueError(f"{source} group {group} prepared affine/resolution mismatch.")
            prepared_axes = affine_lps[first, :3, :3] / prepared_spacing
            if not np.allclose(prepared_axes.T @ prepared_axes, np.eye(3), rtol=1e-4, atol=1e-4):
                raise ValueError(f"{source} group {group} prepared affine is not an orthonormal DICOM local frame.")
            pose = poses[pose_index]
            rotation, trans_first = pose[:, :3], pose[:, 3]
            centre_world = rotation @ trans_first
            h, w = images.shape[1:]
            native_affine = np.eye(4, dtype=np.float64)
            native_affine[:3, 0] = rotation[:, 1] * row_spacing
            native_affine[:3, 1] = rotation[:, 0] * col_spacing
            native_affine[:3, 2] = rotation[:, 2] * thickness[first]
            native_affine[:3, 3] = centre_world - native_affine[:3, 0] * ((h - 1) / 2.0) - native_affine[:3, 1] * ((w - 1) / 2.0)
            grids.append(NativeGrid((int(h), int(w)), native_affine, resolution, label=f"{source}:{int(group)}"))
        offset += int(np.unique(groups).size)
    if offset != len(poses):
        raise ValueError(f"Prepared group count {offset} differs from final pose count {len(poses)}.")
    return grids


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
