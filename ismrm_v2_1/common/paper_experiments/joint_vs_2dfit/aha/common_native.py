"""Shared physical-native-plane sampling for paired reconstruction comparison.

The reconstructed volumes already contain each method's optimized final pose.
Pairwise visualisation and metrics must therefore sample *both* volumes on one
immutable acquisition-native physical plane; sampling each volume on its own
optimized output-slice plane would compare different anatomy pixel by pixel.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..phase2_clean.evaluation import reproject_volume_to_native_plane


def sample_common_native_plane(*, reference: np.ndarray, native_affine_ras: np.ndarray,
                               proposed_volume: np.ndarray, proposed_affine_ras: np.ndarray,
                               fit_first_volume: np.ndarray, fit_first_affine_ras: np.ndarray) -> dict[str, np.ndarray]:
    """Reproject Proposed and Fit-first onto exactly one native physical grid."""
    reference = np.asarray(reference, float)
    if reference.ndim != 2:
        raise ValueError("common native sampling requires a 2-D reference plane")
    proposed, proposed_valid = reproject_volume_to_native_plane(
        proposed_volume, proposed_affine_ras, native_affine_ras, reference.shape)
    fit_first, fit_first_valid = reproject_volume_to_native_plane(
        fit_first_volume, fit_first_affine_ras, native_affine_ras, reference.shape)
    return {
        "reference": reference,
        "proposed": np.asarray(proposed, float),
        "fit_first": np.asarray(fit_first, float),
        "proposed_valid": np.asarray(proposed_valid, bool) & np.isfinite(proposed),
        "fit_first_valid": np.asarray(fit_first_valid, bool) & np.isfinite(fit_first),
    }


def load_volume_images(*, proposed_t1: str, proposed_t2: str, fit_first_t1: str,
                       fit_first_t2: str) -> dict[str, dict[str, Any]]:
    """Load the completed final-volume outputs without rerunning reconstruction."""
    import nibabel as nib

    paths = {
        "t1": {"proposed": proposed_t1, "fit_first": fit_first_t1},
        "t2": {"proposed": proposed_t2, "fit_first": fit_first_t2},
    }
    result: dict[str, dict[str, Any]] = {}
    for parameter, items in paths.items():
        result[parameter] = {}
        for method, path in items.items():
            image = nib.load(str(path))
            result[parameter][method] = image
    return result
