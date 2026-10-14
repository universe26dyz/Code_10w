"""Explicit raw-XCAT to Code_10w xyz/LPS/1-mm canonicalization."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import zoom
from .contracts import Geometry

def canonicalize_volume(data: np.ndarray, *, shape: tuple[int, int, int], spacing_mm: tuple[float, float, float], origin_mm: tuple[float, float, float], axis_order: str, world_convention: str, interpolation: str) -> tuple[np.ndarray, Geometry]:
    array = np.asarray(data)
    if array.shape != tuple(shape): raise ValueError("raw XCAT data shape does not match declared shape")
    order = tuple(axis_order.index(axis) for axis in "xyz")
    array = np.transpose(array, order); spacing = np.asarray([spacing_mm[i] for i in order], float); origin = np.asarray(origin_mm, float)
    if world_convention == "RAS":
        array = np.flip(array, axis=(0, 1)); origin[:2] = -origin[:2] - (np.asarray(array.shape[:2]) - 1) * spacing[:2]
    elif world_convention != "LPS": raise ValueError("raw XCAT world_convention must be RAS or LPS")
    target = tuple(int(round((n - 1) * s)) + 1 for n, s in zip(array.shape, spacing))
    if array.shape != target: array = zoom(array, np.asarray(target) / np.asarray(array.shape), order=1 if interpolation == "linear" else 0, mode="nearest", prefilter=False)
    return array.astype(np.float32, copy=False), Geometry(target, (1.0, 1.0, 1.0), tuple(origin), "xyz", "LPS")
