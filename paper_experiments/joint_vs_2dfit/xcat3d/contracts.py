"""Data contracts for externally generated, full-3-D XCAT property volumes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


FROZEN_LV_T1_MS = 1410.0
FROZEN_LV_T2_MS = 50.0
FROZEN_B1 = 1.0


@dataclass(frozen=True)
class Geometry:
    shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    origin_mm: tuple[float, float, float]
    axis_order: str
    world_convention: str

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "Geometry":
        shape = tuple(int(value) for value in record["shape"])
        spacing = tuple(float(value) for value in record["spacing_mm"])
        origin = tuple(float(value) for value in record["origin_mm"])
        axis_order = str(record["axis_order"])
        convention = str(record["world_convention"])
        if len(shape) != 3 or len(spacing) != 3 or len(origin) != 3:
            raise ValueError("XCAT volume geometry must specify exactly three axes")
        if min(shape) <= 1:
            raise ValueError(f"formal XCAT3D route requires a full 3-D volume; got shape {shape}")
        if not all(np.isfinite(spacing)) or min(spacing) <= 0:
            raise ValueError("XCAT spacing_mm must be finite and positive")
        if not all(np.isfinite(origin)):
            raise ValueError("XCAT origin_mm must be finite")
        if sorted(axis_order) != ["x", "y", "z"] or len(axis_order) != 3:
            raise ValueError("axis_order must be a permutation of xyz")
        if not convention:
            raise ValueError("world_convention is required")
        return cls(shape, spacing, origin, axis_order, convention)


@dataclass(frozen=True)
class XCATVolume:
    name: str
    data: np.ndarray
    geometry: Geometry
    source_path: Path


@dataclass(frozen=True)
class XCATInputs:
    manifest_path: Path
    manifest: dict[str, Any]
    t1_ms: XCATVolume
    t2_ms: XCATVolume
    signal_support: XCATVolume
    myocardium_mask: XCATVolume
    pd_m0: XCATVolume | None


@dataclass(frozen=True)
class QuantitativeGT:
    t1_ms: np.ndarray
    t2_ms: np.ndarray
    b1: np.ndarray
    m0: np.ndarray
    signal_support_mask: np.ndarray
    myocardium_mask: np.ndarray
    geometry: Geometry
    source_manifest: dict[str, Any]
