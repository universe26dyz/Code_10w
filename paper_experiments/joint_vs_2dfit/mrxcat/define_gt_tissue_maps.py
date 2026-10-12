"""Controlled tissue-map contract for Phase 1 (M0=1, no PD map)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TissueValues:
    lv_myocardium_t1_ms: float = 1410.0
    lv_myocardium_t2_ms: float = 50.0
    lv_myocardium_b1: float = 1.0
    blood_t1_ms: float = 1500.0
    blood_t2_ms: float = 120.0
    muscle_t1_ms: float = 1000.0
    muscle_t2_ms: float = 50.0
    fat_t1_ms: float = 350.0
    fat_t2_ms: float = 70.0
    bone_t1_ms: float = 250.0
    bone_t2_ms: float = 20.0
    air_t1_ms: float = 1171.0
    air_t2_ms: float = 61.0


@dataclass(frozen=True)
class GTMaps:
    labels: np.ndarray
    t1_ms: np.ndarray
    t2_ms: np.ndarray
    b1: np.ndarray
    myocardium_mask: np.ndarray
    tissue_values: TissueValues
    m0: float = 1.0


def build_gt_maps(labels: np.ndarray, values: TissueValues | None = None) -> GTMaps:
    values = values or TissueValues()
    labels = np.asarray(labels)
    if labels.ndim != 3 and labels.ndim != 2:
        raise ValueError("labels must be a 2-D or 3-D integer array")
    # XCAT/MRXCAT labels documented in MakePhantom.py: 1 LV wall, 2 RV wall,
    # 5 LV blood, 6 RV blood, 7/8 atrial blood, 36/50 optional aorta/pericardium.
    t1 = np.full(labels.shape, values.air_t1_ms, dtype=np.float32)
    t2 = np.full(labels.shape, values.air_t2_ms, dtype=np.float32)
    b1 = np.ones(labels.shape, dtype=np.float32)
    muscle = np.isin(labels, [2, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35])
    blood = np.isin(labels, [5, 6, 7, 8])
    fat = np.isin(labels, [50])
    bone = np.isin(labels, [36])
    t1[muscle], t2[muscle] = values.muscle_t1_ms, values.muscle_t2_ms
    t1[blood], t2[blood] = values.blood_t1_ms, values.blood_t2_ms
    t1[fat], t2[fat] = values.fat_t1_ms, values.fat_t2_ms
    t1[bone], t2[bone] = values.bone_t1_ms, values.bone_t2_ms
    myocardium = labels == 1
    t1[myocardium], t2[myocardium] = values.lv_myocardium_t1_ms, values.lv_myocardium_t2_ms
    if not np.all(np.isfinite(t1)) or not np.all(np.isfinite(t2)):
        raise ValueError("GT maps must be finite")
    return GTMaps(labels.astype(np.int16, copy=False), t1, t2, b1, myocardium, values)


def save_gt_maps(maps: GTMaps, output: str) -> None:
    from pathlib import Path

    path = Path(output)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, labels=maps.labels, t1_ms=maps.t1_ms, t2_ms=maps.t2_ms, b1=maps.b1, myocardium_mask=maps.myocardium_mask)
