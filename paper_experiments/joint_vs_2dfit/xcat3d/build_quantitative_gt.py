"""Build Phase-1.2 quantitative maps without any external mesh overlay."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .contracts import FROZEN_B1, FROZEN_LV_T1_MS, FROZEN_LV_T2_MS, QuantitativeGT, XCATInputs


def build_quantitative_gt(inputs: XCATInputs) -> QuantitativeGT:
    t1 = np.asarray(inputs.t1_ms.data, dtype=np.float32)
    t2 = np.asarray(inputs.t2_ms.data, dtype=np.float32)
    myocardium = np.asarray(inputs.myocardium_mask.data, dtype=bool)
    support = np.asarray(inputs.signal_support.data, dtype=bool)
    if not np.any(myocardium):
        raise ValueError("single-source XCAT myocardium support is empty")
    if np.any(myocardium & ~support):
        raise ValueError("LV myocardium must lie within the XCAT signal support")
    if not np.all(t1[myocardium] == FROZEN_LV_T1_MS):
        raise ValueError(f"LV myocardium T1 must be exactly {FROZEN_LV_T1_MS:g} ms in the XCAT source")
    if not np.all(t2[myocardium] == FROZEN_LV_T2_MS):
        raise ValueError(f"LV myocardium T2 must be exactly {FROZEN_LV_T2_MS:g} ms in the XCAT source")
    policy = inputs.manifest.get("m0_policy")
    if policy == "controlled_binary_signal_support_m0_1":
        m0 = support.astype(np.float32)
    elif policy == "native_xcat_pd_m0":
        if inputs.pd_m0 is None:
            raise ValueError("native_xcat_pd_m0 policy requires pd_m0")
        m0 = np.asarray(inputs.pd_m0.data, dtype=np.float32)
        if np.any(m0 < 0):
            raise ValueError("XCAT PD/M0 must be non-negative")
        support = m0 > 0
    else:
        raise ValueError("m0_policy must be controlled_binary_signal_support_m0_1 or native_xcat_pd_m0")
    b1 = np.full(t1.shape, FROZEN_B1, dtype=np.float32)
    return QuantitativeGT(t1, t2, b1, m0, support, myocardium, inputs.t1_ms.geometry, inputs.manifest)


def write_quantitative_gt(gt: QuantitativeGT, output_root: str | Path) -> Path:
    root = Path(output_root)
    path = root / "gt_maps.npz"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        t1_ms=gt.t1_ms,
        t2_ms=gt.t2_ms,
        b1=gt.b1,
        m0=gt.m0,
        signal_support_mask=gt.signal_support_mask,
        myocardium_mask=gt.myocardium_mask,
        origin_mm=np.asarray(gt.geometry.origin_mm),
        spacing_mm=np.asarray(gt.geometry.spacing_mm),
        axis_order=np.asarray(gt.geometry.axis_order),
        world_convention=np.asarray(gt.geometry.world_convention),
    )
    return path
