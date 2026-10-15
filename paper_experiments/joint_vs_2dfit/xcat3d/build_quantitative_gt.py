"""Build Phase-1.2 quantitative maps without any external mesh overlay."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from .canonicalize import canonicalize_volume
from .contracts import FROZEN_B1, FROZEN_LV_T1_MS, FROZEN_LV_T2_MS, QuantitativeGT, XCATInputs


def _validate_cardiac_roi_coverage(myocardium: np.ndarray, geometry, inputs: XCATInputs) -> None:
    record = inputs.manifest["xcat"]["geometry_spec"]
    specification = json.loads((inputs.manifest_path.parent / record["path"]).read_text())
    margin_mm = float(specification["minimum_lv_boundary_margin_mm"])
    if not np.isfinite(margin_mm) or margin_mm <= 0:
        raise ValueError("cardiac ROI geometry must define a positive LV boundary margin")
    lower = np.argwhere(myocardium).min(axis=0)
    upper = np.asarray(myocardium.shape) - 1 - np.argwhere(myocardium).max(axis=0)
    margins_mm = np.minimum(lower, upper) * np.asarray(geometry.spacing_mm)
    if np.any(margins_mm < margin_mm):
        raise ValueError(f"LV myocardium boundary margin is below required {margin_mm:g} mm")


def build_quantitative_gt(inputs: XCATInputs) -> QuantitativeGT:
    raw = inputs.t1_ms.geometry
    kwargs = {"shape": raw.shape, "spacing_mm": raw.spacing_mm, "origin_mm": raw.origin_mm, "axis_order": raw.axis_order, "world_convention": raw.world_convention}
    t1, geometry = canonicalize_volume(inputs.t1_ms.data, **kwargs, interpolation="nearest")
    t2, _ = canonicalize_volume(inputs.t2_ms.data, **kwargs, interpolation="nearest")
    myocardium, _ = canonicalize_volume(inputs.myocardium_mask.data, **kwargs, interpolation="nearest")
    support, _ = canonicalize_volume(inputs.signal_support.data, **kwargs, interpolation="nearest")
    myocardium, support = myocardium.astype(bool), support.astype(bool)
    if not np.any(myocardium):
        raise ValueError("single-source XCAT myocardium support is empty")
    _validate_cardiac_roi_coverage(myocardium, geometry, inputs)
    if np.any(myocardium & ~support):
        raise ValueError("LV myocardium must lie within the XCAT signal support")
    if not np.all(t1[myocardium] == FROZEN_LV_T1_MS):
        raise ValueError(f"LV myocardium T1 must be exactly {FROZEN_LV_T1_MS:g} ms in the XCAT source")
    if not np.all(t2[myocardium] == FROZEN_LV_T2_MS):
        raise ValueError(f"LV myocardium T2 must be exactly {FROZEN_LV_T2_MS:g} ms in the XCAT source")
    policy = inputs.manifest.get("m0_policy")
    if policy != "controlled_binary_signal_support_m0_1":
        raise ValueError("formal Phase-1.2 requires controlled_binary_signal_support_m0_1")
    m0 = support.astype(np.float32)
    b1 = np.full(t1.shape, FROZEN_B1, dtype=np.float32)
    return QuantitativeGT(t1, t2, b1, m0, support, myocardium, geometry, inputs.manifest)


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
