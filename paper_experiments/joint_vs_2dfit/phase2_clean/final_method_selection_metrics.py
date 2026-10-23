"""CYJ-only Q002/K8, Q005/K4, and Fit-first common-support selection evidence."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .contracts import write_immutable_json

SCHEMA = "phase4r_method_selection_common_support/v1"
SUPPORT_EQUATION = "primary_native_mask AND finite(reference) AND Q002_endpoint_valid AND Q005_endpoint_valid AND Fit_first_endpoint_valid AND finite(Q002) AND finite(Q005) AND finite(Fit_first)"


def _metrics(reference: np.ndarray, prediction: np.ndarray, support: np.ndarray) -> dict[str, float]:
    delta = np.asarray(prediction)[support] - np.asarray(reference)[support]
    return {"rmse": float(np.sqrt(np.mean(delta ** 2))), "mae": float(np.mean(np.abs(delta))), "bias": float(np.mean(delta))}


def evaluate_method_selection_common_support(*, reference: np.ndarray, q002: np.ndarray, q005: np.ndarray, fit_first: np.ndarray, primary_native_mask: np.ndarray, q002_endpoint_valid: np.ndarray, q005_endpoint_valid: np.ndarray, fit_first_endpoint_valid: np.ndarray, endpoint: str, parameter: str, output: str | Path) -> Path:
    """Write one immutable metric record; endpoint and parameter are never pooled."""
    if endpoint not in {"central", "psf128"} or parameter not in {"t1_ms", "t2_ms"}:
        raise ValueError("method selection metrics require central/psf128 and t1_ms/t2_ms identity")
    arrays = [np.asarray(value) for value in (reference, q002, q005, fit_first, primary_native_mask, q002_endpoint_valid, q005_endpoint_valid, fit_first_endpoint_valid)]
    if len({array.shape for array in arrays}) != 1: raise ValueError("method selection arrays must share native-plane shape")
    target = Path(output)
    if target.exists(): raise FileExistsError(f"refusing overwrite: {target}")
    support = np.asarray(primary_native_mask, bool) & np.isfinite(reference) & np.asarray(q002_endpoint_valid, bool) & np.asarray(q005_endpoint_valid, bool) & np.asarray(fit_first_endpoint_valid, bool) & np.isfinite(q002) & np.isfinite(q005) & np.isfinite(fit_first)
    if not support.any(): raise ValueError("method selection common support is empty")
    target.parent.mkdir(parents=True, exist_ok=True)
    write_immutable_json(target, {"schema": SCHEMA, "subject": "CYJ", "endpoint": endpoint, "parameter": parameter, "support_equation": SUPPORT_EQUATION, "support_voxel_count": int(support.sum()), "coverage_fraction": float(support.sum() / np.asarray(primary_native_mask, bool).sum()) if np.asarray(primary_native_mask, bool).any() else 0.0, "methods": {"Q002S640": _metrics(reference, q002, support), "Q005S640_K4": _metrics(reference, q005, support), "Fit-first": _metrics(reference, fit_first, support)}, "selection_scope": "CYJ method-selection evidence only; not population statistics", "immutable": True})
    return target
