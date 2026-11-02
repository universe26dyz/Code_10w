"""HHZ 10-heartbeat timing contract, translated without changing its physics."""

from __future__ import annotations

import numpy as np


def compute_duration_before_acq(
    acquisition_times_ms: np.ndarray,
    tr_ms: float,
    n_ex: float,
    ti_ms: np.ndarray,
    t2prep_ms: np.ndarray,
) -> np.ndarray:
    """Return HHZ ``Duration_befor_Acq`` with shape ``[10]`` in milliseconds.

    This is the timing block from ``function_T1T2_10HB_bssfp.m``: consecutive
    acquisition intervals lose ``TR * n_ex`` and HB5/HB8/HB9/HB10 additionally
    lose their separately modelled preparation delays. HB1 is always zero.
    """

    acquisition_times_ms = np.asarray(acquisition_times_ms, dtype=np.float64)
    ti_ms = np.asarray(ti_ms, dtype=np.float64)
    t2prep_ms = np.asarray(t2prep_ms, dtype=np.float64)
    if acquisition_times_ms.shape != (10,):
        raise ValueError(
            "acquisition_times_ms must contain sorted HB1..HB10 times with shape (10,), "
            f"got {acquisition_times_ms.shape}."
        )
    if not np.all(np.isfinite(acquisition_times_ms)) or not np.all(
        np.diff(acquisition_times_ms) > 0
    ):
        raise ValueError("acquisition_times_ms must be finite and strictly increasing.")
    if not np.isfinite(tr_ms) or tr_ms <= 0 or not np.isfinite(n_ex) or n_ex <= 0:
        raise ValueError("tr_ms and n_ex must be positive finite values.")
    if ti_ms.shape != (2,) or not np.all(np.isfinite(ti_ms)):
        raise ValueError(f"ti_ms must have shape (2,), got {ti_ms.shape}.")
    if t2prep_ms.shape != (3,) or not np.all(np.isfinite(t2prep_ms)):
        raise ValueError(f"t2prep_ms must have shape (3,), got {t2prep_ms.shape}.")

    duration_before_acq = np.zeros(10, dtype=np.float64)
    duration_before_acq[1:] = np.diff(acquisition_times_ms) - float(tr_ms) * float(n_ex)
    duration_before_acq[4] -= ti_ms[1]
    duration_before_acq[7:] -= t2prep_ms
    return duration_before_acq
