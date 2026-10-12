"""Validate the Python TRAD simulator against the original MATLAB HHZ code."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .simulate_hhz_signals import generate_signal_field


def matlab_parity(
    timing9_ms: np.ndarray,
    tr_ms: float = 2.61,
    vps: int = 87,
    matlab_bin: str = "matlab",
    cases: int = 8,
) -> dict:
    matlab = shutil.which(matlab_bin)
    if matlab is None:
        return {"status": "unresolved_external_gate", "reason": "MATLAB executable unavailable"}
    timing9_ms = np.asarray(timing9_ms, dtype=float)
    if timing9_ms.shape != (9,):
        raise ValueError("timing9_ms must contain nine values")
    root = Path(__file__).resolve().parents[3]
    utility = root / "trad/modules/module_01_preprocess_matlab/hhz_original/Utility"
    rng = np.random.default_rng(20261012)
    t1 = np.linspace(550.0, 1850.0, cases)
    t2 = np.linspace(20.0, 130.0, cases)
    b1 = np.linspace(0.25, 1.15, cases)
    with tempfile.TemporaryDirectory(prefix="phase1_hhz_parity_") as td:
        td = Path(td)
        mat_out = td / "reference.mat"
        script = td / "run_reference.m"
        timing_literal = "[" + ";".join(f"{x:.12g}" for x in timing9_ms) + "]"
        t1_literal = "[" + ";".join(f"{x:.12g}" for x in t1) + "]"
        t2_literal = "[" + ";".join(f"{x:.12g}" for x in t2) + "]"
        b1_literal = "[" + ";".join(f"{x:.12g}" for x in b1) + "]"
        script.write_text(
            "addpath('" + str(utility).replace("'", "''") + "');\n"
            f"T1={t1_literal}; T2={t2_literal}; B1={b1_literal};\n"
            f"timing9_ms={timing_literal}; TR={float(tr_ms):.12g}; n_ex={int(vps)}; FA=[45,45,45];\n"
            "PP_delay=zeros(10,2); PP_delay(1,1)=50; PP_delay(5,1)=150; PP_delay(8,2)=35; PP_delay(9,2)=45; PP_delay(10,2)=55;\n"
            "signal_raw=zeros(numel(T1),10);\n"
            "for k=1:numel(T1), signal_raw(k,:)=sim_T1T2_10HB_bssfp(n_ex,TR,[T1(k),T2(k),B1(k)],10,PP_delay,[0;timing9_ms],FA); end\n"
            f"save('{str(mat_out).replace(chr(39), chr(39)*2)}','T1','T2','B1','timing9_ms','n_ex','TR','FA','signal_raw');\n"
        )
        proc = subprocess.run([matlab, "-batch", f"run('{script.as_posix()}')"], text=True, capture_output=True, timeout=180)
        if proc.returncode != 0:
            return {"status": "unresolved_external_gate", "returncode": int(proc.returncode), "reason": proc.stderr[-2000:] or "MATLAB exited without stderr", "stdout": proc.stdout[-2000:]}
        mat = loadmat(mat_out)
    python_raw = generate_signal_field(t1, t2, b1, timing9_ms, tr_ms=tr_ms, vps=vps)
    matlab_raw = np.asarray(mat["signal_raw"], dtype=float)
    err = np.abs(python_raw.astype(float) - matlab_raw)
    python_norm = python_raw / np.linalg.norm(python_raw, axis=1, keepdims=True)
    matlab_norm = matlab_raw / np.linalg.norm(matlab_raw, axis=1, keepdims=True)
    cosine = np.sum(python_norm * matlab_norm, axis=1)
    result = {
        "status": "pass" if float(err.max()) <= 1e-5 and float(cosine.min()) >= 0.999999 else "fail",
        "max_abs_raw_error": float(err.max()),
        "min_normalized_cosine": float(cosine.min()),
        "all_finite": bool(np.all(np.isfinite(python_raw)) and np.all(np.isfinite(matlab_raw))),
        "cases": cases,
        "tr_ms": tr_ms,
        "vps": vps,
    }
    if result["status"] == "fail":
        raise AssertionError(json.dumps(result))
    return result
