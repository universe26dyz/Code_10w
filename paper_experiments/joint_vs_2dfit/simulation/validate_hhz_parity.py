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
    exporter_dir = root / "trad/modules/module_05_signal_decoder"
    rng = np.random.default_rng(20261012)
    t1 = np.linspace(550.0, 1850.0, cases)
    t2 = np.linspace(20.0, 130.0, cases)
    b1 = np.linspace(0.25, 1.15, cases)
    with tempfile.TemporaryDirectory(prefix="phase1_hhz_parity_") as td:
        td = Path(td)
        mat_out = td / "reference.mat"
        config = td / "phase1_hhz_parity_config.json"
        config.write_text(json.dumps({"T1": t1.tolist(), "T2": t2.tolist(), "B1": b1.tolist(), "timing9_ms": np.repeat(timing9_ms[None, :], cases, axis=0).tolist(), "n_ex": int(vps), "TR": float(tr_ms), "FA": [45, 45, 45]}))
        quote = lambda value: str(value).replace("'", "''")
        command = f"addpath('{quote(exporter_dir)}'); export_matlab_signal_reference('{quote(mat_out)}','{quote(config)}');"
        proc = subprocess.run([matlab, "-batch", command], text=True, capture_output=True, timeout=180)
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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the original MATLAB HHZ parity gate for a selected Phase-1 timing template.")
    parser.add_argument("--timing-selection", required=True, help="timing_selection.json containing timing9_ms")
    parser.add_argument("--output-json", required=True, help="refuse-overwrite parity result path")
    parser.add_argument("--matlab-bin", default="matlab")
    args = parser.parse_args()
    output = Path(args.output_json)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    selection = json.loads(Path(args.timing_selection).read_text())
    result = matlab_parity(np.asarray(selection["timing9_ms"], dtype=float), tr_ms=float(selection["tr_ms"]), vps=int(selection["vps"]), matlab_bin=args.matlab_bin)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
