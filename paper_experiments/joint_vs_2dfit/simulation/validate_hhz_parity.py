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


def build_reference_config(
    timing9_ms: np.ndarray,
    tr_ms: float = 2.61,
    vps: int = 87,
    cases: int = 8,
) -> dict:
    """Return the deterministic original-HHZ input contract for MATLAB."""

    timing9_ms = np.asarray(timing9_ms, dtype=float)
    if timing9_ms.shape != (9,):
        raise ValueError("timing9_ms must contain nine values")
    if cases < 1:
        raise ValueError("cases must be positive")
    return {
        "T1": np.linspace(550.0, 1850.0, cases).tolist(),
        "T2": np.linspace(20.0, 130.0, cases).tolist(),
        "B1": np.linspace(0.25, 1.15, cases).tolist(),
        "timing9_ms": np.repeat(timing9_ms[None, :], cases, axis=0).tolist(),
        "n_ex": int(vps),
        "TR": float(tr_ms),
        "FA": [45, 45, 45],
    }


def validate_matlab_reference(reference_mat: str | Path, reference_config: str | Path | dict) -> dict:
    """Validate an explicitly exported original-MATLAB reference.mat.

    This deliberately reads MATLAB's own raw output.  It does not run, or
    treat, the Python simulator as an independent reference.
    """

    config = json.loads(Path(reference_config).read_text()) if isinstance(reference_config, (str, Path)) else reference_config
    t1 = np.asarray(config["T1"], dtype=float)
    t2 = np.asarray(config["T2"], dtype=float)
    b1 = np.asarray(config["B1"], dtype=float)
    timing = np.asarray(config["timing9_ms"], dtype=float)
    if timing.shape != (len(t1), 9) or t2.shape != t1.shape or b1.shape != t1.shape:
        raise ValueError("reference config must contain matching T1/T2/B1 and [N,9] timing9_ms")
    if not np.allclose(timing, timing[0], rtol=0.0, atol=0.0):
        raise ValueError("the Phase-1 parity contract requires one shared timing9_ms template")
    mat = loadmat(reference_mat)
    if "signal_raw" not in mat:
        raise ValueError("reference.mat is missing signal_raw from export_matlab_signal_reference")
    matlab_raw = np.asarray(mat["signal_raw"], dtype=float)
    if matlab_raw.shape != (len(t1), 10):
        raise ValueError(f"reference.mat signal_raw has unexpected shape {matlab_raw.shape}")
    python_raw = generate_signal_field(t1, t2, b1, timing[0], tr_ms=float(config["TR"]), vps=int(config["n_ex"]))
    err = np.abs(python_raw.astype(float) - matlab_raw)
    python_norm = np.linalg.norm(python_raw, axis=1)
    matlab_norm = np.linalg.norm(matlab_raw, axis=1)
    if np.any(python_norm == 0.0) or np.any(matlab_norm == 0.0):
        cosine = np.full(len(t1), np.nan)
    else:
        cosine = np.sum(python_raw * matlab_raw, axis=1) / (python_norm * matlab_norm)
    return {
        "status": "pass" if bool(np.all(np.isfinite(python_raw)) and np.all(np.isfinite(matlab_raw))) and float(err.max()) <= 1e-5 and float(cosine.min()) >= 0.999999 else "fail",
        "max_abs_raw_error": float(err.max()),
        "min_normalized_cosine": float(cosine.min()),
        "all_finite": bool(np.all(np.isfinite(python_raw)) and np.all(np.isfinite(matlab_raw))),
        "cases": int(len(t1)),
        "tr_ms": float(config["TR"]),
        "vps": int(config["n_ex"]),
        "reference_mat": str(Path(reference_mat).resolve()),
    }


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
    config_data = build_reference_config(timing9_ms, tr_ms=tr_ms, vps=vps, cases=cases)
    root = Path(__file__).resolve().parents[3]
    exporter_dir = root / "trad/modules/module_05_signal_decoder"
    with tempfile.TemporaryDirectory(prefix="phase1_hhz_parity_") as td:
        td = Path(td)
        mat_out = td / "reference.mat"
        config = td / "phase1_hhz_parity_config.json"
        config.write_text(json.dumps(config_data, indent=2) + "\n")
        quote = lambda value: str(value).replace("'", "''")
        command = f"addpath('{quote(exporter_dir)}'); export_matlab_signal_reference('{quote(mat_out)}','{quote(config)}');"
        proc = subprocess.run([matlab, "-batch", command], text=True, capture_output=True, timeout=180)
        if proc.returncode != 0:
            return {"status": "unresolved_external_gate", "returncode": int(proc.returncode), "reason": proc.stderr[-2000:] or "MATLAB exited without stderr", "stdout": proc.stdout[-2000:]}
        result = validate_matlab_reference(mat_out, config_data)
    if result["status"] == "fail":
        raise AssertionError(json.dumps(result))
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the original MATLAB HHZ parity gate for a selected Phase-1 timing template.")
    parser.add_argument("--timing-selection", help="timing_selection.json containing timing9_ms")
    parser.add_argument("--output-json", help="refuse-overwrite parity result path")
    parser.add_argument("--matlab-bin", default="matlab")
    parser.add_argument("--write-reference-config", help="write deterministic config for an external original-MATLAB export, then stop")
    parser.add_argument("--reference-config", help="config used by an externally exported reference.mat")
    parser.add_argument("--reference-mat", help="externally exported reference.mat to validate")
    args = parser.parse_args()
    if args.write_reference_config:
        if not args.timing_selection or args.reference_mat or args.reference_config or args.output_json:
            parser.error("--write-reference-config requires only --timing-selection")
        output = Path(args.write_reference_config)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        selection = json.loads(Path(args.timing_selection).read_text())
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(build_reference_config(np.asarray(selection["timing9_ms"], dtype=float), float(selection["tr_ms"]), int(selection["vps"])), indent=2) + "\n")
        print(str(output))
        return
    if bool(args.reference_config) != bool(args.reference_mat) or not args.output_json:
        parser.error("provide --reference-config and --reference-mat with --output-json, or --timing-selection with --output-json")
    output = Path(args.output_json)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if args.reference_mat:
        result = validate_matlab_reference(args.reference_mat, args.reference_config)
    else:
        if not args.timing_selection:
            parser.error("--timing-selection is required for local MATLAB execution")
        selection = json.loads(Path(args.timing_selection).read_text())
        result = matlab_parity(np.asarray(selection["timing9_ms"], dtype=float), tr_ms=float(selection["tr_ms"]), vps=int(selection["vps"]), matlab_bin=args.matlab_bin)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
