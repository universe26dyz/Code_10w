"""Generate clean high-resolution HHZ fingerprints with the validated TRAD simulator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _trad_simulator_class():
    root = Path(__file__).resolve().parents[3]
    path = root / "trad/modules/module_05_signal_decoder/trad_signal_simulator.py"
    spec = importlib.util.spec_from_file_location("phase1_trad_signal_simulator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load validated simulator from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.TradSignalSimulator, module.TradProtocol


def generate_signal_field(
    t1_ms: np.ndarray,
    t2_ms: np.ndarray,
    b1: np.ndarray,
    timing9_ms: np.ndarray,
    tr_ms: float = 2.61,
    vps: int = 87,
    chunk_size: int = 250_000,
    signal_support_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``(..., 10)`` raw signals; no PD or normalization is applied."""

    t1_ms, t2_ms, b1 = (np.asarray(x, dtype=np.float32) for x in (t1_ms, t2_ms, b1))
    if t1_ms.shape != t2_ms.shape or t1_ms.shape != b1.shape:
        raise ValueError("T1, T2 and B1 shapes must match")
    support = np.ones(t1_ms.shape, dtype=bool) if signal_support_mask is None else np.asarray(signal_support_mask, dtype=bool)
    if support.shape != t1_ms.shape:
        raise ValueError("signal_support_mask shape must match tissue maps")
    timing = np.asarray(timing9_ms, dtype=np.float32)
    if timing.shape != (9,):
        raise ValueError("timing9_ms must contain exactly nine values")
    Simulator, Protocol = _trad_simulator_class()
    protocol = Protocol(tr_ms=float(tr_ms), vps=int(vps))
    simulator = Simulator().eval()
    out = np.empty(t1_ms.shape + (10,), dtype=np.float32)
    flat = out.reshape(-1, 10)
    import torch

    for start in range(0, flat.shape[0], chunk_size):
        stop = min(start + chunk_size, flat.shape[0])
        n = stop - start
        active = support.reshape(-1)[start:stop]
        if not np.any(active):
            flat[start:stop] = 0.0
            continue
        # Inactive voxels are deliberately never passed to the simulator; M0
        # is exactly one on support and exactly zero outside it.
        tt1 = torch.from_numpy(t1_ms.reshape(-1)[start:stop][active])
        tt2 = torch.from_numpy(t2_ms.reshape(-1)[start:stop][active])
        bb1 = torch.from_numpy(b1.reshape(-1)[start:stop][active])
        timing_batch = torch.from_numpy(np.broadcast_to(timing, (int(active.sum()), 9)).copy())
        with torch.no_grad():
            values = simulator(tt1, tt2, bb1, timing_batch, protocol, normalize=False)
        block = np.zeros((n, 10), dtype=np.float32)
        block[active] = values.cpu().numpy().astype(np.float32, copy=False)
        flat[start:stop] = block
    if not np.all(np.isfinite(out)):
        raise ValueError("HHZ signal field contains non-finite values")
    return out
