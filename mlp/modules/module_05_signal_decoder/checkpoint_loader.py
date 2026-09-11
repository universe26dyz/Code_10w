"""Strict compatibility loader for a frozen offline MLP decoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .frozen_decoder import FrozenMLPSignalDecoder
from .mlp_model import MdmSignalMLP
from .trad_teacher.trad_signal_simulator import TradProtocol


EXPECTED_RANGES = {"t1_ms": [20, 2500], "t2_ms": [5, 200], "b1": [0.1, 1.2], "constraint": "T1>T2"}
TIMING_STORAGE_TOLERANCE_MS = 1e-4  # float32 prepared-observation serialization only


def _protocol_from_dataset(dataset: Any, protocol_yaml: str | Path) -> TradProtocol:
    if not hasattr(dataset, "validated_tr_vps") or not hasattr(dataset, "timing"):
        raise TypeError("Online decoder loading requires a QuantPointDataset-compatible dataset.")
    tr_ms, vps = dataset.validated_tr_vps()
    with Path(protocol_yaml).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    required = ("flip_angle_degrees", "inversion_times_ms", "t2prep_ms", "ramp_up_pulses")
    if not isinstance(cfg, dict) or any(key not in cfg for key in required):
        raise ValueError("Protocol YAML lacks fixed HHZ fields.")
    protocol = TradProtocol(float(tr_ms), int(vps), tuple(float(x) for x in cfg["flip_angle_degrees"]), tuple(float(x) for x in cfg["inversion_times_ms"]), tuple(float(x) for x in cfg["t2prep_ms"]), int(cfg["ramp_up_pulses"]))
    protocol.validate()
    return protocol


def _record(protocol: TradProtocol) -> dict[str, Any]:
    return {"tr_ms": protocol.tr_ms, "vps": protocol.vps, "fa_deg": list(protocol.fa_deg), "ti_ms": list(protocol.ti_ms), "t2prep_ms": list(protocol.t2prep_ms), "n_ramp_up": protocol.n_ramp_up}


def _validate_checkpoint(checkpoint: dict[str, Any], protocol: TradProtocol, dataset: Any, allow_functional_fixture: bool) -> None:
    required = {"state_dict", "architecture", "input_normalization", "output_normalization", "protocol_hhz_v1", "parameter_ranges", "functional_fixture", "scientific_checkpoint", "timing9_min_ms", "timing9_max_ms"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"MLP checkpoint lacks required compatibility metadata: {sorted(missing)}")
    if checkpoint["architecture"] != "12-200-200-200-10" or checkpoint["input_normalization"] != "T1/1000,T2/1000,B1,timing9/1000" or checkpoint["output_normalization"] != "raw_l2_normalized":
        raise ValueError("MLP checkpoint architecture or normalization is incompatible with v1 online reconstruction.")
    if checkpoint["parameter_ranges"] != EXPECTED_RANGES:
        raise ValueError("MLP checkpoint tissue-parameter ranges are incompatible with QuantitativeINR v1.")
    if checkpoint["protocol_hhz_v1"] != _record(protocol):
        raise ValueError("MLP checkpoint HHZ protocol/TR/VPS is incompatible with the online dataset.")
    if bool(checkpoint["functional_fixture"]) or not bool(checkpoint["scientific_checkpoint"]):
        if not allow_functional_fixture:
            raise ValueError("functional-fixture MLP checkpoint is rejected for formal online reconstruction; set the explicit smoke allowance only for a functional smoke.")
    lower, upper = np.asarray(checkpoint["timing9_min_ms"], dtype=np.float64), np.asarray(checkpoint["timing9_max_ms"], dtype=np.float64)
    if lower.shape != (9,) or upper.shape != (9,):
        raise ValueError("MLP checkpoint timing range must have exactly nine dimensions.")
    timing = dataset.timing.detach().cpu().numpy().astype(np.float64)
    bad = np.logical_or(timing < lower - TIMING_STORAGE_TOLERANCE_MS, timing > upper + TIMING_STORAGE_TOLERANCE_MS)
    if bad.any():
        details = []
        for dimension in np.flatnonzero(bad.any(axis=0)):
            values = np.unique(timing[bad[:, dimension], dimension]).tolist()
            details.append(f"timing{dimension + 2}: observed={values}, training=[{lower[dimension]}, {upper[dimension]}]")
        raise ValueError("Online timing is outside the MLP training range; no extrapolation is allowed: " + "; ".join(details))


def load_frozen_mlp_decoder(checkpoint_path: str | Path, dataset: Any, protocol_yaml: str | Path, *, allow_functional_fixture: bool, device: torch.device) -> FrozenMLPSignalDecoder:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"MLP checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("MLP checkpoint must be a mapping.")
    protocol = _protocol_from_dataset(dataset, protocol_yaml)
    _validate_checkpoint(checkpoint, protocol, dataset, allow_functional_fixture)
    model = MdmSignalMLP().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    decoder = FrozenMLPSignalDecoder(model, protocol).to(device)
    decoder.eval()
    return decoder
