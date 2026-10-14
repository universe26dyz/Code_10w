"""Read externally generated XCAT property volumes only when their geometry is explicit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import Geometry, XCATInputs, XCATVolume


def _load_array(path: Path, dtype: str, shape: tuple[int, int, int]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"required XCAT volume is missing: {path}")
    expected = np.dtype(dtype)
    if path.suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    elif path.suffix == ".bin":
        array = np.fromfile(path, dtype=expected).reshape(shape)
    else:
        raise ValueError(f"unsupported XCAT volume format {path.suffix!r}; use .npy or raw .bin")
    if array.shape != shape:
        raise ValueError(f"XCAT volume shape mismatch for {path}: declared {shape}, found {array.shape}")
    if array.dtype != expected:
        raise ValueError(f"XCAT volume dtype mismatch for {path}: declared {expected}, found {array.dtype}")
    return array


def _volume(name: str, record: dict[str, Any], root: Path) -> XCATVolume:
    geometry = Geometry.from_record(record)
    path = root / str(record["path"])
    if not record.get("sha256") or str(record["sha256"]).upper() in {"UNKNOWN", "TODO", "REQUIRED"}:
        raise ValueError(f"XCAT raw-volume SHA256 is required for {name}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record["sha256"]:
        raise ValueError(f"XCAT raw-volume SHA256 mismatch for {name}")
    return XCATVolume(name, _load_array(path, str(record["dtype"]), geometry.shape), geometry, path)


def _same_geometry(reference: XCATVolume, candidate: XCATVolume) -> None:
    if reference.geometry != candidate.geometry:
        raise ValueError(
            f"geometry mismatch: {candidate.name} does not match {reference.name} "
            f"(shape, spacing_mm, origin_mm, axis_order, and world_convention must agree)"
        )


def load_xcat_inputs(manifest_path: str | Path) -> XCATInputs:
    """Load the formal XCAT3D input contract; no implicit raw-binary geometry exists."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    if manifest.get("phase") != "1.2" or manifest.get("source_family") != "single_source_xcat_3d_quantitative":
        raise ValueError("input manifest is not a Phase-1.2 single-source XCAT3D manifest")
    xcat = manifest.get("xcat", {})
    for key in ("version_build", "source", "parameter_files", "configuration_hashes", "cardiac_phase", "respiratory_phase"):
        if key not in xcat:
            raise ValueError(f"XCAT provenance is missing {key}")
    records = manifest.get("volumes", {})
    for key in ("t1_ms", "t2_ms", "myocardium_mask"):
        if key not in records:
            raise ValueError(f"input manifest is missing required volume {key}")
    if "signal_support_mask" not in records and "pd_m0" not in records:
        raise ValueError("input manifest requires signal_support_mask or pd_m0")
    root = path.parent
    t1, t2, myocardium = (_volume(key, records[key], root) for key in ("t1_ms", "t2_ms", "myocardium_mask"))
    support = _volume("signal_support_mask", records["signal_support_mask"], root) if "signal_support_mask" in records else _volume("pd_m0", records["pd_m0"], root)
    pd = _volume("pd_m0", records["pd_m0"], root) if "pd_m0" in records else None
    for volume in (t2, myocardium, support) + ((pd,) if pd is not None else ()):
        _same_geometry(t1, volume)
    return XCATInputs(path, manifest, t1, t2, support, myocardium, pd)
