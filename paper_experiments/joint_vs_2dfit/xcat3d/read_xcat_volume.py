"""Read externally generated XCAT volumes only when full provenance is verified."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import Geometry, XCATInputs, XCATVolume


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PARAMETER_FOR_VOLUME = {"t1_ms": "t1_phase1_2.par", "t2_ms": "t2_phase1_2.par", "pd_m0": "pd_phase1_2.par", "myocardium_mask": "lv_mask_phase1_2.par"}


def _invalid(value: object) -> bool:
    return not isinstance(value, str) or not value.strip() or any(token in value.upper() for token in ("REQUIRED", "TODO", "UNKNOWN", "SET_"))


def _require_text(value: object, label: str) -> str:
    if _invalid(value):
        raise ValueError(f"{label} must not be a placeholder")
    return str(value)


def _require_sha(value: object, label: str) -> str:
    value = _require_text(value, label)
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a complete SHA256")
    return value


def _reject_placeholders(value: object, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_placeholders(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_placeholders(child, f"{label}[{index}]")
    elif isinstance(value, str) and _invalid(value):
        raise ValueError(f"{label} contains a placeholder")


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


def _verified_file(record: dict[str, Any], root: Path, label: str) -> tuple[Path, str]:
    path = root / _require_text(record.get("path"), f"{label} path")
    declared = _require_sha(record.get("sha256"), f"{label} SHA256")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != declared:
        raise ValueError(f"{label} SHA256 mismatch")
    return path, declared


def _volume(name: str, record: dict[str, Any], root: Path) -> XCATVolume:
    geometry = Geometry.from_record(record)
    path, _ = _verified_file(record, root, f"XCAT raw-volume {name}")
    return XCATVolume(name, _load_array(path, str(record["dtype"]), geometry.shape), geometry, path)


def _same_geometry(reference: XCATVolume, candidate: XCATVolume) -> None:
    if reference.geometry != candidate.geometry:
        raise ValueError(f"geometry mismatch: {candidate.name} does not match {reference.name} (shape, spacing_mm, origin_mm, axis_order, and world_convention must agree)")


def _validate_provenance(manifest: dict[str, Any], root: Path) -> None:
    xcat = manifest.get("xcat", {})
    for key in ("version_build", "source", "cardiac_phase", "respiratory_phase"):
        _require_text(xcat.get(key), f"XCAT provenance {key}")
    parameter_files = xcat.get("parameter_files")
    if not isinstance(parameter_files, dict) or set(parameter_files) != set(_PARAMETER_FOR_VOLUME.values()):
        raise ValueError("XCAT parameter_files must contain T1, T2, PD and LV-mask files exactly")
    parameter_hashes = {}
    for name, record in parameter_files.items():
        parameter_path, parameter_hashes[name] = _verified_file(record, root, f"XCAT parameter-file {name}")
        _reject_placeholders(parameter_path.read_text(), f"XCAT parameter-file {name}")
    specs = {}
    for key in ("phantom_spec", "geometry_spec"):
        record = xcat.get(key)
        if not isinstance(record, dict):
            raise ValueError(f"XCAT provenance is missing {key}")
        spec_path, specs[key] = _verified_file(record, root, f"XCAT {key}")
        try:
            _reject_placeholders(json.loads(spec_path.read_text()), f"XCAT {key}")
        except json.JSONDecodeError as error:
            raise ValueError(f"XCAT {key} must be valid JSON") from error
    records = manifest.get("volumes", {})
    if set(records) != set(_PARAMETER_FOR_VOLUME):
        raise ValueError("input manifest requires exactly T1, T2, PD and LV-mask raw volumes")
    for name, parameter in _PARAMETER_FOR_VOLUME.items():
        record = records[name]
        if _require_sha(record.get("parameter_file_sha256"), f"{name} parameter-file SHA256") != parameter_hashes[parameter]:
            raise ValueError(f"{name} parameter-file SHA256 mismatch")
        if _require_sha(record.get("shared_geometry_spec_sha256"), f"{name} geometry-spec SHA256") != specs["geometry_spec"]:
            raise ValueError(f"{name} shared geometry-spec SHA256 mismatch")
        if _require_text(record.get("cardiac_phase"), f"{name} cardiac phase") != xcat["cardiac_phase"]:
            raise ValueError(f"{name} cardiac phase does not match shared XCAT phase")
        if _require_text(record.get("respiratory_phase"), f"{name} respiratory phase") != xcat["respiratory_phase"]:
            raise ValueError(f"{name} respiratory phase does not match shared XCAT phase")


def load_xcat_inputs(manifest_path: str | Path) -> XCATInputs:
    """Load the formal XCAT3D input contract; no implicit raw-binary geometry exists."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    if manifest.get("phase") != "1.2" or manifest.get("source_family") != "single_source_xcat_3d_quantitative":
        raise ValueError("input manifest is not a Phase-1.2 single-source XCAT3D manifest")
    if manifest.get("support_policy") != "native_xcat_pd_positive" or manifest.get("m0_policy") != "controlled_binary_signal_support_m0_1":
        raise ValueError("formal Phase-1.2 requires native_xcat_pd_positive support and controlled binary M0")
    root = path.parent
    _validate_provenance(manifest, root)
    records = manifest["volumes"]
    t1, t2, pd, myocardium = (_volume(key, records[key], root) for key in ("t1_ms", "t2_ms", "pd_m0", "myocardium_mask"))
    for volume in (t2, pd, myocardium):
        _same_geometry(t1, volume)
    return XCATInputs(path, manifest, t1, t2, pd, myocardium, pd)
