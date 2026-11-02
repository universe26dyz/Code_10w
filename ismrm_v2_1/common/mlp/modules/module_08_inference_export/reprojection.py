"""Shared native-plane reprojection for the frozen MLP online decoder."""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[3] / "trad" / "modules" / "module_08_inference_export" / "reprojection.py"
_SPEC = importlib.util.spec_from_file_location("code10w_trad_reprojection", _SOURCE)
if _SPEC is None or _SPEC.loader is None: raise ImportError(f"Cannot load shared reprojection: {_SOURCE}")
_MODULE = importlib.util.module_from_spec(_SPEC); sys.modules[_SPEC.name] = _MODULE; _SPEC.loader.exec_module(_MODULE)
export_native_plane_reprojections = _MODULE.export_native_plane_reprojections
