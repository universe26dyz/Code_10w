"""Use the Recovery-verified HB1 registration adapter without changing its contract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[3] / "trad" / "modules" / "module_06_rigid_psf" / "hb1_stack_adapter.py"
_SPEC = importlib.util.spec_from_file_location("code10w_hb1_stack_adapter", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load HB1 stack adapter: {_SOURCE}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

StackInitialization = _MODULE.StackInitialization
initialize_group_poses_from_hb1 = _MODULE.initialize_group_poses_from_hb1
