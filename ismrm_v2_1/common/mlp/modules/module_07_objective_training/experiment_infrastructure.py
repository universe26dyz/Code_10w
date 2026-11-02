"""Expose the route-shared Step-1 experiment infrastructure to MLP execution."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_SOURCE = Path(__file__).resolve().parents[3] / "trad" / "modules" / "module_07_objective_training" / "experiment_infrastructure.py"
_SPEC = importlib.util.spec_from_file_location("code10w_experiment_infrastructure", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load shared experiment infrastructure: {_SOURCE}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

PROFILE_SECTIONS = _MODULE.PROFILE_SECTIONS
CachedBalancedSampler = _MODULE.CachedBalancedSampler
FixedMonitorSet = _MODULE.FixedMonitorSet
IterationProfiler = _MODULE.IterationProfiler
git_provenance = _MODULE.git_provenance
normalize_step1_config = _MODULE.normalize_step1_config
write_experiment_manifest = _MODULE.write_experiment_manifest
