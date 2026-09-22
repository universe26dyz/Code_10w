"""Package-qualified Trad imports must coexist with the MLP decoder package."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_trad_runtime_closure_and_mlp_decoder_do_not_create_legacy_namespaces() -> None:
    """Exercise imports in a clean interpreter, independent of pytest's legacy path shim."""

    repository_root = Path(__file__).resolve().parents[2]
    program = """\
import importlib
import sys

importlib.import_module('trad.scripts.run_training')
importlib.import_module('mlp.modules.module_05_signal_decoder.frozen_decoder')
assert 'modules' not in sys.modules
assert 'third_party' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
