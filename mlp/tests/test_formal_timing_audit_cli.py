import subprocess
import sys
from pathlib import Path


MLP_ROOT = Path(__file__).resolve().parents[1]
AUDIT_MODULE = "modules.module_05_signal_decoder.formal_timing_audit"
AUDIT_PATH = MLP_ROOT / "modules" / "module_05_signal_decoder" / "formal_timing_audit.py"


def test_formal_timing_audit_compiles_and_cli_help_runs():
    subprocess.run([sys.executable, "-m", "py_compile", str(AUDIT_PATH)], check=True)
    result = subprocess.run([sys.executable, "-m", AUDIT_MODULE, "--help"], cwd=MLP_ROOT, check=True, capture_output=True, text=True)
    assert "--pool" in result.stdout
