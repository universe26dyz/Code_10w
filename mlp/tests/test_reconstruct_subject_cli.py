import subprocess
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[2]


def _help(method: str) -> str:
    result = subprocess.run([sys.executable, "-m", "scripts.reconstruct_subject", "--help"], cwd=CODE_ROOT / method, check=True, capture_output=True, text=True)
    return result.stdout


def test_trad_and_mlp_reconstruct_wrappers_use_only_output_flag():
    for method in ("trad", "mlp"):
        help_text = _help(method)
        assert "--output OUTPUT" in help_text
        assert "--output-dir" not in help_text
