"""Fail-fast server environment readiness check; it never installs dependencies."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]


def check_server_environment(mode: str) -> dict[str, object]:
    if mode not in {"formal-mlp", "reconstruction"}:
        raise ValueError("mode must be formal-mlp or reconstruction.")
    checks: dict[str, bool] = {}
    try:
        import h5py  # noqa: F401
        import torch
        checks["torch"] = True
        checks["h5py"] = True
        checks["cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:
        checks.update({"torch": False, "h5py": False, "cuda_available": False})
    if mode == "reconstruction":
        checks["vendored_nesvor"] = (CODE_ROOT / "trad" / "third_party" / "nesvor" / "nesvor").is_dir()
        checks["tinycudann_importable"] = importlib.util.find_spec("tinycudann") is not None
    ready = all(checks.values())
    result: dict[str, object] = {"mode": mode, "checks": checks, "ready": ready}
    if not ready:
        result["status"] = "NOT_READY_FOR_FORMAL_RECONSTRUCTION" if mode == "reconstruction" else "NOT_READY_FOR_FORMAL_MLP"
    else:
        result["status"] = "READY"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("formal-mlp", "reconstruction"))
    args = parser.parse_args()
    result = check_server_environment(args.mode)
    print(json.dumps(result, indent=2))
    if not result["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
