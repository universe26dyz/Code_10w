"""Fail-fast server environment readiness check; it never installs dependencies."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]


REQUIRED_IMPORTS = {
    "torch": "torch",
    "numpy": "numpy",
    "scipy": "scipy",
    "h5py": "h5py",
    "pydicom": "pydicom",
    "nibabel": "nibabel",
    "yaml": "yaml",
}


def _import_check(module_name: str) -> tuple[bool, object | None]:
    try:
        return True, importlib.import_module(module_name)
    except ImportError:
        return False, None


def _project_import_check(module_name: str) -> bool:
    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(CODE_ROOT))
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False
    finally:
        sys.path[:] = original_path


def check_server_environment(mode: str) -> dict[str, object]:
    if mode not in {"formal-mlp", "reconstruction"}:
        raise ValueError("mode must be formal-mlp or reconstruction.")
    checks: dict[str, bool] = {}
    imported: dict[str, object | None] = {}
    for label, module_name in REQUIRED_IMPORTS.items():
        checks[label], imported[label] = _import_check(module_name)
    torch_module = imported["torch"]
    cuda_available = bool(torch_module is not None and torch_module.cuda.is_available())
    cuda_device_count = int(torch_module.cuda.device_count()) if cuda_available else 0
    device_name = torch_module.cuda.get_device_name(0) if cuda_device_count else None
    checks["cuda_available"] = cuda_available
    environment: dict[str, object] = {
        "torch_version": getattr(torch_module, "__version__", None),
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_device_0_name": device_name,
    }
    if mode == "reconstruction":
        checks["tinycudann"] = _import_check("tinycudann")[0]
        checks["vendored_nesvor_import"] = _project_import_check("trad.third_party.nesvor.nesvor")
        checks["project_quantitative_inr_import"] = _project_import_check("trad.modules.module_04_quantitative_inr.quantitative_inr")
        checks["project_rigid_psf_import"] = _project_import_check("trad.modules.module_06_rigid_psf.rigid_psf_forward")
    ready = all(checks.values())
    result: dict[str, object] = {"mode": mode, "checks": checks, "environment": environment, "ready": ready}
    if not ready:
        result["status"] = "NOT_READY_FOR_FORMAL_RECONSTRUCTION" if mode == "reconstruction" else "NOT_READY_FOR_FORMAL_MLP"
    else:
        result["status"] = "READY"
    return result


def write_report(output_path: str | Path, report: dict[str, object]) -> None:
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Preflight output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("formal-mlp", "reconstruction"))
    parser.add_argument("--output")
    args = parser.parse_args()
    result = check_server_environment(args.mode)
    if args.output:
        write_report(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
