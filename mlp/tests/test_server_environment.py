import importlib.util
import json
from pathlib import Path

import pytest


CODE_ROOT = Path(__file__).resolve().parents[2]


def _load_preflight():
    spec = importlib.util.spec_from_file_location("server_environment", CODE_ROOT / "scripts" / "check_server_environment.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_formal_preflight_reports_all_required_imports_and_cuda_fields():
    module = _load_preflight()
    report = module.check_server_environment("formal-mlp")
    assert set(("torch", "numpy", "scipy", "h5py", "pydicom", "nibabel", "yaml")).issubset(report["checks"])
    assert {"torch_version", "cuda_available", "cuda_device_count", "cuda_device_0_name"}.issubset(report["environment"])


def test_preflight_output_refuses_overwrite(tmp_path):
    module = _load_preflight()
    report = {"mode": "formal-mlp", "ready": False}
    output = tmp_path / "reports" / "preflight.json"
    module.write_report(output, report)
    assert json.loads(output.read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError):
        module.write_report(output, report)
