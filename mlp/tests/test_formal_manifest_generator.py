import importlib.util
import json
import sys
from pathlib import Path

import pytest


CODE_ROOT = Path(__file__).resolve().parents[2]
MLP_ROOT = CODE_ROOT / "mlp"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def _load_script():
    spec = importlib.util.spec_from_file_location("formal_manifest_generator", MLP_ROOT / "scripts" / "build_formal_manifest_from_prepared.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_formal_manifest_uses_exact_prepared_paths(tmp_path):
    module = _load_script()
    deployment = tmp_path / "deployment.json"
    deployment.write_text(json.dumps({"schema_version": "deployment-v1", "subjects": [
        {"subject_id": "S01", "stacks": [{"stack": "sax", "dicom_dir": "/raw/S01/sax"}]}
    ]}), encoding="utf-8")
    prepared = tmp_path / "prepared"; expected = prepared / "S01" / "sax" / "observations.npz"
    expected.parent.mkdir(parents=True); expected.write_bytes(b"fixture")
    output = tmp_path / "formal.json"
    result = module.build_formal_manifest(deployment, prepared, output)
    assert result["source_count"] == 1
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["sources"] == [{"subject_id": "S01", "stack": "sax", "observations": str(expected)}]
    with pytest.raises(FileExistsError):
        module.build_formal_manifest(deployment, prepared, output)
