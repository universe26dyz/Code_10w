import json
import sys
from pathlib import Path

import pytest


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.deployment_manifest import load_deployment_manifest


def _write_manifest(path: Path, entries: list[dict]) -> Path:
    path.write_text(json.dumps({"schema_version": "deployment-v1", "subjects": entries}), encoding="utf-8")
    return path


def test_manifest_keeps_only_explicit_subject_stack_entries(tmp_path):
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [{"subject_id": "S01", "stacks": [{"stack": "sax", "dicom_dir": "/raw/S01/sax"}]}],
    )
    entries = load_deployment_manifest(manifest)
    assert entries == [{"subject_id": "S01", "stack": "sax", "dicom_dir": "/raw/S01/sax"}]


def test_manifest_rejects_duplicate_subject_stack_without_searching(tmp_path):
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [{"subject_id": "S01", "stacks": [
            {"stack": "sax", "dicom_dir": "/raw/S01/sax"},
            {"stack": "sax", "dicom_dir": "/raw/S01/sax_duplicate"},
        ]}],
    )
    with pytest.raises(ValueError, match="duplicate subject_id/stack"):
        load_deployment_manifest(manifest)
