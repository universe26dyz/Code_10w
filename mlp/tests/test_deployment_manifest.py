import json
import sys
from pathlib import Path

import pytest


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.deployment_manifest import load_deployment_manifest


def _write_manifest(path: Path, entries: list[dict]) -> Path:
    path.write_text(json.dumps({"schema_version": "deployment-v1.1", "subjects": entries}), encoding="utf-8")
    return path


def test_manifest_keeps_explicit_local_and_server_paths(tmp_path):
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [{"subject_id": "S01", "stacks": [{"stack": "sax", "local_dicom_dir": "/local/S01/sax", "server_dicom_dir": "/server/S01/sax"}]}],
    )
    entries = load_deployment_manifest(manifest)
    assert entries == [{"subject_id": "S01", "stack": "sax", "local_dicom_dir": "/local/S01/sax", "server_dicom_dir": "/server/S01/sax"}]


def test_manifest_rejects_old_ambiguous_dicom_dir_with_migration_message(tmp_path):
    manifest = tmp_path / "old.json"
    manifest.write_text(json.dumps({"schema_version": "deployment-v1", "subjects": [
        {"subject_id": "S01", "stacks": [{"stack": "sax", "dicom_dir": "/ambiguous/S01/sax"}]}
    ]}), encoding="utf-8")
    with pytest.raises(ValueError, match="migrate"):
        load_deployment_manifest(manifest)


def test_manifest_rejects_duplicate_subject_stack_without_searching(tmp_path):
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [{"subject_id": "S01", "stacks": [
            {"stack": "sax", "local_dicom_dir": "/local/S01/sax", "server_dicom_dir": "/server/S01/sax"},
            {"stack": "sax", "local_dicom_dir": "/local/S01/sax2", "server_dicom_dir": "/server/S01/sax2"},
        ]}],
    )
    with pytest.raises(ValueError, match="duplicate subject_id/stack"):
        load_deployment_manifest(manifest)
