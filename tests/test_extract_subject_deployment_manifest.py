from __future__ import annotations

import json


def test_extracts_only_requested_subject(tmp_path) -> None:
    from scripts.extract_subject_deployment_manifest import extract_subject_manifest

    source = tmp_path / "many.json"
    source.write_text(json.dumps({"schema_version": "deployment-v1.1", "subjects": [
        {"subject_id": "A", "stacks": []}, {"subject_id": "B", "stacks": []},
    ]}), encoding="utf-8")
    output = extract_subject_manifest(source, "B", tmp_path / "one.json")
    assert json.loads(output.read_text(encoding="utf-8"))["subjects"] == [{"subject_id": "B", "stacks": []}]
