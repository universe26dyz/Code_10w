"""Strict subject/stack deployment manifest contract shared by local and server tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


VALID_STACKS = ("sax", "2ch", "4ch")


def load_deployment_manifest(path: str | Path) -> list[dict[str, str]]:
    """Return explicit entries without discovering subjects, stacks, or paths."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Deployment manifest does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        payload: Any = json.load(handle)
    if not isinstance(payload, dict) or payload.get("schema_version") != "deployment-v1":
        raise ValueError("Deployment manifest requires schema_version='deployment-v1'.")
    subjects = payload.get("subjects")
    if not isinstance(subjects, list) or not subjects:
        raise ValueError("Deployment manifest requires a non-empty subjects list.")
    entries: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for subject in subjects:
        if not isinstance(subject, dict):
            raise ValueError("Each deployment subject must be an object.")
        subject_id = subject.get("subject_id")
        stacks = subject.get("stacks")
        if not isinstance(subject_id, str) or not subject_id.strip():
            raise ValueError("Each deployment subject requires non-empty subject_id.")
        if not isinstance(stacks, list) or not stacks:
            raise ValueError(f"Deployment subject {subject_id!r} requires a non-empty stacks list.")
        for stack_entry in stacks:
            if not isinstance(stack_entry, dict):
                raise ValueError(f"Deployment subject {subject_id!r} has a non-object stack entry.")
            stack, dicom_dir = stack_entry.get("stack"), stack_entry.get("dicom_dir")
            if stack not in VALID_STACKS:
                raise ValueError(f"Deployment stack must be one of {VALID_STACKS}, got {stack!r}.")
            if not isinstance(dicom_dir, str) or not dicom_dir.strip():
                raise ValueError(f"Deployment {subject_id}/{stack} requires non-empty dicom_dir.")
            key = (subject_id, stack)
            if key in seen:
                raise ValueError(f"Deployment manifest has duplicate subject_id/stack: {subject_id}/{stack}.")
            seen.add(key)
            entries.append({"subject_id": subject_id, "stack": stack, "dicom_dir": dicom_dir})
    return entries
