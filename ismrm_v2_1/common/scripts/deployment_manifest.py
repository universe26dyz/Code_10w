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
    if not isinstance(payload, dict):
        raise ValueError("Deployment manifest must be a JSON object.")
    if payload.get("schema_version") == "deployment-v1":
        raise ValueError("deployment-v1 has one ambiguous dicom_dir; migrate to deployment-v1.1 with local_dicom_dir and server_dicom_dir.")
    if payload.get("schema_version") != "deployment-v1.1":
        raise ValueError("Deployment manifest requires schema_version='deployment-v1.1'.")
    subjects = payload.get("subjects")
    if not isinstance(subjects, list) or not subjects:
        raise ValueError("Deployment manifest requires a non-empty subjects list.")
    entries: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    subject_ids: set[str] = set()
    for subject in subjects:
        if not isinstance(subject, dict):
            raise ValueError("Each deployment subject must be an object.")
        subject_id = subject.get("subject_id")
        stacks = subject.get("stacks")
        if not isinstance(subject_id, str) or not subject_id.strip():
            raise ValueError("Each deployment subject requires non-empty subject_id.")
        if subject_id in subject_ids:
            raise ValueError(f"Deployment manifest has duplicate subject_id: {subject_id}.")
        subject_ids.add(subject_id)
        if not isinstance(stacks, list) or not stacks:
            raise ValueError(f"Deployment subject {subject_id!r} requires a non-empty stacks list.")
        for stack_entry in stacks:
            if not isinstance(stack_entry, dict):
                raise ValueError(f"Deployment subject {subject_id!r} has a non-object stack entry.")
            stack = stack_entry.get("stack")
            local_dicom_dir, server_dicom_dir = stack_entry.get("local_dicom_dir"), stack_entry.get("server_dicom_dir")
            if stack not in VALID_STACKS:
                raise ValueError(f"Deployment stack must be one of {VALID_STACKS}, got {stack!r}.")
            if not isinstance(local_dicom_dir, str) or not local_dicom_dir.strip():
                raise ValueError(f"Deployment {subject_id}/{stack} requires non-empty local_dicom_dir.")
            if not isinstance(server_dicom_dir, str) or not server_dicom_dir.strip():
                raise ValueError(f"Deployment {subject_id}/{stack} requires non-empty server_dicom_dir.")
            key = (subject_id, stack)
            if key in seen:
                raise ValueError(f"Deployment manifest has duplicate subject_id/stack: {subject_id}/{stack}.")
            seen.add(key)
            entries.append({"subject_id": subject_id, "stack": stack, "local_dicom_dir": local_dicom_dir, "server_dicom_dir": server_dicom_dir})
    return entries
