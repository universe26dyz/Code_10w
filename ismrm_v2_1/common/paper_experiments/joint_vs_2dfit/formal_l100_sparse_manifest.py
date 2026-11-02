"""Shared admission for formal L100 sparse manifests used by native metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .sparse_profile_manifest import SCHEMA_VERSION, validate_sparse_profile_manifest

LEGACY_SCHEMA_VERSION = "phase4r_sparse_v1"


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("formal L100 sparse manifest must be a JSON object")
    return value


def _identity_v2(payload: dict[str, Any]) -> dict[str, Any]:
    subject = payload.get("subject")
    if not isinstance(subject, str) or not subject:
        raise ValueError("formal L100 sparse-profile manifest requires a non-empty subject")
    if payload.get("profile_id") != "L100":
        raise ValueError("formal L100 sparse-profile manifest requires profile_id=L100")
    if payload.get("case_id") != "L100":
        raise ValueError("formal L100 sparse-profile manifest requires case_id=L100")
    if payload.get("fold_count") != 1 or payload.get("fold_id") is not None:
        raise ValueError("formal L100 sparse-profile manifest requires fold_count=1 and fold_id=null")
    stacks = payload.get("stacks")
    if not isinstance(stacks, dict) or set(stacks) != {"sax", "2ch", "4ch"}:
        raise ValueError("formal L100 sparse-profile manifest requires sax/2ch/4ch stacks")
    for stack in ("sax", "2ch", "4ch"):
        item = stacks[stack]
        if not isinstance(item, dict) or item.get("mode") != "all":
            raise ValueError(f"formal L100 sparse-profile manifest requires {stack}.mode=all")
        ordered = item.get("ordered_source_group_ids")
        retained = item.get("retained_original_group_ids")
        heldout = item.get("heldout_original_group_ids")
        if not isinstance(ordered, list) or not isinstance(retained, list) or ordered != retained:
            raise ValueError(f"formal L100 sparse-profile manifest requires {stack} full retained source lattice")
        if heldout != []:
            raise ValueError(f"formal L100 sparse-profile manifest requires {stack}.heldout_original_group_ids=[]")
    return {"subject": subject, "sparse_manifest_schema": SCHEMA_VERSION, "sparse_identity": {"profile_id": "L100", "case_id": "L100", "fold_id": None, "fold_count": 1}}


def validate_formal_l100_sparse_manifest(path: str | Path) -> dict[str, Any]:
    """Admit only historical v1 L100 or validated full-lattice profile-v2 L100."""
    path = Path(path)
    raw = _read(path)
    schema = raw.get("schema_version")
    if schema == SCHEMA_VERSION:
        return _identity_v2(validate_sparse_profile_manifest(path))
    if schema == LEGACY_SCHEMA_VERSION:
        subject = raw.get("subject")
        if not isinstance(subject, str) or not subject or raw.get("condition") != "L100":
            raise ValueError("formal legacy L100 sparse manifest requires named subject and condition=L100")
        return {"subject": subject, "sparse_manifest_schema": LEGACY_SCHEMA_VERSION, "sparse_identity": {"condition": "L100"}}
    raise ValueError(f"unsupported sparse manifest schema for formal L100: {schema!r}")
