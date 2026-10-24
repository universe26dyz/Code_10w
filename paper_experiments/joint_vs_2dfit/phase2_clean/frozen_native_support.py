"""Read frozen native-plane L100 sampling-support records without morphology."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_frozen_l100_native_support_records(root: str | Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Resolve the immutable, already-native L100 sampling support per plane."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema not in {"q004_direct_native_endpoint_audit/v2", "phase4r_l100_native_support/v1"}:
        raise ValueError("frozen L100 native support must be q004_direct_native_endpoint_audit/v2 or phase4r_l100_native_support/v1")
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    for item in payload.get("planes", []):
        group_key = "group_idx" if schema == "q004_direct_native_endpoint_audit/v2" else "original_group_idx"
        key = (str(item["parameter"]).lower(), str(item["stack"]), int(item[group_key]))
        if key in records:
            raise ValueError(f"duplicate frozen L100 support identity: {key}")
        relative = item["output"] if schema == "q004_direct_native_endpoint_audit/v2" else item["path"]
        path = Path(relative)
        records[key] = {
            "path": str((path if path.is_absolute() else root / path).resolve()),
            "field": "l100_native" if schema == "q004_direct_native_endpoint_audit/v2" else "phase4r_l100",
            "declared_sha256": item.get("sha256"),
            "source_schema": schema,
        }
    return records
