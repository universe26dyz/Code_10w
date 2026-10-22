"""Immutable physical-geometry standardization to SAX/2CH/4CH = 12/15/10."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import sha256, write_immutable_json
from .sparse import STACKS, _git_sha, _physical_group_order, _read, _subset


TARGET_COUNTS = {"sax": 12, "2ch": 15, "4ch": 10}
SELECTION_POLICY_VERSION = "physical_contiguous_symmetric_v1"


def _group_fingerprint(data: dict[str, np.ndarray], group: int) -> str:
    """Hash group content independent of its later standardized group index."""
    rows = np.flatnonzero(np.asarray(data["group_idx"], int) == int(group))
    digest = hashlib.sha256()
    for key in sorted(data):
        if key == "group_idx":
            continue
        value = np.asarray(data[key])
        selected = value[rows] if value.shape[:1] == np.asarray(data["group_idx"]).shape else value
        digest.update(key.encode("utf-8")); digest.update(np.ascontiguousarray(selected).tobytes())
    return digest.hexdigest()


def _select_contiguous(ordered: list[int], positions: dict[int, float], target: int, path: Path) -> tuple[list[int], list[int], dict[str, Any]]:
    if len(ordered) < target:
        raise ValueError(f"{path}: source has {len(ordered)} groups, fewer than target {target}")
    differences = np.diff([positions[group] for group in ordered])
    regular = bool(len(differences) <= 1 or np.allclose(differences, differences[0], rtol=1e-5, atol=1e-5))
    if not regular:
        raise ValueError(f"{path}: geometry audit requires regular physical gaps before trimming")
    excess = len(ordered) - target
    lower = excess // 2
    upper = excess - lower
    retained = ordered[lower:len(ordered) - upper if upper else len(ordered)]
    dropped = ordered[:lower] + ordered[len(ordered) - upper:] if upper else ordered[:lower]
    if len(retained) != target or retained != ordered[lower:lower + target]:
        raise ValueError(f"{path}: selection is not one contiguous physical interval")
    return retained, dropped, {
        "regular_gap": regular,
        "gap_mm": None if not len(differences) else float(differences[0]),
        "contiguous_physical_interval": True,
        "excess_count": excess,
        "odd_trim_tie_break": "drop_highest_signed_coordinate_outer_group" if excess % 2 else "symmetric_equal_end_trimming",
    }


def validate_standardization_manifest(manifest: str | Path, *, subject: str | None = None) -> dict[str, Any]:
    """Validate the evidence required before a standardization can be trusted."""
    path = Path(manifest); payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "phase4r_common_slice_standardization/v1" or not isinstance(payload.get("subject"), str):
        raise ValueError("invalid standardization manifest schema or subject")
    if subject is not None and payload["subject"] != subject:
        raise ValueError("standardization manifest subject mismatch")
    if not isinstance(payload.get("git_sha"), str) or not payload["git_sha"]:
        raise ValueError("standardization manifest lacks code SHA")
    for stack, target in TARGET_COUNTS.items():
        item = payload.get("stacks", {}).get(stack)
        if not isinstance(item, dict) or int(item.get("output_count", -1)) != target:
            raise ValueError(f"{stack}: target count is invalid")
        for source_name in ("source_observation", "output_observation"):
            evidence = item.get(source_name, {}); observation = Path(str(evidence.get("path", "")))
            if not observation.is_file() or sha256(observation) != evidence.get("sha256"):
                raise ValueError(f"{stack}: {source_name} SHA256 mismatch")
        mapping = {int(key): int(value) for key, value in item.get("derived_to_original_group_idx", {}).items()}
        retained = [int(value) for value in item.get("retained_original_group_ids", [])]
        if sorted(mapping) != list(range(target)) or list(mapping.values()) != retained or len(set(retained)) != target:
            raise ValueError(f"{stack}: original-to-standardized mapping is invalid")
        audit = item.get("geometry_audit", {})
        if audit.get("regular_gap") is not True or audit.get("contiguous_physical_interval") is not True:
            raise ValueError(f"{stack}: geometry audit is invalid")
        with np.load(Path(item["source_observation"]["path"]), allow_pickle=False) as source, np.load(Path(item["output_observation"]["path"]), allow_pickle=False) as data:
            if int(item.get("source_count", -1)) != len(np.unique(source["group_idx"])):
                raise ValueError(f"{stack}: source group count is invalid")
            if np.unique(data["group_idx"]).tolist() != list(range(target)):
                raise ValueError(f"{stack}: output group indices are not contiguous")
    return payload


def standardized_subject_readiness(subject: str, standardization_root: str | Path) -> dict[str, str]:
    """Return a conservative state; a manifest alone is never completion proof."""
    manifest = Path(standardization_root) / subject / "slice_standardization_manifest.json"
    if not manifest.is_file():
        return {"subject": subject, "status": "READY_FOR_STANDARDIZATION" if subject == "DYL_2" else "RAW_AUDIT_PENDING"}
    try:
        validate_standardization_manifest(manifest, subject=subject)
    except (OSError, ValueError, KeyError, TypeError):
        return {"subject": subject, "status": "STANDARDIZATION_AUDIT_REQUIRED"}
    return {"subject": subject, "status": "STANDARDIZATION_COMPLETE"}


def standardize_subject_slices(prepared_root: str | Path | None, subject: str, output_root: str | Path, *, prepared_subject_root: str | Path | None = None, targets: dict[str, int] | None = None, dry_run: bool = False, git_sha: str | None = None) -> Path | dict[str, Any]:
    """Subset only complete physical groups and write a new immutable prepared root."""
    targets = dict(TARGET_COUNTS if targets is None else targets)
    if targets != TARGET_COUNTS:
        raise ValueError("standardization targets must be exactly sax=12, 2ch=15, 4ch=10")
    if (prepared_root is None) == (prepared_subject_root is None):
        raise ValueError("provide exactly one of prepared_root or prepared_subject_root")
    source_root = Path(prepared_subject_root).resolve() if prepared_subject_root is not None else Path(prepared_root).resolve() / subject
    destination = Path(output_root).resolve() / subject
    if destination.exists() and not dry_run:
        raise FileExistsError(f"refusing overwrite: {destination}")
    source_paths = {stack: source_root / stack / "observations.npz" for stack in STACKS}
    source_data = {stack: _read(path) for stack, path in source_paths.items()}
    plans: dict[str, dict[str, Any]] = {}
    for stack in STACKS:
        ordered, positions, normal = _physical_group_order(source_data[stack], source_paths[stack])
        retained, dropped, audit = _select_contiguous(ordered, positions, targets[stack], source_paths[stack])
        plans[stack] = {"ordered": ordered, "positions": positions, "normal": normal, "retained": retained, "dropped": dropped, "audit": audit}
    if dry_run:
        return {"subject": subject, "status": "DRY_RUN", "dry_run": True, "output": str(destination), "targets": targets, "plans": {stack: {"retained_original_group_ids": plan["retained"], "dropped_original_group_ids": plan["dropped"]} for stack, plan in plans.items()}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{subject}.standardization-", dir=destination.parent))
    try:
        records: dict[str, Any] = {}
        for stack in STACKS:
            plan = plans[stack]; derived, mapping = _subset(source_data[stack], plan["retained"])
            target = staging / stack / "observations.npz"; target.parent.mkdir(parents=True, exist_ok=False)
            np.savez_compressed(target, **derived)
            records[stack] = {
                "source_observation": {"path": str(source_paths[stack]), "sha256": sha256(source_paths[stack])},
                "output_observation": {"path": str((destination / stack / "observations.npz")), "sha256": sha256(target)},
                "source_count": len(plan["ordered"]), "output_count": len(plan["retained"]),
                "ordered_original_group_ids": plan["ordered"], "ordered_physical_positions_mm": [plan["positions"][group] for group in plan["ordered"]], "stack_normal_lps": plan["normal"],
                "retained_original_group_ids": plan["retained"], "dropped_original_group_ids": plan["dropped"], "derived_to_original_group_idx": mapping,
                "source_group_content_sha256": {str(group): _group_fingerprint(source_data[stack], group) for group in plan["retained"]}, "geometry_audit": plan["audit"],
                "odd_trim_tie_break": plan["audit"]["odd_trim_tie_break"],
            }
        manifest = {"schema": "phase4r_common_slice_standardization/v1", "subject": subject, "resolved_source_root": str(source_root), "targets": targets, "selection_policy_version": SELECTION_POLICY_VERSION, "input_not_modified": True, "stacks": records, "git_sha": git_sha or _git_sha(), "status": "STANDARDIZATION_COMPLETE"}
        write_immutable_json(staging / "slice_standardization_manifest.json", manifest)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    roots = parser.add_mutually_exclusive_group(required=True)
    roots.add_argument("--prepared-root"); roots.add_argument("--prepared-subject-root")
    parser.add_argument("--subject", required=True); parser.add_argument("--output-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(standardize_subject_slices(args.prepared_root, args.subject, args.output_root, prepared_subject_root=args.prepared_subject_root, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
