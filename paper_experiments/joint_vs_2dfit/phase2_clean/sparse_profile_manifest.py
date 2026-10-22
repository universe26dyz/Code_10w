"""Read-only validation for Phase-4R sparse-profile manifests.

The sparse generator is the sole writer of these manifests.  This module is a
consumer-side admission gate: it never creates observations or changes the
manifest, and rejects any incomplete evidence instead of inferring it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import sha256
from .sparse import _json_sha256, _resolve_profile, load_sparse_profile


STACKS = ("sax", "2ch", "4ch")
SCHEMA_VERSION = "phase4r_sparse_profile_v2"


def _fail(message: str) -> None:
    raise ValueError(f"invalid sparse-profile manifest: {message}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"sparse-profile manifest does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid sparse-profile manifest JSON: {path}") from exc
    if not isinstance(value, dict):
        _fail("top-level document must be an object")
    return value


def _ids(value: object, *, field: str, stack: str) -> list[int]:
    if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        _fail(f"{stack}.{field} must be a list of integer group IDs")
    result = [int(item) for item in value]
    if len(result) != len(set(result)):
        _fail(f"{stack}.{field} contains duplicate group IDs")
    return result


def _required_npz(data: np.lib.npyio.NpzFile, path: Path) -> None:
    required = {
        "images", "masks", "group_idx", "weight_idx", "affine_lps_rc",
        "pixel_spacing_rc_mm", "slice_thickness_mm", "timing9_ms", "tr_ms", "vps",
    }
    missing = required.difference(data.files)
    if missing:
        _fail(f"{path}: observation NPZ is missing {sorted(missing)}")
    rows = len(data["group_idx"])
    if any(len(data[name]) != rows for name in required):
        _fail(f"{path}: observation NPZ arrays have inconsistent row counts")


def _groups_and_weights(data: np.lib.npyio.NpzFile, path: Path) -> list[int]:
    _required_npz(data, path)
    groups = [int(value) for value in np.unique(data["group_idx"])]
    for group in groups:
        rows = np.flatnonzero(data["group_idx"] == group)
        if rows.size != 10 or not np.array_equal(np.sort(data["weight_idx"][rows]), np.arange(10)):
            _fail(f"{path}: group {group} must contain exactly weights 0..9")
    tr_ms = np.asarray(data["tr_ms"], dtype=float)
    vps = np.asarray(data["vps"])
    if not np.isfinite(tr_ms).all() or not np.allclose(tr_ms, 2.61, atol=1.0e-8, rtol=0.0):
        _fail(f"{path}: frozen acquisition requires finite tr_ms=2.61")
    if not np.isfinite(np.asarray(vps, dtype=float)).all() or not np.all(vps == 87):
        _fail(f"{path}: frozen acquisition requires vps=87")
    return groups


def _physical_group_order(data: np.lib.npyio.NpzFile, path: Path) -> tuple[list[int], dict[int, float], list[float]]:
    """Independently re-check the generator's signed-normal geometry audit."""
    groups = _groups_and_weights(data, path)
    positions: dict[int, float] = {}
    normal: np.ndarray | None = None
    for group in groups:
        rows = np.flatnonzero(data["group_idx"] == group)
        affine = np.asarray(data["affine_lps_rc"])[rows]
        if affine.ndim != 3 or affine.shape[1:] != (4, 4) or not np.allclose(affine, affine[0]):
            _fail(f"{path}: group {group} lacks a consistent physical affine")
        candidate = np.cross(affine[0, :3, 0], affine[0, :3, 1])
        length = float(np.linalg.norm(candidate))
        if not np.isfinite(length) or length == 0.0:
            _fail(f"{path}: group {group} has a degenerate stack normal")
        candidate /= length
        if normal is None:
            dominant = int(np.argmax(np.abs(candidate)))
            normal = candidate if candidate[dominant] >= 0.0 else -candidate
        elif not np.isclose(abs(float(np.dot(normal, candidate))), 1.0, atol=1.0e-5):
            _fail(f"{path}: group {group} orientation is inconsistent with the stack normal")
        positions[group] = float(np.dot(affine[0, :3, 3], normal))
    ordered = sorted(groups, key=lambda group: positions[group])
    if len(set(np.round([positions[group] for group in ordered], decimals=8))) != len(ordered):
        _fail(f"{path}: spatial group positions are not unique")
    return ordered, positions, [float(value) for value in normal]


def _stack_observation(record: object, *, stack: str, label: str) -> tuple[Path, str]:
    if not isinstance(record, dict):
        _fail(f"{stack}.{label} must be an object")
    path_value, digest = record.get("path"), record.get("sha256")
    if not isinstance(path_value, str) or not isinstance(digest, str) or len(digest) != 64:
        _fail(f"{stack}.{label} requires path and SHA256")
    path = Path(path_value)
    if not path.is_file():
        _fail(f"{stack}.{label} does not exist: {path}")
    if sha256(path) != digest:
        _fail(f"{stack}.{label} SHA256 mismatch")
    return path.resolve(), digest


def _mapping(value: object, *, stack: str, retained: list[int]) -> list[tuple[int, int]]:
    if not isinstance(value, list):
        _fail(f"{stack}.derived_to_original_group_idx must be a list")
    result: list[tuple[int, int]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"derived_group_idx", "original_group_idx"}:
            _fail(f"{stack}.derived_to_original_group_idx has an invalid entry")
        derived, original = item["derived_group_idx"], item["original_group_idx"]
        if isinstance(derived, bool) or isinstance(original, bool) or not isinstance(derived, int) or not isinstance(original, int):
            _fail(f"{stack}.derived_to_original_group_idx entries must be integer pairs")
        result.append((int(derived), int(original)))
    if [derived for derived, _ in result] != list(range(len(retained))):
        _fail(f"{stack}.derived_to_original_group_idx must enumerate derived groups from zero")
    if [original for _, original in result] != retained:
        _fail(f"{stack}.derived_to_original_group_idx must match retained group order")
    return result


def _same_group_content(source: np.lib.npyio.NpzFile, source_group: int, derived: np.lib.npyio.NpzFile, derived_group: int, *, stack: str) -> None:
    source_rows = np.flatnonzero(source["group_idx"] == source_group)
    derived_rows = np.flatnonzero(derived["group_idx"] == derived_group)
    source_order = source_rows[np.argsort(source["weight_idx"][source_rows])]
    derived_order = derived_rows[np.argsort(derived["weight_idx"][derived_rows])]
    for name in ("images", "masks", "weight_idx", "affine_lps_rc", "pixel_spacing_rc_mm", "slice_thickness_mm", "timing9_ms", "tr_ms", "vps"):
        if not np.array_equal(source[name][source_order], derived[name][derived_order], equal_nan=True):
            _fail(f"{stack}: derived group {derived_group} content/geometry does not match original group {source_group}")


def _gap_audit(value: object, *, stack: str, positions: list[float]) -> None:
    if not isinstance(value, dict) or value.get("regular_gap") is not True or value.get("group_count") != len(positions):
        _fail(f"{stack}.stack_gap_qc must explicitly pass regular_gap")
    if len(positions) > 1:
        gaps = np.diff(positions)
        if not np.allclose(gaps, gaps[0], atol=1.0e-5, rtol=1.0e-5):
            _fail(f"{stack}: derived geometry does not have a regular physical gap")
        if not isinstance(value.get("gap_mm"), (int, float)) or not np.isclose(float(value["gap_mm"]), abs(float(gaps[0])), atol=1.0e-5, rtol=1.0e-5):
            _fail(f"{stack}.stack_gap_qc gap_mm disagrees with derived geometry")


def _strict_profile_contract(payload: dict[str, Any], *, expected_generator_git_sha: str | None, permitted_stack_strides: dict[str, int] | None, permitted_stack_modes: dict[str, str] | None) -> dict[str, Any]:
    """Recreate the generator's resolved profile record from its source JSON."""
    if not isinstance(expected_generator_git_sha, str) or len(expected_generator_git_sha) != 40:
        _fail("strict audit requires an explicit 40-character expected_generator_git_sha established from generation history")
    if payload["git_sha"] != expected_generator_git_sha:
        _fail("generator git_sha does not match explicit expected_generator_git_sha")
    resolved = payload.get("resolved_config")
    if not isinstance(resolved, dict):
        _fail("strict audit requires resolved_config")
    profile_path = resolved.get("profile_path")
    if not isinstance(profile_path, str):
        _fail("resolved_config.profile_path is required")
    profile = load_sparse_profile(Path(profile_path))
    actual_strides = resolved.get("stack_stride_overrides")
    actual_modes = resolved.get("stack_mode_overrides")
    if not isinstance(actual_strides, dict) or not isinstance(actual_modes, dict):
        _fail("resolved_config override records must be objects")
    allowed_strides = dict(permitted_stack_strides or {})
    allowed_modes = dict(permitted_stack_modes or {})
    if actual_strides != allowed_strides or actual_modes != allowed_modes:
        _fail("resolved_config contains unapproved configuration overrides")
    fold_count = payload.get("fold_count")
    fold_id = payload.get("fold_id")
    if not isinstance(fold_count, int) or fold_count < 1:
        _fail("fold_count must be a positive integer")
    if fold_count == 1:
        if fold_id is not None or payload.get("phase") is not None:
            _fail("single-fold profile must use fold_id=null and phase=null")
        requested_fold: int | None = None
    else:
        if not isinstance(fold_id, int) or not 0 <= fold_id < fold_count or payload.get("phase") != fold_id:
            _fail("multi-fold profile fold_id/phase is inconsistent")
        requested_fold = fold_id
    expected = _resolve_profile(profile, fold=requested_fold, stack_strides=allowed_strides, stack_modes=allowed_modes)
    expected_record = {
        **expected,
        "subject": payload["subject"],
        "prepared_root": str(Path(payload["source_prepared_root"]).resolve()),
        "stack_stride_overrides": allowed_strides,
        "stack_mode_overrides": allowed_modes,
    }
    if resolved != expected_record:
        _fail("profile/case/fold resolved_config identity mismatch")
    if payload.get("resolved_config_sha256") != _json_sha256(expected_record):
        _fail("resolved_config_sha256 mismatch")
    for field in ("profile_id", "case_id", "fold_id", "fold_count"):
        if payload.get(field) != expected[field]:
            _fail(f"profile {field} does not match source profile")
    return expected_record


def validate_sparse_profile_manifest(manifest: str | Path, *, subject: str | None = None, strict: bool = False, expected_generator_git_sha: str | None = None, expected_manifest_sha256: str | None = None, permitted_stack_strides: dict[str, int] | None = None, permitted_stack_modes: dict[str, str] | None = None) -> dict[str, Any]:
    """Validate a sparse profile before it is admitted to any reconstruction.

    Returns a shallowly enriched copy with private ``_manifest_path`` and
    ``_manifest_sha256`` provenance fields for launchers.  No file is written.
    """
    path = Path(manifest).resolve()
    actual_manifest_sha256 = sha256(path)
    if expected_manifest_sha256 is not None:
        if not isinstance(expected_manifest_sha256, str) or len(expected_manifest_sha256) != 64 or actual_manifest_sha256 != expected_manifest_sha256:
            _fail("manifest SHA256 does not match the explicit expected_manifest_sha256")
    payload = _read_json(path)
    if payload.get("schema_version") != SCHEMA_VERSION:
        _fail(f"schema_version must be {SCHEMA_VERSION}")
    if payload.get("status") != "ELIGIBLE":
        _fail("status must be ELIGIBLE")
    for field in ("subject", "profile_id", "case_id", "source_prepared_root", "prepared_root", "git_sha"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            _fail(f"{field} must be a non-empty string")
    if subject is not None and payload["subject"] != subject:
        _fail(f"subject is {payload['subject']!r}, expected {subject!r}")
    if len(payload["git_sha"]) != 40:
        _fail("git_sha must identify the sparse generator commit")
    strict_profile = _strict_profile_contract(payload, expected_generator_git_sha=expected_generator_git_sha, permitted_stack_strides=permitted_stack_strides, permitted_stack_modes=permitted_stack_modes) if strict else None
    prepared_root = Path(payload["prepared_root"]).resolve()
    if path != prepared_root / "sparse_manifest.json":
        _fail("manifest path must be prepared_root/sparse_manifest.json")
    if set(payload.get("stacks", {})) != set(STACKS):
        _fail("stack names must be exactly sax/2ch/4ch")
    source_root = Path(payload["source_prepared_root"]).resolve()
    for stack in STACKS:
        item = payload["stacks"][stack]
        if not isinstance(item, dict):
            _fail(f"{stack} record must be an object")
        source_path, _ = _stack_observation(item.get("source_observation"), stack=stack, label="source_observation")
        if source_path != source_root / stack / "observations.npz":
            _fail(f"{stack}.source_observation path disagrees with source_prepared_root")
        with np.load(source_path, allow_pickle=False) as source:
            source_groups, source_positions, source_normal = _physical_group_order(source, source_path)
            ordered = _ids(item.get("ordered_source_group_ids"), field="ordered_source_group_ids", stack=stack)
            retained = _ids(item.get("retained_original_group_ids"), field="retained_original_group_ids", stack=stack)
            heldout = _ids(item.get("heldout_original_group_ids"), field="heldout_original_group_ids", stack=stack)
            if ordered != source_groups:
                _fail(f"{stack}.ordered_source_group_ids does not exactly match source NPZ groups")
            recorded_positions = item.get("ordered_source_positions_mm")
            if not isinstance(recorded_positions, list) or len(recorded_positions) != len(source_groups) or not np.allclose(recorded_positions, [source_positions[group] for group in source_groups], atol=1.0e-5, rtol=1.0e-5):
                _fail(f"{stack}.ordered_source_positions_mm disagrees with physical geometry")
            if not isinstance(item.get("stack_normal_lps"), list) or not np.allclose(item["stack_normal_lps"], source_normal, atol=1.0e-5, rtol=1.0e-5):
                _fail(f"{stack}.stack_normal_lps disagrees with physical geometry")
            if set(retained).intersection(heldout) or set(retained).union(heldout) != set(source_groups):
                _fail(f"{stack}: retained and held-out groups must form an exhaustive disjoint source partition")
            if [group for group in ordered if group in retained] != retained or [group for group in ordered if group in heldout] != heldout:
                _fail(f"{stack}: retained/held-out group order must follow source order")
            counts = (item.get("source_count"), item.get("retained_count"), item.get("heldout_count"))
            if counts != (len(source_groups), len(retained), len(heldout)):
                _fail(f"{stack}: group-count fields disagree with group partition")
            present = item.get("training_stack_present")
            if not isinstance(present, bool) or present != bool(retained):
                _fail(f"{stack}.training_stack_present must match retained groups")
            if strict_profile is not None:
                spec = strict_profile["stacks"][stack]
                if item.get("mode") != spec["mode"] or item.get("stride") != spec.get("stride"):
                    _fail(f"{stack}: resolved effective mode/stride disagrees with source profile")
                if spec["mode"] == "all":
                    expected_retained = ordered
                elif spec["mode"] == "none":
                    expected_retained = []
                else:
                    expected_retained = [group for ordinal, group in enumerate(ordered) if ordinal % int(spec["stride"]) == int(strict_profile["fold_id"])]
                expected_heldout = [group for group in ordered if group not in expected_retained]
                if retained != expected_retained or heldout != expected_heldout:
                    _fail(f"{stack}: selection does not match physical-ordinal profile selection")
            expected_derived = prepared_root / stack / "observations.npz"
            if not present:
                forbidden = {"derived_observation", "derived_to_original_group_idx"}.intersection(item)
                if forbidden or expected_derived.exists():
                    _fail(f"{stack}: empty training stack must not have a derived observation")
                # There is intentionally no derived stack to audit.  Source
                # geometry and its complete group partition were validated
                # above; requiring a fabricated derived gap audit here would
                # contradict the no-fake-observation contract.
                continue
            derived_path, _ = _stack_observation(item.get("derived_observation"), stack=stack, label="derived_observation")
            if derived_path != expected_derived:
                _fail(f"{stack}.derived_observation path disagrees with prepared_root")
            mapping = _mapping(item.get("derived_to_original_group_idx"), stack=stack, retained=retained)
            with np.load(derived_path, allow_pickle=False) as derived:
                derived_groups, derived_positions, _ = _physical_group_order(derived, derived_path)
                if derived_groups != list(range(len(retained))):
                    _fail(f"{stack}: derived NPZ groups must be renumbered consecutively from zero")
                _gap_audit(item.get("stack_gap_qc"), stack=stack, positions=[derived_positions[group] for group in derived_groups])
                for derived_group, original_group in mapping:
                    _same_group_content(source, original_group, derived, derived_group, stack=stack)
    result = dict(payload)
    result["_manifest_path"] = str(path)
    result["_manifest_sha256"] = actual_manifest_sha256
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--subject")
    parser.add_argument("--strict", action="store_true", help="require source-profile reconstruction and explicit generator provenance")
    parser.add_argument("--expected-generator-git-sha", help="generator commit established from the recorded sparse-input generation run")
    parser.add_argument("--expected-manifest-sha256", help="external immutable manifest digest for this named case")
    args = parser.parse_args()
    payload = validate_sparse_profile_manifest(args.manifest, subject=args.subject, strict=args.strict, expected_generator_git_sha=args.expected_generator_git_sha, expected_manifest_sha256=args.expected_manifest_sha256)
    print(json.dumps({"manifest": payload["_manifest_path"], "sha256": payload["_manifest_sha256"], "profile_id": payload["profile_id"], "case_id": payload["case_id"]}, sort_keys=True))


if __name__ == "__main__":
    main()
