"""Immutable real-data sparse long-axis inputs for Phase-4R.

Prepared observations are never changed.  Each derived condition contains all
SAX groups and only complete 10-weight 2CH/4CH groups selected by one shared
regular-stride phase.  The manifest is the authority for original identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .contracts import sha256, write_immutable_json

CONDITIONS = {"L100": 1, "L50": 2, "L33": 3, "L25": 4}
REQUIRED_FIELDS = (
    "images", "masks", "group_idx", "weight_idx", "stack_idx", "acquisition_time_ms",
    "timing9_ms", "affine_lps_rc", "pixel_spacing_rc_mm", "slice_thickness_mm", "tr_ms", "vps",
)
PREPROCESSING_LEAKAGE_STATEMENT = (
    "MIND operates within one multi-weight group and MP-PCA operates per spatial slice on "
    "7x7x10 data; held-out spatial groups therefore do not contribute signal to retained groups."
)
PROFILE_SCHEMA = "phase4r_sparse_profile/v2"
PROFILE_MANIFEST_SCHEMA = "phase4r_sparse_profile_v2"
STACKS = ("sax", "2ch", "4ch")


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _read(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = set(REQUIRED_FIELDS).difference(data.files)
        if missing:
            raise ValueError(f"{path}: prepared observations missing required fields {sorted(missing)}")
        return {key: np.asarray(data[key]) for key in data.files}


def _validate_atomic_groups(data: dict[str, np.ndarray], path: Path) -> list[int]:
    groups = [int(value) for value in np.unique(data["group_idx"])]
    if not groups:
        raise ValueError(f"{path}: no spatial groups")
    for group in groups:
        weights = np.sort(data["weight_idx"][data["group_idx"] == group])
        if not np.array_equal(weights, np.arange(10)):
            raise ValueError(f"{path}: group {group} must contain exactly weights 0..9")
    if not np.all(np.isfinite(data["tr_ms"])) or not np.all(np.isfinite(data["vps"])):
        raise ValueError(f"{path}: non-finite TR/VPS")
    if not np.allclose(data["tr_ms"], 2.61) or not np.all(data["vps"] == 87):
        raise ValueError(f"{path}: expected frozen TR=2.61 and VPS=87")
    return groups


def _subset(data: dict[str, np.ndarray], original_ids: list[int]) -> tuple[dict[str, np.ndarray], list[dict[str, int]]]:
    selected = np.isin(data["group_idx"], original_ids)
    value = {key: array[selected].copy() if array.shape[:1] == selected.shape else array.copy() for key, array in data.items()}
    mapping = [{"derived_group_idx": index, "original_group_idx": group} for index, group in enumerate(original_ids)]
    lookup = {group: index for index, group in enumerate(original_ids)}
    value["group_idx"] = np.asarray([lookup[int(group)] for group in value["group_idx"]], dtype=np.int64)
    return value, mapping


def _write_npz(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(path, **data)


def _json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _physical_group_order(data: Mapping[str, np.ndarray], path: Path) -> tuple[list[int], dict[int, float], list[float]]:
    """Order source identities on a consistently signed geometric stack normal."""
    groups = _validate_atomic_groups(dict(data), path)
    positions: dict[int, float] = {}
    normal: np.ndarray | None = None
    for group in groups:
        rows = np.flatnonzero(np.asarray(data["group_idx"], int) == group)
        affine = np.asarray(data["affine_lps_rc"])[rows]
        if affine.ndim != 3 or affine.shape[1:] != (4, 4) or not np.allclose(affine, affine[0]):
            raise ValueError(f"{path}: group {group} lacks one consistent physical affine")
        candidate = np.cross(affine[0, :3, 0], affine[0, :3, 1])
        length = float(np.linalg.norm(candidate))
        if not np.isfinite(length) or length == 0.0:
            raise ValueError(f"{path}: group {group} has a degenerate stack normal")
        candidate /= length
        if normal is None:
            dominant = int(np.argmax(np.abs(candidate)))
            normal = candidate if candidate[dominant] >= 0.0 else -candidate
        elif not np.isclose(abs(float(np.dot(normal, candidate))), 1.0, atol=1e-5):
            raise ValueError(f"{path}: group {group} orientation is inconsistent with the stack normal")
        positions[group] = float(np.dot(affine[0, :3, 3], normal))
    ordered = sorted(groups, key=lambda group: positions[group])
    coordinates = [positions[group] for group in ordered]
    if len(set(np.round(coordinates, decimals=8))) != len(coordinates):
        raise ValueError(f"{path}: spatial group positions are not unique")
    return ordered, positions, [float(value) for value in normal]


def load_sparse_profile(path: str | Path) -> dict[str, Any]:
    """Load one strict v2 custom-profile configuration without legacy inference."""
    profile_path = Path(path)
    value = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("sparse profile must be a JSON object")
    unknown = set(value).difference({"schema", "profile_id", "stacks"})
    if unknown:
        raise ValueError(f"sparse profile contains unknown keys: {sorted(unknown)}")
    if value.get("schema") != PROFILE_SCHEMA or not isinstance(value.get("profile_id"), str) or not value["profile_id"]:
        raise ValueError("sparse profile requires schema and non-empty profile_id")
    stacks = value.get("stacks")
    if not isinstance(stacks, dict) or set(stacks) != set(STACKS):
        raise ValueError("sparse profile must define sax, 2ch, and 4ch stacks exactly")
    normalized: dict[str, dict[str, Any]] = {}
    for stack in STACKS:
        spec = stacks[stack]
        if not isinstance(spec, dict) or set(spec).difference({"mode", "stride"}):
            raise ValueError(f"{stack}: sparse profile stack has unknown keys")
        mode = spec.get("mode")
        if mode not in {"all", "stride", "none"}:
            raise ValueError(f"{stack}: mode must be all, stride, or none")
        stride = spec.get("stride")
        if mode == "stride":
            if type(stride) is not int or stride < 2:
                raise ValueError(f"{stack}: stride mode requires integer stride >= 2")
        elif stride is not None:
            raise ValueError(f"{stack}: only stride mode may define stride")
        normalized[stack] = {"mode": mode, **({"stride": stride} if mode == "stride" else {})}
    if normalized["sax"]["mode"] != "all":
        raise ValueError("custom sparse profiles require SAX mode=all")
    return {"schema": PROFILE_SCHEMA, "profile_id": value["profile_id"], "stacks": normalized, "profile_path": str(profile_path.resolve())}


def _resolve_profile(profile: Mapping[str, Any], *, fold: int | None, stack_strides: Mapping[str, int] | None, stack_modes: Mapping[str, str] | None) -> dict[str, Any]:
    stacks = {name: dict(spec) for name, spec in profile["stacks"].items()}
    for stack, mode in (stack_modes or {}).items():
        if stack not in STACKS or mode not in {"all", "none", "stride"}:
            raise ValueError("stack mode overrides must be STACK=all|stride|none")
        if mode == "stride" and "stride" not in stacks[stack]:
            raise ValueError(f"{stack}: stride override requires an explicit --stack-stride")
        stacks[stack] = {"mode": mode, **({"stride": stacks[stack]["stride"]} if mode == "stride" else {})}
    for stack, stride in (stack_strides or {}).items():
        if stack not in STACKS or type(stride) is not int or stride < 2:
            raise ValueError("stack stride overrides must be STACK=INTEGER>=2")
        stacks[stack] = {"mode": "stride", "stride": stride}
    if stacks["sax"]["mode"] != "all":
        raise ValueError("custom sparse profiles require SAX mode=all")
    stride_values = {int(spec["stride"]) for spec in stacks.values() if spec["mode"] == "stride"}
    if len(stride_values) > 1:
        raise ValueError("all stride stacks in one profile must use the same fold count")
    fold_count = next(iter(stride_values), 1)
    actual_fold = 0 if fold is None else fold
    if type(actual_fold) is not int or not 0 <= actual_fold < fold_count:
        raise ValueError(f"fold must be in [0, {fold_count})")
    case_id = f"{profile['profile_id']}_fold{actual_fold}" if fold_count > 1 else str(profile["profile_id"])
    return {"profile_id": profile["profile_id"], "case_id": case_id, "fold_id": actual_fold if fold_count > 1 else None, "fold_count": fold_count, "stacks": stacks, "profile_path": profile["profile_path"]}


def build_sparse_profile_condition(prepared_root: str | Path, subject: str, profile: str | Path, fold: int | None, output_root: str | Path, *, stack_strides: Mapping[str, int] | None = None, stack_modes: Mapping[str, str] | None = None, dry_run: bool = False, git_sha: str | None = None, audit_stack: Callable[[Path], object] | None = None) -> dict[str, Any]:
    """Build one immutable v2 profile input, selecting by audited physical ordinal."""
    loaded = load_sparse_profile(profile)
    resolved = _resolve_profile(loaded, fold=fold, stack_strides=stack_strides, stack_modes=stack_modes)
    source_root = Path(prepared_root).resolve() / subject
    sources = {stack: source_root / stack / "observations.npz" for stack in STACKS}
    source_data = {stack: _read(path) for stack, path in sources.items()}
    geometry = {stack: _physical_group_order(source_data[stack], sources[stack]) for stack in STACKS}
    selected: dict[str, list[int]] = {}
    heldout: dict[str, list[int]] = {}
    for stack in STACKS:
        ordered, _positions, _normal = geometry[stack]
        spec = resolved["stacks"][stack]
        if spec["mode"] == "all":
            selected[stack] = ordered
        elif spec["mode"] == "none":
            selected[stack] = []
        else:
            selected[stack] = [group for ordinal, group in enumerate(ordered) if ordinal % int(spec["stride"]) == int(resolved["fold_id"])]
        heldout[stack] = [group for group in ordered if group not in selected[stack]]
    if not any(selected.values()):
        raise ValueError("sparse profile has no active training stacks")
    destination = Path(output_root).resolve() / subject / str(resolved["case_id"])
    if destination.exists() and not dry_run:
        raise FileExistsError(f"refusing overwrite: {destination}")
    resolved_record = {**resolved, "subject": subject, "prepared_root": str(source_root), "stack_stride_overrides": dict(stack_strides or {}), "stack_mode_overrides": dict(stack_modes or {})}
    stacks: dict[str, dict[str, Any]] = {}
    for stack in STACKS:
        ordered, positions, normal = geometry[stack]
        source_ids = list(ordered)
        retained, omitted = selected[stack], heldout[stack]
        stacks[stack] = {
            "source_observation": {"path": str(sources[stack]), "sha256": sha256(sources[stack])},
            "mode": resolved["stacks"][stack]["mode"], "stride": resolved["stacks"][stack].get("stride"),
            "training_stack_present": bool(retained), "ordered_source_group_ids": source_ids,
            "ordered_source_positions_mm": [positions[group] for group in source_ids], "stack_normal_lps": normal,
            "retained_original_group_ids": retained, "heldout_original_group_ids": omitted,
            "source_count": len(source_ids), "retained_count": len(retained), "heldout_count": len(omitted),
            "retained_fraction": float(len(retained) / len(source_ids)), "heldout_fraction": float(len(omitted) / len(source_ids)),
            "weight_integrity": "every source, retained, and held-out group has exactly weights 0..9",
        }
    payload: dict[str, Any] = {
        "schema_version": PROFILE_MANIFEST_SCHEMA, "subject": subject, "profile_id": resolved["profile_id"], "case_id": resolved["case_id"],
        "fold_id": resolved["fold_id"], "phase": resolved["fold_id"], "fold_count": resolved["fold_count"], "source_prepared_root": str(source_root),
        "prepared_root": str(destination), "status": "ELIGIBLE", "reason": None, "stacks": stacks,
        "resolved_config": resolved_record, "resolved_config_sha256": _json_sha256(resolved_record), "no_fake_observations": True,
        "preprocessing_leakage_statement": PREPROCESSING_LEAKAGE_STATEMENT,
        "leakage_gate": all(set(selected[stack]).isdisjoint(heldout[stack]) and set(selected[stack]) | set(heldout[stack]) == set(geometry[stack][0]) for stack in STACKS),
        "git_sha": git_sha or _git_sha(), "r1_signal_domain_evaluation": "DEFERRED_BY_DESIGN", "r1_lax_myocardium_evaluation": "OPTIONAL_NOT_REQUIRED_FOR_CURRENT_PILOT",
    }
    if dry_run:
        return {"status": "DRY_RUN", "dry_run": True, "manifest_path": str(destination / "sparse_manifest.json"), "prepared_root": destination, "resolved_config": resolved_record, "manifest": payload}
    destination.mkdir(parents=True)
    for stack in STACKS:
        retained = selected[stack]
        if not retained:
            continue
        derived, mapping = _subset(source_data[stack], retained)
        output = destination / stack / "observations.npz"
        _write_npz(output, derived)
        stacks[stack]["derived_observation"] = {"path": str(output), "sha256": sha256(output)}
        stacks[stack]["derived_to_original_group_idx"] = mapping
        stacks[stack]["stack_gap_qc"] = _audit(output, audit_stack)
    write_immutable_json(destination / "sparse_manifest.json", payload)
    return {"status": "ELIGIBLE", "manifest_path": str(destination / "sparse_manifest.json"), "prepared_root": destination}


def _default_audit(path: Path) -> dict[str, object]:
    try:
        from trad.modules.module_06_rigid_psf.hb1_stack_adapter import audit_hb1_stack
    except ModuleNotFoundError as exc:
        raise RuntimeError("HB1 stack audit requires the project CPU PyTorch environment") from exc
    audit = audit_hb1_stack(path)
    return {"regular_gap": bool(audit.regular_gap), "gap_mm": float(audit.gap_mm), "group_count": int(len(audit.sorted_group_ids))}


def _audit(path: Path, audit_stack: Callable[[Path], object] | None) -> dict[str, object]:
    result = _default_audit(path) if audit_stack is None else audit_stack(path)
    if hasattr(result, "regular_gap"):
        result = {"regular_gap": bool(result.regular_gap), "gap_mm": float(result.gap_mm), "group_count": int(len(result.sorted_group_ids))}
    if not isinstance(result, dict) or result.get("regular_gap") is not True:
        raise ValueError(f"{path}: HB1 regular-gap audit failed")
    return result


def build_sparse_condition(
    prepared_root: str | Path, subject: str, condition: str, phase: int, output_root: str | Path,
    *, git_sha: str | None = None, audit_stack: Callable[[Path], object] | None = None,
) -> dict[str, object]:
    """Create one immutable sparse condition, or an L100 identity manifest.

    ``audit_stack`` exists only to allow CPU-only unit fixtures to supply the
    already-audited geometry result; production callers use ``audit_hb1_stack``.
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown sparse condition: {condition}")
    stride = CONDITIONS[condition]
    if not 0 <= phase < stride:
        raise ValueError(f"invalid phase {phase} for stride {stride}")
    source_root = Path(prepared_root).resolve() / subject
    sources = {stack: source_root / stack / "observations.npz" for stack in ("sax", "2ch", "4ch")}
    source_data = {stack: _read(path) for stack, path in sources.items()}
    source_ids = {stack: _validate_atomic_groups(data, sources[stack]) for stack, data in source_data.items()}
    destination = Path(output_root).resolve() / subject / f"{condition}_fold{phase}"
    manifest_path = destination / "sparse_manifest.json"
    if destination.exists():
        raise FileExistsError(f"refusing overwrite: {destination}")
    selected = {"sax": source_ids["sax"]}
    for stack in ("2ch", "4ch"):
        selected[stack] = [group for group in source_ids[stack] if group % stride == phase]
    heldout = {stack: [group for group in source_ids[stack] if group not in selected[stack]] for stack in ("2ch", "4ch")}
    status, reason = "ELIGIBLE", None
    if condition != "L100" and any(len(selected[stack]) < 2 for stack in ("2ch", "4ch")):
        status, reason = "INELIGIBLE", "fewer than two retained long-axis groups for at least one stack"
    stacks: dict[str, dict[str, object]] = {}
    for stack in ("sax", "2ch", "4ch"):
        stacks[stack] = {
            "source_observation": {"path": str(sources[stack]), "sha256": sha256(sources[stack])},
            "retained_original_group_ids": selected[stack],
            "heldout_original_group_ids": [] if stack == "sax" else heldout[stack],
            "weight_integrity": "every source, retained, and held-out group has exactly weights 0..9",
        }
    payload: dict[str, object] = {
        "schema_version": "phase4r_sparse_v1", "subject": subject, "source_prepared_root": str(source_root),
        "condition": condition, "stride": stride, "phase": phase, "status": status, "reason": reason,
        "stacks": stacks, "preprocessing_leakage_statement": PREPROCESSING_LEAKAGE_STATEMENT,
        "leakage_gate": all(set(selected[stack]).isdisjoint(heldout[stack]) and set(selected[stack]) | set(heldout[stack]) == set(source_ids[stack]) for stack in ("2ch", "4ch")),
        "git_sha": git_sha or _git_sha(), "r1_signal_domain_evaluation": "DEFERRED_BY_DESIGN",
        "r1_lax_myocardium_evaluation": "OPTIONAL_NOT_REQUIRED_FOR_CURRENT_PILOT",
    }
    destination.mkdir(parents=True)
    if condition == "L100":
        payload["identity_source"] = "R0_full_prepared_observations"
        payload["prepared_root"] = str(source_root)
        write_immutable_json(manifest_path, payload)
        return {"status": status, "manifest_path": str(manifest_path), "prepared_root": source_root}
    if status == "ELIGIBLE":
        for stack in ("sax", "2ch", "4ch"):
            derived, mapping = _subset(source_data[stack], selected[stack])
            output = destination / stack / "observations.npz"
            _write_npz(output, derived)
            audit = _audit(output, audit_stack)
            stacks[stack]["derived_observation"] = {"path": str(output), "sha256": sha256(output)}
            stacks[stack]["derived_to_original_group_idx"] = mapping
            stacks[stack]["stack_gap_qc"] = audit
        payload["prepared_root"] = str(destination)
    write_immutable_json(manifest_path, payload)
    return {"status": status, "manifest_path": str(manifest_path), "prepared_root": destination}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", required=True); parser.add_argument("--subject", required=True)
    parser.add_argument("--condition", choices=tuple(CONDITIONS)); parser.add_argument("--phase", type=int)
    parser.add_argument("--profile"); parser.add_argument("--fold", type=int)
    parser.add_argument("--stack-stride", action="append", default=[], metavar="STACK=INTEGER")
    parser.add_argument("--stack-mode", action="append", default=[], metavar="STACK=all|stride|none")
    parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--print-resolved-config", action="store_true")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    if args.profile and args.condition:
        parser.error("--profile and --condition are mutually exclusive")
    if not args.profile:
        if args.condition is None or args.phase is None:
            parser.error("legacy sparse mode requires --condition and --phase")
        if args.fold is not None or args.stack_stride or args.stack_mode or args.dry_run or args.print_resolved_config:
            parser.error("profile-only options require --profile")
        result = build_sparse_condition(args.prepared_root, args.subject, args.condition, args.phase, args.output_root)
    else:
        def parse_pairs(values: list[str], *, allowed: set[str] | None = None) -> dict[str, Any]:
            output: dict[str, Any] = {}
            for item in values:
                key, separator, value = item.partition("=")
                if not separator or key not in STACKS or key in output or (allowed is not None and value not in allowed):
                    parser.error("stack overrides must be unique STACK=VALUE for sax,2ch,4ch")
                output[key] = value
            return output
        raw_strides = parse_pairs(args.stack_stride)
        try:
            strides = {stack: int(value) for stack, value in raw_strides.items()}
        except ValueError:
            parser.error("--stack-stride values must be integers")
        modes = parse_pairs(args.stack_mode, allowed={"all", "stride", "none"})
        result = build_sparse_profile_condition(args.prepared_root, args.subject, args.profile, args.fold, args.output_root, stack_strides=strides, stack_modes=modes, dry_run=args.dry_run)
    print(json.dumps(result if not args.print_resolved_config else result, indent=2, default=str))


if __name__ == "__main__":
    main()
