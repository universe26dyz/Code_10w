"""Immutable real-data sparse long-axis inputs for Phase-4R.

Prepared observations are never changed.  Each derived condition contains all
SAX groups and only complete 10-weight 2CH/4CH groups selected by one shared
regular-stride phase.  The manifest is the authority for original identities.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Callable

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
    parser.add_argument("--condition", required=True, choices=tuple(CONDITIONS)); parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    print(json.dumps(build_sparse_condition(args.prepared_root, args.subject, args.condition, args.phase, args.output_root), indent=2, default=str))


if __name__ == "__main__":
    main()
