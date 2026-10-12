"""Inventory prepared observations without modifying data or historical runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from _common import json_safe, write_json


STACKS = ("sax", "2ch", "4ch")
CSV_COLUMNS = ("subject", "stack", "exists", "group_count", "weights_per_group", "image_hw", "pixel_spacing_rc_mm", "slice_thickness_mm", "inter_group_gap_mm", "tr_ms", "vps", "timing9_min_ms", "timing9_max_ms", "native_reference_available", "myocardium_mask_available", "eligible_full_data", "exclusion_reason")


def _range(values: np.ndarray) -> list[float]:
    return [float(np.min(values)), float(np.max(values))]


def _reference_available(root: Path | None, subject: str, stack: str, *, mask: bool = False) -> bool:
    if root is None or not root.exists():
        return False
    candidates = (root / subject / f"{stack}.npz", root / subject / stack, root / f"{subject}_{stack}.npz")
    if any(path.exists() for path in candidates):
        return True
    manifest = root / "manifest.json"
    if not manifest.is_file():
        manifest = root / "native_reference_manifest.json"
    if not manifest.is_file():
        return False
    try:
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if metadata.get("subject_id") != subject:
        return False
    if mask:
        return stack == "sax" and (root / "sax_myocardium_masks.npz").is_file()
    candidates = (root / f"{stack}.npz", root / stack)
    return any(path.exists() for path in candidates)


def inspect_observations(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        required = {"images", "masks", "group_idx", "weight_idx", "stack_idx", "acquisition_time_ms", "timing9_ms", "affine_lps_rc", "pixel_spacing_rc_mm", "slice_thickness_mm", "tr_ms", "vps"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{source} lacks required observation fields: {missing}")
        groups = data["group_idx"].astype(int)
        weights = data["weight_idx"].astype(int)
        group_ids = sorted(set(groups.tolist()))
        counts = [len(set(weights[groups == group].tolist())) for group in group_ids]
        expected = set(range(10))
        complete = all(set(weights[groups == group].tolist()) == expected for group in group_ids)
        affine = data["affine_lps_rc"]
        origins = np.asarray([affine[np.flatnonzero(groups == group)[0], :3, 3] for group in group_ids], dtype=float)
        gaps = np.linalg.norm(np.diff(origins, axis=0), axis=1) if len(origins) > 1 else np.asarray([], dtype=float)
        return {
            "exists": True,
            "group_count": len(group_ids),
            "weights_per_group": sorted(set(counts)),
            "image_hw": list(map(int, data["images"].shape[1:3])),
            "pixel_spacing_rc_mm": _range(data["pixel_spacing_rc_mm"]),
            "slice_thickness_mm": _range(data["slice_thickness_mm"]),
            "inter_group_gap_mm": None if not len(gaps) else _range(gaps),
            "tr_ms": _range(data["tr_ms"]),
            "vps": _range(data["vps"]),
            "timing9_min_ms": [float(value) for value in np.min(data["timing9_ms"], axis=0)],
            "timing9_max_ms": [float(value) for value in np.max(data["timing9_ms"], axis=0)],
            "eligible_full_data": bool(complete),
            "exclusion_reason": None if complete else "one or more spatial groups do not contain exactly weights 0..9",
        }


def audit_cohort(prepared_root: str | Path, *, preprocessed_root: str | Path | None, native_reference_root: str | Path | None, myocardium_mask_root: str | Path | None) -> dict[str, Any]:
    prepared = Path(prepared_root)
    if not prepared.is_dir():
        raise FileNotFoundError(f"Prepared root does not exist: {prepared}")
    native_root = None if native_reference_root is None else Path(native_reference_root)
    mask_root = None if myocardium_mask_root is None else Path(myocardium_mask_root)
    subjects = sorted(path.name for path in prepared.iterdir() if path.is_dir())
    rows: list[dict[str, Any]] = []
    for subject in subjects:
        for stack in STACKS:
            observation = prepared / subject / stack / "observations.npz"
            if observation.is_file():
                row = inspect_observations(observation)
            else:
                row = {"exists": False, "group_count": 0, "weights_per_group": [], "image_hw": None, "pixel_spacing_rc_mm": None, "slice_thickness_mm": None, "inter_group_gap_mm": None, "tr_ms": None, "vps": None, "timing9_min_ms": None, "timing9_max_ms": None, "eligible_full_data": False, "exclusion_reason": "observations.npz missing"}
            row.update({"subject": subject, "stack": stack, "native_reference_available": _reference_available(native_root, subject, stack), "myocardium_mask_available": _reference_available(mask_root, subject, stack, mask=True)})
            rows.append(row)
    eligible = sorted({row["subject"] for row in rows if row["eligible_full_data"]} & {subject for subject in subjects if all(row["eligible_full_data"] for row in rows if row["subject"] == subject)})
    return {"audit_type": "real_cohort_inventory", "prepared_root": str(prepared.resolve()), "preprocessed_root": None if preprocessed_root is None else str(Path(preprocessed_root).resolve()), "native_reference_root": None if native_root is None else str(native_root.resolve()), "myocardium_mask_root": None if mask_root is None else str(mask_root.resolve()), "subjects": subjects, "eligible_full_data_subjects": eligible, "rows": rows}


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) if not isinstance(row.get(name), (list, dict)) else json_safe(row.get(name)) for name in CSV_COLUMNS})
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--preprocessed-root")
    parser.add_argument("--native-reference-root")
    parser.add_argument("--myocardium-mask-root")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    report = audit_cohort(args.prepared_root, preprocessed_root=args.preprocessed_root, native_reference_root=args.native_reference_root, myocardium_mask_root=args.myocardium_mask_root)
    write_json(args.output_json, json_safe(report))
    _write_csv(args.output_csv, report["rows"])
    print(Path(args.output_json))
    print(Path(args.output_csv))


if __name__ == "__main__":
    main()
