"""Inventory prepared data, decoder compatibility, and evaluation evidence read-only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from _common import json_safe, sha256, write_json


STACKS = ("sax", "2ch", "4ch")
CSV_COLUMNS = ("subject", "stack", "exists", "group_count", "weights_per_group", "image_hw", "pixel_spacing_rc_mm", "slice_thickness_mm", "inter_group_gap_mm", "tr_ms", "vps", "prepared_input_eligible", "decoder_compatible", "reconstruction_eligible_current", "native_reference_available", "myocardium_mask_available", "quantitative_evaluation_eligible", "exclusion_reason")
EPS = 1e-6


def _range(values: np.ndarray) -> list[float]:
    return [float(np.min(values)), float(np.max(values))]


def _record(path: Path, provenance: str) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path), "provenance": provenance}


def _decoder_contract(path: str | Path) -> dict[str, float | str]:
    manifest = Path(path)
    if not manifest.is_file():
        raise FileNotFoundError(f"Approved decoder manifest does not exist: {manifest}")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    protocol = value.get("protocol_hhz_v1")
    if not value.get("approved_for_paper") or not isinstance(protocol, dict):
        raise ValueError("Decoder manifest is not approved for paper use or lacks protocol_hhz_v1.")
    if protocol.get("tr_ms") is None or protocol.get("vps") is None:
        raise ValueError("Decoder manifest lacks protocol TR/VPS.")
    return {"manifest": str(manifest.resolve()), "manifest_sha256": sha256(manifest), "tr_ms": float(protocol["tr_ms"]), "vps": int(protocol["vps"])}


def inspect_observations(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        required = {"images", "masks", "group_idx", "weight_idx", "stack_idx", "acquisition_time_ms", "timing9_ms", "affine_lps_rc", "pixel_spacing_rc_mm", "slice_thickness_mm", "tr_ms", "vps"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{source} lacks required observation fields: {missing}")
        groups, weights = data["group_idx"].astype(int), data["weight_idx"].astype(int)
        group_ids = sorted(set(groups.tolist()))
        counts = [len(set(weights[groups == group].tolist())) for group in group_ids]
        complete = all(set(weights[groups == group].tolist()) == set(range(10)) for group in group_ids)
        affine = data["affine_lps_rc"]
        origins = np.asarray([affine[np.flatnonzero(groups == group)[0], :3, 3] for group in group_ids], dtype=float)
        gaps = np.linalg.norm(np.diff(origins, axis=0), axis=1) if len(origins) > 1 else np.asarray([], dtype=float)
        return {
            "exists": True, "group_count": len(group_ids), "weights_per_group": sorted(set(counts)),
            "image_hw": list(map(int, data["images"].shape[1:3])), "pixel_spacing_rc_mm": _range(data["pixel_spacing_rc_mm"]),
            "slice_thickness_mm": _range(data["slice_thickness_mm"]), "inter_group_gap_mm": None if not len(gaps) else _range(gaps),
            "tr_ms": _range(data["tr_ms"]), "vps": _range(data["vps"]),
            "timing9_min_ms": [float(item) for item in np.min(data["timing9_ms"], axis=0)],
            "timing9_max_ms": [float(item) for item in np.max(data["timing9_ms"], axis=0)],
            "prepared_input_eligible": bool(complete),
            "exclusion_reason": None if complete else "one or more spatial groups do not contain exactly weights 0..9",
        }


def _subject_root(root: Path, subject: str) -> Path | None:
    candidate = root / "subjects" / subject
    return candidate if candidate.is_dir() else None


def _native_evidence(subject: str, stack: str, roots: list[Path], bundle_root: Path | None) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if bundle_root is not None and bundle_root.is_dir():
        manifest = bundle_root / "native_reference_manifest.json"
        if manifest.is_file():
            try:
                if json.loads(manifest.read_text(encoding="utf-8")).get("subject_id") == subject and (bundle_root / f"{stack}.npz").is_file():
                    found.extend([_record(bundle_root / f"{stack}.npz", "Phase-0 Code_10w native-reference bundle"), _record(manifest, "Phase-0 Code_10w native-reference provenance")])
            except json.JSONDecodeError:
                pass
    for root in roots:
        subject_root = _subject_root(root, subject)
        if subject_root is None:
            continue
        base = subject_root / "results" / "2D_fit_first" / "experiments" / "20260822_v5_full_pipeline_v1"
        maps = [base / "fitting_qc" / "NIfTI_Slices" / f"{stack}_{parameter}_stack_r0100.nii.gz" for parameter in ("T1", "T2")]
        config = base / "config.yaml"
        if all(path.is_file() for path in maps):
            found.extend(_record(path, "2D-fit-first v5 fitting_qc native map") for path in maps)
            if config.is_file():
                found.append(_record(config, "2D-fit-first v5 experiment provenance"))
    return found


def _mask_evidence(subject: str, roots: list[Path], bundle_root: Path | None) -> tuple[list[dict[str, str]], str | None]:
    if bundle_root is not None and bundle_root.is_dir():
        manifest, mask = bundle_root / "manifest.json", bundle_root / "sax_myocardium_masks.npz"
        if manifest.is_file() and mask.is_file():
            try:
                if json.loads(manifest.read_text(encoding="utf-8")).get("subject_id") == subject:
                    return ([_record(mask, "Phase-0 Code_10w myocardium-mask bundle"), _record(manifest, "Phase-0 Code_10w myocardium-mask provenance")], None)
            except json.JSONDecodeError:
                pass
    for root in roots:
        subject_root = _subject_root(root, subject)
        if subject_root is None:
            continue
        roi = subject_root / "data" / "roi"
        pending = list(roi.rglob("ROI_MANUAL_STEP_REQUIRED.md")) if roi.is_dir() else []
        if pending:
            return ([_record(pending[0], "ROI status: manual annotation required")], "manual myocardium ROI annotation and compatible mask export are required; legacy evaluation masks are not reusable")
        direct = [roi / "LV_myocardium_full.nii.gz", roi / "LV_myocardium_core.nii.gz"]
        if all(path.is_file() for path in direct):
            return ([_record(path, "subject SAX ROI mask") for path in direct], None)
        evaluated = subject_root / "results" / "2D_fit_first" / "evaluations" / "20260823_dual_full_v1" / "nesvor_v5_128_w1" / "masks" / "native_sax"
        existing = [evaluated / "myocardium_full.nii.gz", evaluated / "myocardium_core.nii.gz"]
        if all(path.is_file() for path in existing):
            return ([_record(path, "2D-fit-first formal evaluation native SAX mask") for path in existing], None)
    return ([], "no local myocardium full/core mask was found; migrate an approved subject-specific mask bundle")


def _compatible(row: dict[str, Any], decoder: dict[str, float | str]) -> tuple[bool, str | None]:
    if not row["exists"]:
        return False, "observations.npz missing"
    tr, vps = row["tr_ms"], row["vps"]
    if abs(tr[0] - tr[1]) > EPS or abs(tr[0] - float(decoder["tr_ms"])) > EPS:
        return False, f"TR mismatch: observations={tr}, approved_decoder={decoder['tr_ms']}"
    if vps[0] != vps[1] or int(vps[0]) != int(decoder["vps"]):
        return False, f"VPS mismatch: observations={vps}, approved_decoder={decoder['vps']}"
    return True, None


def audit_cohort(prepared_root: str | Path, *, preprocessed_root: str | Path | None, native_reference_root: str | Path | None, myocardium_mask_root: str | Path | None, decoder_manifest: str | Path | None = None, reference_search_roots: list[str | Path] | None = None) -> dict[str, Any]:
    prepared = Path(prepared_root)
    if not prepared.is_dir():
        raise FileNotFoundError(f"Prepared root does not exist: {prepared}")
    decoder = None if decoder_manifest is None else _decoder_contract(decoder_manifest)
    roots = [Path(path) for path in (reference_search_roots or [])]
    subjects = sorted(path.name for path in prepared.iterdir() if path.is_dir())
    rows: list[dict[str, Any]] = []
    evidence: dict[str, dict[str, Any]] = {}
    for subject in subjects:
        native = {stack: _native_evidence(subject, stack, roots, Path(native_reference_root) if native_reference_root else None) for stack in STACKS}
        masks, mask_reason = _mask_evidence(subject, roots, Path(myocardium_mask_root) if myocardium_mask_root else None)
        evidence[subject] = {"native_reference_evidence": native, "myocardium_mask_evidence": masks, "myocardium_mask_reason": mask_reason}
        for stack in STACKS:
            observation = prepared / subject / stack / "observations.npz"
            row = inspect_observations(observation) if observation.is_file() else {"exists": False, "group_count": 0, "weights_per_group": [], "image_hw": None, "pixel_spacing_rc_mm": None, "slice_thickness_mm": None, "inter_group_gap_mm": None, "tr_ms": None, "vps": None, "timing9_min_ms": None, "timing9_max_ms": None, "prepared_input_eligible": False, "exclusion_reason": "observations.npz missing"}
            compatible, reason = _compatible(row, decoder) if decoder else (False, "approved decoder manifest not supplied")
            row.update({"subject": subject, "stack": stack, "decoder_compatible": compatible, "decoder_incompatibility_reason": reason, "reconstruction_eligible_current": bool(row["prepared_input_eligible"] and compatible), "native_reference_available": bool(native[stack]), "myocardium_mask_available": bool(masks) and mask_reason is None if stack == "sax" else False})
            rows.append(row)
    summary = []
    for subject in subjects:
        current = [row for row in rows if row["subject"] == subject]
        prepared_ok, decoder_ok = all(row["prepared_input_eligible"] for row in current), all(row["decoder_compatible"] for row in current)
        reconstruction_ok = all(row["reconstruction_eligible_current"] for row in current)
        native_ok = all(row["native_reference_available"] for row in current)
        masks_ok = bool(evidence[subject]["myocardium_mask_evidence"]) and evidence[subject]["myocardium_mask_reason"] is None
        quantitative_ok = reconstruction_ok and native_ok and masks_ok
        reasons = [row["decoder_incompatibility_reason"] for row in current if row["decoder_incompatibility_reason"]]
        if not native_ok: reasons.append("native T1/T2 reference missing for one or more stacks")
        if not masks_ok: reasons.append(evidence[subject]["myocardium_mask_reason"])
        for row in current: row["quantitative_evaluation_eligible"] = quantitative_ok
        summary.append({"subject": subject, "prepared_input_eligible": prepared_ok, "decoder_compatible": decoder_ok, "reconstruction_eligible_current": reconstruction_ok, "native_reference_available": native_ok, "myocardium_mask_available": masks_ok, "quantitative_evaluation_eligible": quantitative_ok, "reasons": sorted(set(reasons)), **evidence[subject]})
    return {"audit_type": "real_cohort_inventory_phase0_5", "prepared_root": str(prepared.resolve()), "preprocessed_root": None if preprocessed_root is None else str(Path(preprocessed_root).resolve()), "decoder_contract": decoder, "reference_search_roots": [str(path.resolve()) for path in roots], "rows": rows, "subject_summary": summary}


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    target = Path(path)
    if target.exists(): raise FileExistsError(f"Refusing to overwrite audit output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n"); writer.writeheader()
        for row in rows: writer.writerow({name: row.get(name) if not isinstance(row.get(name), (list, dict)) else json.dumps(json_safe(row.get(name))) for name in CSV_COLUMNS})
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", required=True); parser.add_argument("--preprocessed-root")
    parser.add_argument("--native-reference-root"); parser.add_argument("--myocardium-mask-root")
    parser.add_argument("--decoder-manifest", required=True); parser.add_argument("--reference-search-root", action="append", default=[])
    parser.add_argument("--output-json", required=True); parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    report = audit_cohort(args.prepared_root, preprocessed_root=args.preprocessed_root, native_reference_root=args.native_reference_root, myocardium_mask_root=args.myocardium_mask_root, decoder_manifest=args.decoder_manifest, reference_search_roots=args.reference_search_root)
    write_json(args.output_json, json_safe(report)); _write_csv(args.output_csv, report["rows"])
    print(Path(args.output_json)); print(Path(args.output_csv))


if __name__ == "__main__": main()
