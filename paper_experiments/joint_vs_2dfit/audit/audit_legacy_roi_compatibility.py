"""Read-only provenance and geometry audit for legacy SAX myocardium ROIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import nibabel as nib
import numpy as np
import pydicom

from _common import json_safe, sha256, write_json


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation_common.myocardium.legacy_masks import audit_legacy_subject, classify_affine_relation


SUBJECTS = ("CYJ", "DYL", "DYZ", "HHZ", "HJL")
LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])
EPS = 2e-4


def _array_sha256(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    array = np.ascontiguousarray(values)
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _mat_data(path: Path) -> dict[str, Any]:
    if not path.is_file() or not h5py.is_hdf5(path):
        raise FileNotFoundError(f"Expected HDF5 MAT file: {path}")
    with h5py.File(path, "r") as handle:
        required = ("Mag", "Acq_time", "TR", "VPS")
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(f"{path} lacks {missing}")
        arrays = {name: np.asarray(handle[name]) for name in required}
    return {
        "path": str(path.resolve()), "sha256": sha256(path),
        "mag_shape": list(arrays["Mag"].shape), "mag_sha256": _array_sha256(arrays["Mag"]),
        "acquisition_time_shape": list(arrays["Acq_time"].shape),
        "acquisition_time_sha256": _array_sha256(arrays["Acq_time"]),
        "tr": float(arrays["TR"].reshape(-1)[0]), "vps": float(arrays["VPS"].reshape(-1)[0]),
        "_mag": arrays["Mag"], "_acquisition_time": arrays["Acq_time"],
    }


def _public_mat(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _dicom_record(path: Path) -> dict[str, Any]:
    tags = ["SOPInstanceUID", "SeriesInstanceUID", "AcquisitionTime", "Rows", "Columns", "PixelSpacing", "ImagePositionPatient", "ImageOrientationPatient", "InstanceNumber"]
    dataset = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=tags)
    return {
        "path": str(path.resolve()), "sop_instance_uid": str(dataset.SOPInstanceUID),
        "series_instance_uid": str(dataset.SeriesInstanceUID), "acquisition_time": str(dataset.AcquisitionTime),
        "rows": int(dataset.Rows), "columns": int(dataset.Columns),
        "pixel_spacing": [float(value) for value in dataset.PixelSpacing],
        "image_position_patient": [float(value) for value in dataset.ImagePositionPatient],
        "image_orientation_patient": [float(value) for value in dataset.ImageOrientationPatient],
        "instance_number": int(dataset.InstanceNumber), "sha256": sha256(path),
    }


def _dicom_inventory(directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        return {"status": "missing", "directory": str(directory), "records_by_uid": {}}
    records = []
    for path in sorted(item for item in directory.iterdir() if item.is_file()):
        try:
            records.append(_dicom_record(path))
        except Exception:
            continue
    by_uid = {item["sop_instance_uid"]: item for item in records}
    if len(by_uid) != len(records):
        raise ValueError(f"Duplicate SOP Instance UID in {directory}")
    content = hashlib.sha256()
    for item in sorted(records, key=lambda value: value["sop_instance_uid"]):
        content.update(item["sop_instance_uid"].encode("ascii")); content.update(item["sha256"].encode("ascii"))
    return {
        "status": "present", "directory": str(directory.resolve()), "file_count": len(records),
        "sop_instance_uids": sorted(by_uid), "sop_instance_uids_sha256": hashlib.sha256("\n".join(sorted(by_uid)).encode("ascii")).hexdigest(),
        "source_content_sha256": content.hexdigest(), "series_instance_uids": sorted({item["series_instance_uid"] for item in records}),
        "rows": sorted({item["rows"] for item in records}), "columns": sorted({item["columns"] for item in records}),
        "pixel_spacing": sorted({tuple(item["pixel_spacing"]) for item in records}),
        "records_by_uid": by_uid,
    }


def _dicom_comparison(legacy: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    if legacy["status"] != "present" or current["status"] != "present":
        return {"status": "unavailable", "uid_match": None, "reason": "legacy or current DICOM directory unavailable"}
    legacy_uids, current_uids = set(legacy["records_by_uid"]), set(current["records_by_uid"])
    uid_match = legacy_uids == current_uids
    keys = ("rows", "columns", "pixel_spacing", "image_position_patient", "image_orientation_patient", "acquisition_time")
    geometry_match = uid_match and all(
        all(np.allclose(legacy["records_by_uid"][uid][key], current["records_by_uid"][uid][key], atol=1e-6, rtol=0.0) if isinstance(legacy["records_by_uid"][uid][key], list) else legacy["records_by_uid"][uid][key] == current["records_by_uid"][uid][key] for key in keys)
        for uid in legacy_uids
    )
    return {"status": "compared", "uid_match": uid_match, "geometry_match": geometry_match,
            "legacy_only_uid_count": len(legacy_uids - current_uids), "current_only_uid_count": len(current_uids - legacy_uids)}


def _current_grid(prepared_manifest: Path, current_dicom: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(prepared_manifest.read_text(encoding="utf-8"))
    groups = value.get("groups", [])
    if not groups or [group["group_idx"] for group in groups] != list(range(len(groups))):
        raise ValueError(f"{prepared_manifest} has no contiguous SAX groups")
    if current_dicom["status"] != "present" or len(current_dicom["rows"]) != 1 or len(current_dicom["columns"]) != 1:
        raise ValueError("Current DICOM must supply one full image Rows/Columns geometry")
    full, crop = np.asarray(groups[0]["full_affine_lps_rc"], float), np.asarray(groups[0]["affine_lps_rc"], float)
    step = np.asarray(groups[1]["full_affine_lps_rc"], float)[:3, 3] - full[:3, 3] if len(groups) > 1 else full[:3, 2]
    crop_shape = tuple(int(item) for item in groups[0]["image_shape_rows_cols"])
    offsets = []
    for index, group in enumerate(groups):
        full_group, crop_group = np.asarray(group["full_affine_lps_rc"], float), np.asarray(group["affine_lps_rc"], float)
        if tuple(group["image_shape_rows_cols"]) != crop_shape or not np.allclose(full_group[:3, 3], full[:3, 3] + index * step, atol=EPS, rtol=0.0):
            raise ValueError("Current SAX groups do not have stable ordered full/crop geometry")
        transform = np.linalg.solve(full_group, crop_group)
        if not np.allclose(transform[:3, :3], np.eye(3), atol=EPS, rtol=0.0):
            raise ValueError("Current crop axes differ from full DICOM axes")
        offset = np.rint(transform[:3, 3]).astype(int)
        if not np.allclose(transform[:3, 3], offset, atol=EPS, rtol=0.0) or offset[2] != 0:
            raise ValueError("Current crop is not an integer in-plane transform")
        offsets.append(offset[:2].tolist())
    if len({tuple(item) for item in offsets}) != 1:
        raise ValueError("Current crop offset changes across SAX groups")
    full_stack, crop_stack = full.copy(), crop.copy()
    full_stack[:3, 2] = step; crop_stack[:3, 2] = step
    hb1 = [group["hb1_sop_instance_uid"] for group in groups]
    missing = [uid for uid in hb1 if uid not in current_dicom["records_by_uid"]]
    if missing:
        raise ValueError("Current prepared HB1 SOP UIDs are absent from current source DICOM")
    return {
        "prepared_manifest": {"path": str(prepared_manifest.resolve()), "sha256": sha256(prepared_manifest)},
        "full_shape_row_col_group": [current_dicom["rows"][0], current_dicom["columns"][0], len(groups)],
        "crop_shape_group_row_col": [len(groups), *crop_shape], "crop_offset_full_row_col": offsets[0],
        "full_affine_ras_rc": (LPS_TO_RAS @ full_stack).tolist(), "crop_affine_ras_rc": (LPS_TO_RAS @ crop_stack).tolist(),
        "hb1_slice_identity": [{"group_idx": group["group_idx"], "sop_instance_uid": group["hb1_sop_instance_uid"], "source": current_dicom["records_by_uid"][group["hb1_sop_instance_uid"]]} for group in groups],
        "required_mask_space": "current Code_10w SAX [group,row,col] cropped native evaluation grid",
    }


def classify_compatibility(*, uid_match: bool | None, raw_pixels_match: bool | None, geometry_class: str, deterministic_crop: bool, marker_text: str | None) -> dict[str, Any]:
    """Return the Phase 0.6 four-state result; marker prose is never decisive alone."""

    reasons = []
    if uid_match is False:
        return {"status": "incompatible", "reasons": ["SOP Instance UID sets differ"]}
    if raw_pixels_match is False:
        return {"status": "incompatible", "reasons": ["legacy and current raw Mag arrays differ"]}
    if geometry_class == "NON_EQUIVALENT_GRID":
        return {"status": "incompatible", "reasons": ["ROI geometry is not an exact native/current lattice relation"]}
    if uid_match is None and raw_pixels_match is not True:
        return {"status": "unresolved_manual_review", "reasons": ["insufficient same-acquisition provenance"]}
    if geometry_class == "IDENTICAL_GRID" and not deterministic_crop:
        status = "reusable_exact"
    elif geometry_class in {"IDENTICAL_GRID", "EXACT_REORIENTABLE_GRID"} and deterministic_crop:
        status = "reusable_with_deterministic_transform"
    else:
        status = "unresolved_manual_review"
        reasons.append("geometry evidence does not specify an exact direct or crop transform")
    if marker_text:
        reasons.append("ROI marker was recorded but is not an acquisition/geometry incompatibility proof")
    return {"status": status, "reasons": reasons}


def _marker(source_dir: Path) -> dict[str, Any] | None:
    paths = sorted(source_dir.glob("ROI_MANUAL_STEP_REQUIRED.md"))
    if not paths:
        return None
    path = paths[0]
    return {"path": str(path.resolve()), "sha256": sha256(path), "text": path.read_text(encoding="utf-8"),
            "interpretation": "Workflow-state note: it requests manual labels; it is not itself evidence of a changed acquisition or incompatible geometry."}


def audit_subject(subject: str, *, subjects_root: Path, preprocessed_root: Path, prepared_root: Path, current_dicom_root: Path) -> dict[str, Any]:
    legacy_subject = subjects_root / subject
    legacy_meta = legacy_subject / "data" / "metadata" / "predata.sax.json"
    legacy_source = Path(json.loads(legacy_meta.read_text(encoding="utf-8"))["dicom_dir"])
    legacy_mat = _mat_data(legacy_subject / "data" / "matlab_preprocessed" / "native" / "sax_data_all.mat")
    current_mat = _mat_data(preprocessed_root / subject / "sax" / "preprocessed.mat")
    raw_match = bool(np.array_equal(legacy_mat["_mag"], current_mat["_mag"]) and np.array_equal(legacy_mat["_acquisition_time"], current_mat["_acquisition_time"]) and legacy_mat["tr"] == current_mat["tr"] and legacy_mat["vps"] == current_mat["vps"])
    legacy_dicom, current_dicom = _dicom_inventory(legacy_source), _dicom_inventory(current_dicom_root / subject / "sax")
    dicom = _dicom_comparison(legacy_dicom, current_dicom)
    grid = _current_grid(prepared_root / subject / "sax" / "manifest.json", current_dicom)
    legacy_roi = audit_legacy_subject(subject, subjects_root)
    marker = _marker(Path(legacy_roi["source_directory"])) if legacy_roi.get("status") == "PASS" else None
    geometry_class, deterministic_crop = "NON_EQUIVALENT_GRID", False
    geometry_chain: dict[str, Any] = {"legacy_mask_audit_status": legacy_roi.get("status")}
    if legacy_roi.get("status") == "PASS":
        composite = nib.load(legacy_roi["composite_reference"]["path"])
        relation = classify_affine_relation(composite.affine, np.asarray(grid["full_affine_ras_rc"], float), composite.shape, grid["full_shape_row_col_group"])
        direct = classify_affine_relation(nib.load(legacy_roi["manual_masks"]["outer_filled"]["path"]).affine, np.asarray(grid["crop_affine_ras_rc"], float), legacy_roi["manual_masks"]["outer_filled"]["shape_xyz"], [grid["crop_shape_group_row_col"][2], grid["crop_shape_group_row_col"][1], grid["crop_shape_group_row_col"][0]])
        relations_ok = all(item["status"] == "PASS" for item in legacy_roi["geometry_relations_to_composite"].values())
        geometry_class = "IDENTICAL_GRID" if direct["geometry_class"] == "IDENTICAL_GRID" else "EXACT_REORIENTABLE_GRID" if relations_ok and relation["status"] == "PASS" else "NON_EQUIVALENT_GRID"
        deterministic_crop = geometry_class != "NON_EQUIVALENT_GRID" and grid["crop_offset_full_row_col"] != [0, 0]
        geometry_chain.update({"legacy_label_to_composite": legacy_roi["geometry_relations_to_composite"], "composite_to_current_full": relation, "legacy_label_to_current_crop": direct, "current_crop_offset_full_row_col": grid["crop_offset_full_row_col"]})
    compatibility = classify_compatibility(uid_match=dicom["uid_match"], raw_pixels_match=raw_match, geometry_class=geometry_class, deterministic_crop=deterministic_crop, marker_text=None if marker is None else marker["text"])
    return {
        "subject_id": subject, "compatibility": compatibility, "legacy_source_dicom": {key: value for key, value in legacy_dicom.items() if key != "records_by_uid"},
        "current_source_dicom": {key: value for key, value in current_dicom.items() if key != "records_by_uid"}, "source_dicom_comparison": dicom,
        "legacy_mat": _public_mat(legacy_mat), "current_mat": _public_mat(current_mat),
        "raw_data_identity": {"mag_exact": bool(np.array_equal(legacy_mat["_mag"], current_mat["_mag"])), "acquisition_time_exact": bool(np.array_equal(legacy_mat["_acquisition_time"], current_mat["_acquisition_time"])), "same_protocol": legacy_mat["tr"] == current_mat["tr"] and legacy_mat["vps"] == current_mat["vps"]},
        "current_evaluation_mask_space": grid, "legacy_roi": legacy_roi, "geometry_chain": geometry_chain, "marker": marker,
        "migration_plan": {"allowed": compatibility["status"] in {"reusable_exact", "reusable_with_deterministic_transform"}, "source": None if legacy_roi.get("status") != "PASS" else legacy_roi["source_directory"], "destination": "must be explicitly supplied to reference/migrate_verified_legacy_roi.py; this audit does not write masks", "method": "exact axis permutation/reversal plus integer crop; nearest-neighbor/no interpolation only", "required_qc": "inverse round-trip label check and overlay against the explicit native reference"},
    }


def audit_cohort(*, subjects_root: str | Path, preprocessed_root: str | Path, prepared_root: str | Path, current_dicom_root: str | Path, subjects: tuple[str, ...] = SUBJECTS) -> dict[str, Any]:
    roots = {"subjects_root": Path(subjects_root), "preprocessed_root": Path(preprocessed_root), "prepared_root": Path(prepared_root), "current_dicom_root": Path(current_dicom_root)}
    entries = {subject: audit_subject(subject, **roots) for subject in subjects}
    return {"audit_type": "legacy_roi_compatibility_phase0_6", "subjects": entries,
            "summary": [{"subject_id": subject, "status": value["compatibility"]["status"], "reasons": value["compatibility"]["reasons"]} for subject, value in entries.items()]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects-root", required=True); parser.add_argument("--preprocessed-root", required=True)
    parser.add_argument("--prepared-root", required=True); parser.add_argument("--current-dicom-root", required=True)
    parser.add_argument("--subject", action="append", choices=SUBJECTS); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit_cohort(subjects_root=args.subjects_root, preprocessed_root=args.preprocessed_root, prepared_root=args.prepared_root, current_dicom_root=args.current_dicom_root, subjects=tuple(args.subject or SUBJECTS))
    write_json(args.output, json_safe(report)); print(Path(args.output))


if __name__ == "__main__":
    main()
