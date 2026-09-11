"""Build a strict 10-weight observation manifest from MAT UIDs and current DICOMs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom

from .geometry import crop_affine_lps_rc, dicom_lps_affine_rc
from .timing import compute_duration_before_acq


STACK_INDEX = {"sax": 0, "2ch": 1, "4ch": 2}
GEOMETRY_FIELDS = ("ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing")


@dataclass(frozen=True)
class DicomRecord:
    sop_instance_uid: str
    series_instance_uid: str
    path: Path
    dataset: Any


def _required_text(dataset: Any, field: str, path: Path) -> str:
    value = str(getattr(dataset, field, "")).strip()
    if not value:
        raise ValueError(f"DICOM lacks required {field}: {path}")
    return value


def scan_dicom_by_sop(dicom_dir: str | Path) -> dict[str, DicomRecord]:
    """Index exactly one current stack directory by SOPInstanceUID; never by filename."""

    dicom_dir = Path(dicom_dir)
    if not dicom_dir.is_dir():
        raise FileNotFoundError(f"DICOM directory does not exist: {dicom_dir}")
    paths = sorted(dicom_dir.glob("*.dcm"))
    if not paths:
        raise ValueError(f"No .dcm files found in DICOM directory: {dicom_dir}")
    by_sop: dict[str, DicomRecord] = {}
    for path in paths:
        dataset = pydicom.dcmread(path, stop_before_pixels=True)
        sop = _required_text(dataset, "SOPInstanceUID", path)
        series = _required_text(dataset, "SeriesInstanceUID", path)
        if sop in by_sop:
            raise ValueError(
                f"SOPInstanceUID is not unique in --dicom-dir: {sop} in "
                f"{by_sop[sop].path} and {path}"
            )
        by_sop[sop] = DicomRecord(sop, series, path, dataset)
    return by_sop


def _geometry_value(dataset: Any, field: str, path: Path) -> np.ndarray:
    if not hasattr(dataset, field):
        raise ValueError(f"DICOM lacks {field}: {path}")
    values = np.asarray(getattr(dataset, field), dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"DICOM {field} contains non-finite values: {path}")
    return values


def _verify_group_geometry(records: list[DicomRecord], group_idx: int) -> float:
    if len(records) != 10:
        raise ValueError(f"Group {group_idx} must contain 10 records, got {len(records)}.")
    reference = records[0]
    for field in GEOMETRY_FIELDS:
        reference_value = _geometry_value(reference.dataset, field, reference.path)
        for weight_idx, record in enumerate(records[1:], start=1):
            current = _geometry_value(record.dataset, field, record.path)
            if current.shape != reference_value.shape or not np.allclose(
                current, reference_value, rtol=1e-6, atol=1e-6
            ):
                raise ValueError(
                    f"Group {group_idx} weight {weight_idx} {field} differs from HB1; "
                    f"HB1={reference_value.tolist()}, current={current.tolist()}, file={record.path}"
                )
    for field in ("Rows", "Columns"):
        reference_value = int(getattr(reference.dataset, field, -1))
        if reference_value <= 0:
            raise ValueError(f"Group {group_idx} HB1 lacks valid {field}: {reference.path}")
        for weight_idx, record in enumerate(records[1:], start=1):
            current = int(getattr(record.dataset, field, -1))
            if current != reference_value:
                raise ValueError(
                    f"Group {group_idx} weight {weight_idx} {field}={current} differs from "
                    f"HB1 {reference_value}; file={record.path}"
                )
    if not hasattr(reference.dataset, "SliceThickness"):
        raise ValueError(f"Group {group_idx} HB1 lacks SliceThickness: {reference.path}")
    thickness = float(reference.dataset.SliceThickness)
    if not np.isfinite(thickness) or thickness <= 0:
        raise ValueError(f"Group {group_idx} HB1 SliceThickness is invalid: {thickness}")
    for weight_idx, record in enumerate(records[1:], start=1):
        if not hasattr(record.dataset, "SliceThickness"):
            raise ValueError(f"Group {group_idx} weight {weight_idx} lacks SliceThickness: {record.path}")
        current = float(record.dataset.SliceThickness)
        if not np.isclose(current, thickness, rtol=1e-6, atol=1e-6):
            raise ValueError(
                f"Group {group_idx} weight {weight_idx} SliceThickness={current} differs from "
                f"HB1 {thickness}; file={record.path}"
            )
    return thickness


def build_manifest(
    preprocessed: dict[str, Any], dicom_dir: str | Path, stack_name: str, max_groups: int | None
) -> dict[str, Any]:
    """Resolve stored UIDs to current DICOM paths and build group-shared geometry.

    Geometry remains in DICOM LPS millimetres. ``affine_lps_rc`` maps zero-based
    cropped `[row, col, slice]` coordinates; later NeSVoR adaptation must make
    any coordinate-convention conversion explicitly at its boundary.
    """

    stack_key = stack_name.lower()
    if stack_key not in STACK_INDEX:
        raise ValueError(f"--stack must be one of {sorted(STACK_INDEX)}, got {stack_name!r}.")
    n_groups = int(preprocessed["mag_crop"].shape[3])
    if max_groups is None:
        selected_groups = n_groups
    else:
        if max_groups < 1 or max_groups > n_groups:
            raise ValueError(f"--max-groups must be within [1, {n_groups}], got {max_groups}.")
        selected_groups = int(max_groups)
    current_by_sop = scan_dicom_by_sop(dicom_dir)
    observations: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for group_idx in range(selected_groups):
        uid_start = group_idx * 10
        group_uids = preprocessed["sop_instance_uids"][uid_start : uid_start + 10]
        if len(group_uids) != 10:
            raise ValueError(f"Preprocessed group {group_idx} does not contain exactly 10 UIDs.")
        missing = [uid for uid in group_uids if uid not in current_by_sop]
        if missing:
            raise ValueError(
                f"DICOM directory cannot resolve stored SOPInstanceUIDs for group {group_idx}: {missing}"
            )
        records = [current_by_sop[uid] for uid in group_uids]
        stored_series = preprocessed["series_instance_uids"][uid_start : uid_start + 10]
        if [record.series_instance_uid for record in records] != stored_series:
            raise ValueError(f"Current DICOM SeriesInstanceUID differs from preprocessed metadata in group {group_idx}.")
        thickness = _verify_group_geometry(records, group_idx)
        hb1 = records[0]
        full_affine = dicom_lps_affine_rc(hb1.dataset, thickness)
        cropped_affine = crop_affine_lps_rc(
            full_affine,
            preprocessed["crop_row_start_zero_based"],
            preprocessed["crop_col_start_zero_based"],
        )
        image_shape = tuple(int(v) for v in preprocessed["mag_crop"].shape[:2])
        if (int(hb1.dataset.Rows), int(hb1.dataset.Columns)) != tuple(
            int(v) for v in preprocessed["mag"].shape[:2]
        ):
            raise ValueError(
                f"HB1 full DICOM matrix does not match MAT Mag dimensions in group {group_idx}: "
                f"DICOM={(hb1.dataset.Rows, hb1.dataset.Columns)}, MAT={preprocessed['mag'].shape[:2]}"
            )
        timing10 = compute_duration_before_acq(
            preprocessed["acq_time_ms"][:, group_idx],
            preprocessed["tr_ms"],
            preprocessed["vps"],
            preprocessed["ti_ms"],
            preprocessed["t2prep_ms"],
        )
        timing9 = timing10[1:]
        pixel_spacing_rc = np.asarray(hb1.dataset.PixelSpacing, dtype=np.float64)
        group = {
            "group_idx": group_idx,
            "hb1_sop_instance_uid": hb1.sop_instance_uid,
            "series_instance_uid": hb1.series_instance_uid,
            "weight_count": 10,
            "image_shape_rows_cols": image_shape,
            "affine_lps_rc": cropped_affine,
            "full_affine_lps_rc": full_affine,
            "pixel_spacing_rc_mm": pixel_spacing_rc,
            "slice_thickness_mm": thickness,
            "timing10_ms": timing10,
            "timing9_ms": timing9,
        }
        groups.append(group)
        for weight_idx, record in enumerate(records):
            observations.append(
                {
                    "group_idx": group_idx,
                    "weight_idx": weight_idx,
                    "stack_idx": STACK_INDEX[stack_key],
                    "stack_name": stack_key,
                    "sop_instance_uid": record.sop_instance_uid,
                    "dicom_path": str(record.path),
                    "acquisition_time_ms": float(preprocessed["acq_time_ms"][weight_idx, group_idx]),
                    "timing9_ms": timing9,
                    "affine_lps_rc": cropped_affine,
                    "pixel_spacing_rc_mm": pixel_spacing_rc,
                    "slice_thickness_mm": thickness,
                }
            )
    return {
        "coordinate_system": "DICOM_LPS_mm; affine_lps_rc maps [row,col,slice,1]",
        "stack_name": stack_key,
        "stack_idx": STACK_INDEX[stack_key],
        "groups": groups,
        "observations": observations,
    }
