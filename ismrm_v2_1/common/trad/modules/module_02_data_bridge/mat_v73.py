"""Explicit reader for the numeric subset written by MATLAB ``save -v7.3``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


REQUIRED_DATASETS = (
    "Mag",
    "Mag_crop",
    "Acq_time",
    "TR",
    "VPS",
    "FA",
    "TI",
    "T2_prep",
    "crop_row_start_zero_based",
    "crop_col_start_zero_based",
    "crop_size_rows_cols",
    "sop_instance_uid_ascii",
    "sop_instance_uid_lengths",
    "series_instance_uid_ascii",
    "series_instance_uid_lengths",
    "HB1_sop_instance_uid_ascii",
    "HB1_sop_instance_uid_lengths",
)


def _read_matlab_numeric(dataset: h5py.Dataset) -> np.ndarray:
    """Read numeric MATLAB v7.3 storage and restore MATLAB dimension order."""

    value = np.asarray(dataset)
    if value.ndim > 1:
        value = value.transpose(tuple(range(value.ndim - 1, -1, -1)))
    return value


def _scalar(value: np.ndarray, name: str) -> float:
    if value.size != 1:
        raise ValueError(f"MAT variable {name} must be scalar, got shape {value.shape}.")
    return float(value.reshape(-1)[0])


def decode_ascii_columns(ascii_matrix: np.ndarray, lengths: np.ndarray, name: str) -> list[str]:
    """Decode explicit MATLAB ``uint8[max_length, count]`` UID storage."""

    ascii_matrix = np.asarray(ascii_matrix, dtype=np.uint8)
    lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
    if ascii_matrix.ndim != 2 or ascii_matrix.shape[1] != lengths.size:
        raise ValueError(
            f"{name} ASCII matrix must be [max_length, count] with matching lengths; "
            f"got {ascii_matrix.shape} and {lengths.shape}."
        )
    decoded: list[str] = []
    for index, length in enumerate(lengths):
        if length <= 0 or length > ascii_matrix.shape[0]:
            raise ValueError(f"{name}[{index}] has invalid encoded length {length}.")
        try:
            decoded.append(bytes(ascii_matrix[:length, index]).decode("ascii"))
        except UnicodeDecodeError as exc:
            raise ValueError(f"{name}[{index}] is not ASCII.") from exc
    return decoded


def load_preprocessed_v73(mat_path: str | Path) -> dict[str, Any]:
    """Load one v1 preprocessing MAT through one fixed h5py implementation.

    No scipy loader or shape fallback is attempted: Phase 1 always writes
    ``-v7.3`` and this function asserts the exact bridge contract.
    """

    mat_path = Path(mat_path)
    if not mat_path.is_file():
        raise FileNotFoundError(f"Preprocessed MAT does not exist: {mat_path}")
    if not h5py.is_hdf5(mat_path):
        raise ValueError(f"Expected MATLAB -v7.3 HDF5 MAT, not HDF5: {mat_path}")
    with h5py.File(mat_path, "r") as handle:
        missing = [name for name in REQUIRED_DATASETS if name not in handle]
        if missing:
            raise ValueError(f"MAT lacks required Phase-2 datasets: {missing}")
        data = {name: _read_matlab_numeric(handle[name]) for name in REQUIRED_DATASETS}

    mag = data["Mag"]
    mag_crop = data["Mag_crop"]
    acq_time = data["Acq_time"]
    # MATLAB elides a trailing singleton on save: a one-group [Nx,Ny,10,1]
    # Phase-1 array is stored as 3-D. This is an explicit v7.3 convention,
    # not an alternate loader or inferred acquisition grouping.
    if mag.ndim == 3 and mag.shape[2] == 10:
        mag = mag[..., np.newaxis]
    if mag_crop.ndim == 3 and mag_crop.shape[2] == 10:
        mag_crop = mag_crop[..., np.newaxis]
    if acq_time.ndim == 1 and acq_time.shape == (10,):
        acq_time = acq_time[:, np.newaxis]
    if (
        mag.ndim != 4
        or mag_crop.ndim != 4
        or mag.shape[2:] != mag_crop.shape[2:]
    ):
        raise ValueError(
            "Mag and Mag_crop must both be 4-D and share [10, Nslice]; "
            f"got {mag.shape} and {mag_crop.shape}."
        )
    if mag.shape[2] != 10:
        raise ValueError(f"MAT Mag third dimension must be 10 weights, got {mag.shape}.")
    if acq_time.shape != (10, mag.shape[3]):
        raise ValueError(
            "Acq_time must restore to [10, Nslice]; "
            f"got {acq_time.shape} for Nslice={mag.shape[3]}."
        )
    if tuple(np.asarray(data["crop_size_rows_cols"]).reshape(-1).astype(int)) != mag_crop.shape[:2]:
        raise ValueError(
            "crop_size_rows_cols does not match cropped image dimensions: "
            f"{data['crop_size_rows_cols'].reshape(-1)} vs {mag_crop.shape[:2]}."
        )

    sop_uids = decode_ascii_columns(
        data["sop_instance_uid_ascii"], data["sop_instance_uid_lengths"], "SOPInstanceUID"
    )
    series_uids = decode_ascii_columns(
        data["series_instance_uid_ascii"],
        data["series_instance_uid_lengths"],
        "SeriesInstanceUID",
    )
    hb1_uids = decode_ascii_columns(
        data["HB1_sop_instance_uid_ascii"],
        data["HB1_sop_instance_uid_lengths"],
        "HB1 SOPInstanceUID",
    )
    expected_count = 10 * mag.shape[3]
    if len(sop_uids) != expected_count or len(series_uids) != expected_count:
        raise ValueError(
            "UID count must equal 10 * Nslice; "
            f"got SOP={len(sop_uids)}, series={len(series_uids)}, Nslice={mag.shape[3]}."
        )
    if hb1_uids != sop_uids[::10]:
        raise ValueError("HB1 SOPInstanceUID metadata does not equal every group’s weight 0 UID.")
    if len(set(sop_uids)) != len(sop_uids):
        raise ValueError("SOPInstanceUID metadata contains duplicates.")

    return {
        "mag": mag,
        "mag_crop": mag_crop,
        "acq_time_ms": np.asarray(acq_time, dtype=np.float64),
        "tr_ms": _scalar(data["TR"], "TR"),
        "vps": _scalar(data["VPS"], "VPS"),
        "fa_deg": np.asarray(data["FA"], dtype=np.float64).reshape(-1),
        "ti_ms": np.asarray(data["TI"], dtype=np.float64).reshape(-1),
        "t2prep_ms": np.asarray(data["T2_prep"], dtype=np.float64).reshape(-1),
        "crop_row_start_zero_based": int(_scalar(data["crop_row_start_zero_based"], "crop row")),
        "crop_col_start_zero_based": int(_scalar(data["crop_col_start_zero_based"], "crop col")),
        "sop_instance_uids": sop_uids,
        "series_instance_uids": series_uids,
        "hb1_sop_instance_uids": hb1_uids,
    }
