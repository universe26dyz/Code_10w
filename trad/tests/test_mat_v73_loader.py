import h5py
import numpy as np

from modules.module_02_data_bridge.mat_v73 import load_preprocessed_v73


def _write_matlab_numeric(handle, name, array):
    array = np.asarray(array)
    stored = array.transpose(tuple(range(array.ndim - 1, -1, -1))) if array.ndim > 1 else array
    handle.create_dataset(name, data=stored)


def _ascii_columns(values):
    width = max(len(value) for value in values)
    out = np.zeros((width, len(values)), dtype=np.uint8)
    for index, value in enumerate(values):
        out[: len(value), index] = np.frombuffer(value.encode("ascii"), dtype=np.uint8)
    return out, np.asarray([len(value) for value in values], dtype=np.int64)


def test_v73_loader_restores_matlab_numeric_dimension_order(tmp_path):
    path = tmp_path / "case.mat"
    # Match real MATLAB one-group storage: trailing singleton Nslice is elided.
    mag = np.arange(2 * 3 * 10, dtype=np.float32).reshape(2, 3, 10)
    mag_crop = np.arange(1 * 2 * 10, dtype=np.float32).reshape(1, 2, 10)
    uids = [f"1.2.840.{index}" for index in range(10)]
    uid_ascii, uid_lengths = _ascii_columns(uids)
    series_ascii, series_lengths = _ascii_columns(["1.2.840.series"] * 10)
    with h5py.File(path, "w") as handle:
        for name, value in {
            "Mag": mag,
            "Mag_crop": mag_crop,
            "Acq_time": np.arange(10, dtype=np.float64).reshape(10, 1),
            "TR": np.array([[4.0]]),
            "VPS": np.array([[5.0]]),
            "FA": np.array([[45.0, 45.0, 45.0]]),
            "TI": np.array([[50.0, 150.0]]),
            "T2_prep": np.array([[35.0, 45.0, 55.0]]),
            "crop_row_start_zero_based": np.array([[63]]),
            "crop_col_start_zero_based": np.array([[63]]),
            "crop_size_rows_cols": np.array([[1, 2]]),
            "sop_instance_uid_ascii": uid_ascii,
            "sop_instance_uid_lengths": uid_lengths.reshape(-1, 1),
            "series_instance_uid_ascii": series_ascii,
            "series_instance_uid_lengths": series_lengths.reshape(-1, 1),
            "HB1_sop_instance_uid_ascii": uid_ascii[:, :1],
            "HB1_sop_instance_uid_lengths": uid_lengths[:1].reshape(-1, 1),
        }.items():
            _write_matlab_numeric(handle, name, value)
    loaded = load_preprocessed_v73(path)
    assert loaded["mag"].shape == (2, 3, 10, 1)
    assert loaded["mag_crop"].shape == (1, 2, 10, 1)
    assert loaded["acq_time_ms"].shape == (10, 1)
    np.testing.assert_array_equal(loaded["mag"], mag[..., None])
    assert loaded["sop_instance_uids"] == uids
