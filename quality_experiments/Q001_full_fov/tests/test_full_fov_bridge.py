"""Real HDF5-MAT and shuffled-DICOM geometry contracts for Q001 full FOV."""

from __future__ import annotations

import h5py
import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

from trad.modules.module_02_data_bridge.geometry import apply_affine_rc, crop_affine_lps_rc, dicom_lps_affine_rc
from trad.modules.module_02_data_bridge.prepare_observations import prepare_observations

from quality_experiments.Q001_full_fov.bridge.full_fov_bridge import load_full_fov_preprocessed, prepare_full_fov_observations


def _matlab_numeric(handle, name, value):
    value = np.asarray(value)
    handle.create_dataset(name, data=value.transpose(tuple(range(value.ndim - 1, -1, -1))) if value.ndim > 1 else value)


def _ascii(values):
    width = max(len(value) for value in values)
    result = np.zeros((width, len(values)), dtype=np.uint8)
    for index, value in enumerate(values): result[:len(value), index] = np.frombuffer(value.encode("ascii"), dtype=np.uint8)
    return result, np.asarray([len(value) for value in values], dtype=np.int64)


def _dicom(path, uid, acquisition_time):
    meta = FileMetaDataset(); meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage; meta.MediaStorageSOPInstanceUID = uid; meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = SecondaryCaptureImageStorage; ds.SOPInstanceUID = uid; ds.SeriesInstanceUID = "1.2.826.0.1.3680043.10.999.77"
    ds.Rows = 5; ds.Columns = 7; ds.PixelSpacing = [2.0, 3.0]; ds.SliceThickness = 8.0
    ds.ImagePositionPatient = [10.0, 20.0, 30.0]; ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.AcquisitionTime = acquisition_time; ds.RepetitionTime = 3.2; ds.EchoTrainLength = 32
    ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = "MONOCHROME2"; ds.BitsAllocated = 16; ds.BitsStored = 16; ds.HighBit = 15; ds.PixelRepresentation = 0
    ds.PixelData = np.zeros((5, 7), dtype=np.uint16).tobytes(); pydicom.dcmwrite(path, ds, write_like_original=False)


def _dicoms(root):
    root.mkdir(); uids = [f"1.2.826.0.1.3680043.10.999.{index}" for index in range(10)]
    for index, uid in enumerate(uids): _dicom(root / f"shuffled_{9 - index:02d}.dcm", uid, f"1200{index:02d}.000")
    return uids


def _mat(path, uids, *, full):
    mag = np.arange(5 * 7 * 10, dtype=np.float32).reshape(5, 7, 10, 1)
    mag_crop = mag if full else mag[1:4, 2:5]
    uid_ascii, uid_lengths = _ascii(uids); series_ascii, series_lengths = _ascii(["1.2.826.0.1.3680043.10.999.77"] * 10)
    values = {"Mag": mag, "Mag_crop": mag_crop, "Acq_time": np.arange(10, dtype=np.float64).reshape(10, 1), "TR": np.array([[3.2]]), "VPS": np.array([[32]]), "FA": np.array([[45., 45., 45.]]), "TI": np.array([[50., 150.]]), "T2_prep": np.array([[35., 45., 55.]]), "crop_row_start_zero_based": np.array([[0 if full else 1]]), "crop_col_start_zero_based": np.array([[0 if full else 2]]), "crop_size_rows_cols": np.asarray([mag_crop.shape[:2]]), "sop_instance_uid_ascii": uid_ascii, "sop_instance_uid_lengths": uid_lengths.reshape(-1, 1), "series_instance_uid_ascii": series_ascii, "series_instance_uid_lengths": series_lengths.reshape(-1, 1), "HB1_sop_instance_uid_ascii": uid_ascii[:, :1], "HB1_sop_instance_uid_lengths": uid_lengths[:1].reshape(-1, 1)}
    with h5py.File(path, "w") as handle:
        for name, value in values.items(): _matlab_numeric(handle, name, value)
        if full:
            _matlab_numeric(handle, "spatial_mode_ascii", np.frombuffer(b"full_fov", dtype=np.uint8))
            _matlab_numeric(handle, "final_data_semantics_ascii", np.frombuffer(b"MP-PCA(full-FOV MIND_mag_reg)", dtype=np.uint8))
            _matlab_numeric(handle, "geometry_rule_ascii", np.frombuffer(b"After full-FOV MIND, all 10 weights in each group use HB1 geometry.", dtype=np.uint8))
            _matlab_numeric(handle, "full_fov_shape_rows_cols", np.asarray([[5, 7]], dtype=np.int64))


def test_full_fov_bridge_preserves_full_matrix_and_dicom_affine_with_shuffled_files(tmp_path):
    dicoms = tmp_path / "dicoms"; uids = _dicoms(dicoms); mat = tmp_path / "full.mat"; _mat(mat, uids, full=True)
    output = tmp_path / "prepared"

    result = prepare_full_fov_observations(mat, dicoms, "sax", output)

    with np.load(output / "observations.npz") as data:
        assert data["images"].shape == (10, 5, 7)
        assert np.array_equal(data["weight_idx"], np.arange(10))
        full_affine = dicom_lps_affine_rc(pydicom.dcmread(next(dicoms.glob("*.dcm")), stop_before_pixels=True), 8.0)
        assert np.array_equal(data["affine_lps_rc"], np.repeat(full_affine[None], 10, axis=0))
        np.testing.assert_allclose(apply_affine_rc(data["affine_lps_rc"][0], np.array([2.]), np.array([3.])), apply_affine_rc(full_affine, np.array([2.]), np.array([3.])))
        np.testing.assert_allclose(apply_affine_rc(data["affine_lps_rc"][0], np.array([4.]), np.array([6.])), apply_affine_rc(full_affine, np.array([4.]), np.array([6.])))
    assert result["qc_summary"]["spatial_mode"] == "full_fov"
    assert result["qc_summary"]["crop_offsets_zero_based"] == [0, 0]
    assert result["manifest"]["final_data_semantics"] == "MP-PCA(full-FOV MIND_mag_reg)"


def test_full_fov_loader_rejects_cropped_contract_and_baseline_crop_translation_remains(tmp_path):
    dicoms = tmp_path / "dicoms"; uids = _dicoms(dicoms); cropped = tmp_path / "cropped.mat"; _mat(cropped, uids, full=False)

    with pytest.raises(ValueError, match="full_fov"):
        load_full_fov_preprocessed(cropped)
    prepare_observations(cropped, dicoms, "sax", tmp_path / "cropped_prepared")
    with np.load(tmp_path / "cropped_prepared" / "observations.npz") as data:
        full_affine = dicom_lps_affine_rc(pydicom.dcmread(next(dicoms.glob("*.dcm")), stop_before_pixels=True), 8.0)
        expected = crop_affine_lps_rc(full_affine, 1, 2)
        assert np.array_equal(data["affine_lps_rc"], np.repeat(expected[None], 10, axis=0))
        assert data["images"].shape == (10, 3, 3)


def test_cropped_vs_full_qc_compares_geometry_and_not_processed_pixel_equality(tmp_path):
    from quality_experiments.Q001_full_fov.bridge.cropped_vs_full_qc import compare_prepared

    dicoms = tmp_path / "dicoms"; uids = _dicoms(dicoms)
    full_mat, cropped_mat = tmp_path / "full.mat", tmp_path / "cropped.mat"; _mat(full_mat, uids, full=True); _mat(cropped_mat, uids, full=False)
    full_dir, cropped_dir = tmp_path / "full", tmp_path / "cropped"
    prepare_full_fov_observations(full_mat, dicoms, "sax", full_dir); prepare_observations(cropped_mat, dicoms, "sax", cropped_dir)

    report = compare_prepared(cropped_dir, full_dir, tmp_path / "qc.json")

    assert report["full_spatial_mode"] == "full_fov"
    assert report["full_shape_rows_cols"] == [5, 7]
    assert report["cropped_shape_rows_cols"] == [3, 3]
    assert report["timing_equal"] is True
    assert report["pixel_value_equality_required"] is False
    assert (tmp_path / "qc.json").is_file()
