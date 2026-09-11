from types import SimpleNamespace

import numpy as np

from modules.module_02_data_bridge.geometry import apply_affine_rc, crop_affine_lps_rc, dicom_lps_affine_rc


def test_crop_top_left_world_coordinate_equals_full_image_crop_coordinate():
    dicom = SimpleNamespace(
        ImagePositionPatient=[10.0, 20.0, 30.0],
        ImageOrientationPatient=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        PixelSpacing=[2.0, 3.0],
    )
    full_affine = dicom_lps_affine_rc(dicom, slice_thickness_mm=8.0)
    cropped_affine = crop_affine_lps_rc(full_affine, row_start_zero_based=4, col_start_zero_based=5)
    full_crop_top_left = apply_affine_rc(full_affine, np.array([4]), np.array([5]))[0]
    cropped_origin = apply_affine_rc(cropped_affine, np.array([0]), np.array([0]))[0]
    np.testing.assert_allclose(full_crop_top_left, cropped_origin)
    np.testing.assert_allclose(cropped_origin, [25.0, 28.0, 30.0])
