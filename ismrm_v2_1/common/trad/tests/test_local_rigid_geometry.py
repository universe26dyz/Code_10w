import numpy as np
import torch

from modules.module_02_data_bridge.geometry import (
    apply_affine_rc,
    cropped_affine_lps_rc_to_initial_rigid,
    lps_to_ras,
)
from third_party.nesvor.nesvor.transform import ax_transform_points


def test_centered_local_points_map_to_cropped_dicom_lps_then_ras_for_oblique_plane():
    row = np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)
    col = np.array([1.0, 0.0, 0.0])
    normal = np.cross(col, row)
    affine = np.eye(4)
    affine[:3, 0] = row * 2.0
    affine[:3, 1] = col * 1.25
    affine[:3, 2] = normal * 7.0
    affine[:3, 3] = [14.0, -32.0, 9.0]
    h, w = 5, 7
    rigid = cropped_affine_lps_rc_to_initial_rigid(affine, (h, w), np.array([1.25, 2.0, 7.0]))
    rows = np.array([0.0, (h - 1) / 2.0, h - 1.0])
    cols = np.array([0.0, (w - 1) / 2.0, w - 1.0])
    local = torch.tensor(
        np.column_stack(((cols - (w - 1) / 2.0) * 1.25, (rows - (h - 1) / 2.0) * 2.0, np.zeros(3))),
        dtype=torch.float32,
    )
    observed = ax_transform_points(rigid.axisangle(), local, trans_first=True).detach().numpy()
    expected = lps_to_ras(apply_affine_rc(affine, rows, cols))
    np.testing.assert_allclose(observed, expected, atol=2e-5)


def test_lps_to_ras_is_explicit_sign_conversion():
    np.testing.assert_array_equal(lps_to_ras(np.array([[4.0, -6.0, 8.0]])), np.array([[-4.0, 6.0, 8.0]]))
