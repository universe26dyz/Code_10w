import numpy as np

from modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset


def _valid_affine() -> np.ndarray:
    # DICOM affine columns are [row=e_y, col=e_x, slice=e_z].
    affine = np.eye(4, dtype=np.float64)
    affine[:3, 0] = [0.0, 2.0, 0.0]
    affine[:3, 1] = [1.5, 0.0, 0.0]
    affine[:3, 2] = [0.0, 0.0, 8.0]
    affine[:3, 3] = [10.0, 20.0, 30.0]
    return affine


def test_quantitative_dataset_keeps_ten_weights_under_one_group_pose(tmp_path):
    path = tmp_path / "observations.npz"
    np.savez_compressed(
        path,
        images=np.ones((10, 2, 2), dtype=np.float32),
        masks=np.ones((10, 2, 2), dtype=bool),
        group_idx=np.zeros(10, dtype=np.int64),
        weight_idx=np.arange(10, dtype=np.int64),
        stack_idx=np.zeros(10, dtype=np.int64),
        acquisition_time_ms=np.arange(10, dtype=np.float64),
        timing9_ms=np.ones((10, 9), dtype=np.float64),
        affine_lps_rc=np.repeat(_valid_affine()[None, :, :], 10, axis=0),
        pixel_spacing_rc_mm=np.repeat(np.array([[2.0, 1.5]]), 10, axis=0),
        slice_thickness_mm=np.full(10, 8.0),
        tr_ms=np.full(10, 3.2), vps=np.full(10, 32, dtype=np.int64),
    )
    dataset = QuantPointDataset([path])
    assert dataset.group_pose_count == 1
    assert dataset.group_resolution_xyz_mm.tolist() == [[1.5, 2.0, 8.0]]
    assert dataset.xyz.shape == (40, 3)
    batch = dataset.get_batch(8)
    assert set(batch) == {"xyz", "v", "group_idx", "weight_idx", "stack_idx", "timing", "acquisition_time_ms"}
    assert batch["timing"].shape == (8, 9)
    assert batch["group_idx"].unique().tolist() == [0]
