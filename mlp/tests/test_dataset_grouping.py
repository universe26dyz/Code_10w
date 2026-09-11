import numpy as np

from modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset


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
        affine_lps_rc=np.repeat(np.eye(4, dtype=np.float64)[None, :, :], 10, axis=0),
    )
    dataset = QuantPointDataset(path)
    assert dataset.group_pose_count == 1
    assert dataset.xyz.shape == (40, 3)
    batch = dataset.get_batch(8)
    assert set(batch) == {"xyz", "v", "group_idx", "weight_idx", "stack_idx", "timing", "acquisition_time_ms"}
    assert batch["timing"].shape == (8, 9)
    assert batch["group_idx"].unique().tolist() == [0]
