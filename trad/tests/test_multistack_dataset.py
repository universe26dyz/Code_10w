import numpy as np

from modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset


def _write_stack(path, stack_idx, origin):
    affine = np.eye(4, dtype=np.float64)
    affine[:3, 0] = [0.0, 2.0, 0.0]
    affine[:3, 1] = [1.0, 0.0, 0.0]
    affine[:3, 2] = [0.0, 0.0, 6.0]
    affine[:3, 3] = origin
    np.savez_compressed(
        path,
        images=np.ones((10, 3, 3), dtype=np.float32),
        masks=np.ones((10, 3, 3), dtype=bool),
        group_idx=np.zeros(10, dtype=np.int64),
        weight_idx=np.arange(10, dtype=np.int64),
        stack_idx=np.full(10, stack_idx, dtype=np.int64),
        acquisition_time_ms=np.arange(10, dtype=np.float64),
        timing9_ms=np.ones((10, 9), dtype=np.float64),
        affine_lps_rc=np.repeat(affine[None], 10, axis=0),
        pixel_spacing_rc_mm=np.repeat(np.array([[2.0, 1.0]]), 10, axis=0),
        slice_thickness_mm=np.full(10, 6.0),
        tr_ms=np.full(10, 3.2),
        vps=np.full(10, 32, dtype=np.int64),
    )


def test_multistack_reindexes_groups_and_preserves_stack_ids(tmp_path):
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    _write_stack(first, 2, [0.0, 0.0, 0.0])
    _write_stack(second, 7, [100.0, 50.0, 20.0])
    dataset = QuantPointDataset([first, second])
    assert dataset.group_pose_count == 2
    assert dataset.group_idx.unique().tolist() == [0, 1]
    assert dataset.stack_idx.unique().tolist() == [2, 7]
    assert dataset.group_resolution_xyz_mm.tolist() == [[1.0, 2.0, 6.0], [1.0, 2.0, 6.0]]
    assert dataset.bounding_box.shape == (2, 3)
    assert float(dataset.bounding_box[1, 0] - dataset.bounding_box[0, 0]) > 90.0
    bare = dataset.xyz_transformed
    margin = 2 * dataset.group_resolution_xyz_mm.max()
    torch_min = bare.amin(0) - margin
    torch_max = bare.amax(0) + margin
    import torch
    torch.testing.assert_close(dataset.bounding_box[0], torch_min)
    torch.testing.assert_close(dataset.bounding_box[1], torch_max)
