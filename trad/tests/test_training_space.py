import numpy as np
import torch

from modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset
from modules.module_07_objective_training.training_space import TrainingSpace
from third_party.nesvor.nesvor.transform import RigidTransform


def _dataset(path):
    affine = np.eye(4, dtype=np.float64)
    affine[:3, 0] = [0.0, 2.0, 0.0]
    affine[:3, 1] = [1.0, 0.0, 0.0]
    affine[:3, 2] = [0.0, 0.0, 6.0]
    affine[:3, 3] = [12.0, -7.0, 30.0]
    np.savez_compressed(path, images=np.ones((10, 3, 3), dtype=np.float32), masks=np.ones((10, 3, 3), dtype=bool), group_idx=np.zeros(10, dtype=np.int64), weight_idx=np.arange(10, dtype=np.int64), stack_idx=np.zeros(10, dtype=np.int64), acquisition_time_ms=np.arange(10, dtype=np.float64), timing9_ms=np.ones((10, 9), dtype=np.float64), affine_lps_rc=np.repeat(affine[None], 10, axis=0), pixel_spacing_rc_mm=np.repeat(np.array([[2.0, 1.0]]), 10, axis=0), slice_thickness_mm=np.full(10, 6.0), tr_ms=np.full(10, 3.2), vps=np.full(10, 32, dtype=np.int64))
    return QuantPointDataset([path])


def test_nesvor_center_scaling_roundtrip_preserves_initial_physical_group_pose(tmp_path):
    dataset = _dataset(tmp_path / "observations.npz")
    space = TrainingSpace.from_dataset(dataset, spatial_scaling=30.0)
    recovered = space.axisangle_train_to_physical(space.group_axisangle_init_train)
    torch.testing.assert_close(
        RigidTransform(recovered).matrix(), RigidTransform(dataset.group_axisangle_init).matrix(), atol=2e-5, rtol=0
    )
    point = dataset.xyz[:1]
    torch.testing.assert_close(space.train_ras_to_physical(space.physical_ras_to_train(point)), point)
    assert torch.allclose(space.group_resolution_train * 30.0, dataset.group_resolution_xyz_mm)
