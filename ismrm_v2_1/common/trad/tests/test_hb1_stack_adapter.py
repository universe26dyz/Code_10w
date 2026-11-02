import numpy as np

from modules.module_06_rigid_psf.hb1_stack_adapter import (
    HB1_GEOMETRY_TOLERANCE,
    audit_hb1_stack,
    build_hb1_registration_stack,
    round_trip_stack_geometry,
)


def _prepared(path):
    groups = np.array([9, 1, 5])
    images = np.zeros((30, 3, 5), dtype=np.float32)
    masks = np.zeros_like(images, dtype=bool)
    weight_idx = np.tile(np.arange(10), 3)
    group_idx = np.repeat(groups, 10)
    affine = np.repeat(np.eye(4, dtype=np.float64)[None], 30, axis=0)
    affine[:, :3, 0] = [0.0, 2.0, 0.0]
    affine[:, :3, 1] = [1.25, 0.0, 0.0]
    affine[:, :3, 2] = [0.0, 0.0, 8.0]
    for observation, group in enumerate(group_idx):
        affine[observation, :3, 3] = [10.0, -3.0, {9: 16.0, 1: 0.0, 5: 8.0}[int(group)]]
        images[observation].fill(observation)
        masks[observation, 1:, 1:-1] = True
    np.savez_compressed(path, images=images, masks=masks, group_idx=group_idx, weight_idx=weight_idx,
                        stack_idx=np.zeros(30, dtype=np.int64), acquisition_time_ms=np.zeros(30),
                        timing9_ms=np.zeros((30, 9)), affine_lps_rc=affine,
                        pixel_spacing_rc_mm=np.repeat([[2.0, 1.25]], 30, axis=0),
                        slice_thickness_mm=np.full(30, 8.0), tr_ms=np.full(30, 2.61), vps=np.full(30, 87))


def test_hb1_adapter_orders_dense_groups_losslessly_and_round_trips_geometry(tmp_path):
    path = tmp_path / "observations.npz"
    _prepared(path)
    audit = audit_hb1_stack(path, tolerance=HB1_GEOMETRY_TOLERANCE)
    assert audit.sorted_group_ids.tolist() == [1, 5, 9]
    assert audit.regular_gap and audit.gap_mm == 8.0
    adapted = build_hb1_registration_stack(path, audit)
    assert adapted.stack.slices.shape == (3, 1, 3, 5)
    assert adapted.stack.slices[:, 0, 0, 0].tolist() == [10.0, 20.0, 0.0]
    assert adapted.stack.mask[:, 0].equal(adapted.masks)
    result = round_trip_stack_geometry(adapted)
    assert result.max_error_mm <= HB1_GEOMETRY_TOLERANCE.world_mm
