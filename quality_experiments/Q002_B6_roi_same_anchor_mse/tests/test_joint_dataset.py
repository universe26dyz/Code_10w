from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from trad.modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset


def _write_observations(path: Path, *, masks: np.ndarray | None = None, timing_delta: bool = False, affine_delta: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    images = np.stack([np.full((2, 3), weight + 1, np.float32) for weight in range(10)])
    if masks is None:
        masks = np.ones_like(images, dtype=bool)
    affine = np.repeat(np.eye(4, dtype=np.float64)[None], 10, axis=0)
    affine[:, :3, 0] = [0.0, 1.0, 0.0]
    affine[:, :3, 1] = [2.0, 0.0, 0.0]
    affine[:, :3, 2] = [0.0, 0.0, 5.0]
    if affine_delta:
        affine[9, 0, 3] = 1.0
    timing = np.repeat(np.arange(9, dtype=np.float64)[None], 10, axis=0)
    if timing_delta:
        timing[9, 0] += 1.0
    np.savez_compressed(
        path,
        images=images,
        masks=masks,
        group_idx=np.zeros(10, dtype=np.int64),
        weight_idx=np.arange(10, dtype=np.int64),
        stack_idx=np.zeros(10, dtype=np.int64),
        acquisition_time_ms=np.zeros(10, dtype=np.float64),
        timing9_ms=timing,
        affine_lps_rc=affine,
        pixel_spacing_rc_mm=np.repeat([[1.0, 2.0]], 10, axis=0),
        slice_thickness_mm=np.full(10, 5.0),
        tr_ms=np.full(10, 4.0),
        vps=np.full(10, 5, dtype=np.int64),
    )


def _dataset(path: Path) -> QuantPointDataset:
    return QuantPointDataset([path], device="cpu")


class _ScalarMetadataOnly:
    group_pose_count = 1

    @staticmethod
    def validate_balanced_samples():
        return {0: 1.0}


def test_joint_anchors_use_integer_pixel_identity_ordered_weights_and_and_mask(tmp_path):
    from quality_experiments.Q002_B6_roi_same_anchor_mse.joint_dataset import JointAnchorDataset

    masks = np.ones((10, 2, 3), bool)
    for weight in range(10):
        masks[weight, 0, 1 if weight % 2 == 0 else 2] = False
    path = tmp_path / "prepared" / "CYJ" / "sax" / "observations.npz"
    _write_observations(path, masks=masks)
    joint = JointAnchorDataset([path], _dataset(path))

    assert joint.observed.shape == (4, 10)
    assert joint.row_col.tolist() == [[0, 0], [1, 0], [1, 1], [1, 2]]
    assert torch.equal(joint.group_idx, torch.zeros(4, dtype=torch.long))
    assert torch.equal(joint.observed[0], torch.arange(1, 11, dtype=torch.float32))
    assert joint.xyz.shape == (4, 3)
    assert joint.qc["groups"]["0"]["joint_valid_count"] == 4
    assert joint.qc["groups"]["0"]["joint_support_fraction_of_union"] == pytest.approx(4 / 6)
    assert joint.qc["scalar_valid_pixels_per_weight"]["3"] == 5


@pytest.mark.parametrize(("kwargs", "message"), [({"timing_delta": True}, "timing"), ({"affine_delta": True}, "affine")])
def test_joint_anchors_reject_group_metadata_mismatch(tmp_path, kwargs, message):
    from quality_experiments.Q002_B6_roi_same_anchor_mse.joint_dataset import JointAnchorDataset

    path = tmp_path / "prepared" / "CYJ" / "sax" / "observations.npz"
    _write_observations(path, **kwargs)
    with pytest.raises(ValueError, match=message):
        JointAnchorDataset([path], _ScalarMetadataOnly())


def test_joint_anchors_reject_empty_intersection_without_zero_filling(tmp_path):
    from quality_experiments.Q002_B6_roi_same_anchor_mse.joint_dataset import JointAnchorDataset

    masks = np.ones((10, 2, 3), bool)
    masks[0] = False
    path = tmp_path / "prepared" / "CYJ" / "sax" / "observations.npz"
    _write_observations(path, masks=masks)
    with pytest.raises(ValueError, match="no joint-valid"):
        JointAnchorDataset([path], _ScalarMetadataOnly())


def test_cropped_path_contract_rejects_full_fov_and_route_forbids_warm_start(tmp_path):
    from quality_experiments.Q002_B6_roi_same_anchor_mse.contracts import (
        Q002_EXPERIMENT_ID,
        build_q002_route_config,
        cropped_observation_paths,
    )

    root = tmp_path / "Code_10w_prepared"
    for stack in ("sax", "2ch", "4ch"):
        _write_observations(root / "CYJ" / stack / "observations.npz")
    assert [path.parent.name for path in cropped_observation_paths(root)] == ["sax", "2ch", "4ch"]
    assert Q002_EXPERIMENT_ID == "Q002_B6_roi_same_anchor_mse"
    with pytest.raises(ValueError, match="full-FOV"):
        cropped_observation_paths(tmp_path / "Q001_full_fov_prepared")
    config = {"training": {"seed": 20260911}, "decoder": {"checkpoint": "/approved.pth"}}
    route = build_q002_route_config(config, cropped_prepared_root=root, reconstruction_checkpoint=None)
    assert route["route"]["training_prepared_root"] == str(root)
    with pytest.raises(ValueError, match="warm-start"):
        build_q002_route_config(config, cropped_prepared_root=root, reconstruction_checkpoint="/b6/model.pt")


def test_fixed_monitor_is_seed_stable_and_records_integer_anchor_hash(tmp_path):
    from quality_experiments.Q002_B6_roi_same_anchor_mse.joint_dataset import JointAnchorDataset

    path = tmp_path / "prepared" / "CYJ" / "sax" / "observations.npz"
    _write_observations(path)
    joint = JointAnchorDataset([path], _dataset(path))
    first = joint.fixed_monitor(seed=20260911, samples_per_stack=8)
    second = joint.fixed_monitor(seed=20260911, samples_per_stack=8)
    assert torch.equal(first["row_col"], second["row_col"])
    assert first["anchor_identity_sha256"] == second["anchor_identity_sha256"]
    identities = np.column_stack((first["group_idx"].numpy(), first["row_col"].numpy()))
    assert first["anchor_identity_sha256"] == hashlib.sha256(identities.astype("<i8", copy=False).tobytes()).hexdigest()
