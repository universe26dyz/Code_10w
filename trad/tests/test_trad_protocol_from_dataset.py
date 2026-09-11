import numpy as np
import pytest

from modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset
from modules.module_07_objective_training.trad_trainer import build_protocol_from_dataset


def _write(path, tr_values, vps_values):
    affine = np.eye(4); affine[:3, 0] = [0, 2, 0]; affine[:3, 1] = [1, 0, 0]; affine[:3, 2] = [0, 0, 6]
    np.savez_compressed(path, images=np.ones((10, 2, 2), dtype=np.float32), masks=np.ones((10, 2, 2), dtype=bool), group_idx=np.zeros(10, dtype=np.int64), weight_idx=np.arange(10, dtype=np.int64), stack_idx=np.zeros(10, dtype=np.int64), acquisition_time_ms=np.arange(10, dtype=np.float64), timing9_ms=np.ones((10, 9)), affine_lps_rc=np.repeat(affine[None], 10, axis=0), pixel_spacing_rc_mm=np.repeat([[2.0, 1.0]], 10, axis=0), slice_thickness_mm=np.full(10, 6.0), tr_ms=np.asarray(tr_values), vps=np.asarray(vps_values))


def test_protocol_uses_validated_dataset_tr_vps_and_hhz_yaml(tmp_path):
    path = tmp_path / "observations.npz"; _write(path, [3.2] * 10, [32] * 10)
    protocol = build_protocol_from_dataset(QuantPointDataset([path]), "configs/protocol_hhz_v1.yaml")
    assert protocol.tr_ms == 3.2 and protocol.vps == 32 and protocol.ti_ms == (50.0, 150.0)


def test_protocol_rejects_inconsistent_group_tr_vps(tmp_path):
    first, second = tmp_path / "one.npz", tmp_path / "two.npz"
    _write(first, [3.2] * 10, [32] * 10); _write(second, [3.3] * 10, [32] * 10)
    with pytest.raises(ValueError, match="group=0.*group=1"):
        build_protocol_from_dataset(QuantPointDataset([first, second]), "configs/protocol_hhz_v1.yaml")
