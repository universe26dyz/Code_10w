import h5py
import numpy as np
import torch

from modules.module_05_signal_decoder.synthetic_dataset import generate_mlp_dataset
from modules.module_05_signal_decoder.trad_teacher.trad_signal_simulator import TradProtocol, TradSignalSimulator


def test_h5_target_is_the_normalized_hhz_trad_teacher_signal(tmp_path):
    pool = tmp_path / "pool.npz"
    np.savez_compressed(pool, timing9_ms=np.asarray([[60] * 9, [70] * 9, [80] * 9], dtype=np.float64), tr_ms=np.asarray(3.2), vps=np.asarray(32), source_id=np.asarray(["a", "b", "c"]), group_id=np.asarray([0, 1, 2]), stack_idx=np.asarray([0, 0, 0]), functional_fixture=np.asarray(True))
    generate_mlp_dataset(pool, "configs/protocol_hhz_v1.yaml", tmp_path / "h5", {"train": 4, "valid": 2, "test": 2, "split": {"mode": "timing"}}, seed=4, chunk_size=2)
    with h5py.File(tmp_path / "h5" / "train.h5", "r") as handle: input12, target = torch.from_numpy(handle["input12"][:1]), torch.from_numpy(handle["target_signal10"][:1])
    expected = TradSignalSimulator()(input12[:, 0] * 1000, input12[:, 1] * 1000, input12[:, 2], input12[:, 3:] * 1000, TradProtocol(3.2, 32), normalize=True)
    torch.testing.assert_close(target, expected)
