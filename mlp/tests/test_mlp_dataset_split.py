import json

import h5py
import numpy as np

from modules.module_05_signal_decoder.synthetic_dataset import generate_mlp_dataset


def test_generated_h5_splits_are_disjoint_by_timing_id(tmp_path):
    pool = tmp_path / "pool.npz"
    np.savez_compressed(pool, timing9_ms=np.asarray([[60] * 9, [70] * 9, [80] * 9], dtype=np.float64), tr_ms=np.asarray(3.2), vps=np.asarray(32), source_id=np.asarray(["a", "b", "c"]), group_id=np.asarray([0, 1, 2], dtype=np.int64), stack_idx=np.asarray([0, 0, 0], dtype=np.int64), functional_fixture=np.asarray(True))
    result = generate_mlp_dataset(pool, "configs/protocol_hhz_v1.yaml", tmp_path / "h5", {"train": 8, "valid": 4, "test": 4, "split": {"mode": "timing"}}, seed=3, chunk_size=4)
    sets = []
    for split in ("train", "valid", "test"):
        with h5py.File(tmp_path / "h5" / f"{split}.h5", "r") as handle:
            assert handle["input12"].shape[1] == 12 and handle["target_signal10"].shape[1] == 10
            sets.append(set(handle["timing_id"][:].tolist()))
    assert not sets[0].intersection(sets[1]) and not sets[0].intersection(sets[2]) and not sets[1].intersection(sets[2])
    assert set(result["timing_split_ids"]) == {"train", "valid", "test"}
