import hashlib
import json

import h5py
import numpy as np
import pytest

from modules.module_05_signal_decoder.train_mlp import validate_dataset_provenance


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dataset_metadata_rejects_different_timing_pool(tmp_path):
    pool_a, pool_b = tmp_path / "a.npz", tmp_path / "b.npz"
    for path, timing in ((pool_a, 70.0), (pool_b, 71.0)):
        np.savez_compressed(path, timing9_ms=np.full((3, 9), timing), tr_ms=np.asarray(3.2), vps=np.asarray(32), source_id=np.asarray(["a", "b", "c"]), group_id=np.arange(3), stack_idx=np.zeros(3, dtype=np.int64), functional_fixture=np.asarray(False))
    dataset = tmp_path / "dataset"; dataset.mkdir()
    for split, n in (("train", 4), ("valid", 2), ("test", 2)):
        with h5py.File(dataset / f"{split}.h5", "w") as handle:
            handle["input12"] = np.zeros((n, 12), dtype=np.float32)
            handle["target_signal10"] = np.zeros((n, 10), dtype=np.float32)
            handle["timing_id"] = np.zeros(n, dtype=np.int64)
    metadata = {"timing_pool_sha256": _sha256(pool_a), "functional_fixture": False, "timing9_min_ms": [70.0] * 9, "timing9_max_ms": [70.0] * 9, "tr_ms": 3.2, "vps": 32, "protocol": {"tr_ms": 3.2, "vps": 32, "fa_deg": [45.0, 45.0, 45.0], "ti_ms": [50.0, 150.0], "t2prep_ms": [35.0, 45.0, 55.0], "n_ramp_up": 10}, "sizes": {"train": 4, "valid": 2, "test": 2}}
    (dataset / "dataset_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="timing-pool SHA256"):
        validate_dataset_provenance(dataset, pool_b, "configs/protocol_hhz_v1.yaml")
