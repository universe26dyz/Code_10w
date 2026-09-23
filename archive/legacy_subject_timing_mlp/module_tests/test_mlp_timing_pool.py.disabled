import numpy as np
import pytest

from modules.module_05_signal_decoder.build_timing_pool import build_timing_pool


def _write(path, timing, tr=3.2, vps=32):
    n = 10 * len(timing)
    np.savez_compressed(path, group_idx=np.repeat(np.arange(len(timing)), 10), weight_idx=np.tile(np.arange(10), len(timing)), stack_idx=np.zeros(n, dtype=np.int64), timing9_ms=np.repeat(np.asarray(timing, dtype=np.float64), 10, axis=0), tr_ms=np.full(n, tr), vps=np.full(n, vps, dtype=np.int64))


def test_timing_pool_keeps_unique_groups_and_protocol_provenance(tmp_path):
    source, output = tmp_path / "obs.npz", tmp_path / "pool.npz"
    _write(source, [[60] * 9, [70] * 9, [60] * 9])
    result = build_timing_pool([source], output)
    assert result["timing_count"] == 2 and result["tr_ms"] == 3.2 and result["vps"] == 32
    with np.load(output, allow_pickle=False) as pool: assert pool["timing9_ms"].shape == (2, 9)


def test_timing_pool_rejects_protocol_mismatch_without_selecting_one(tmp_path):
    one, two = tmp_path / "one.npz", tmp_path / "two.npz"
    _write(one, [[60] * 9]); _write(two, [[70] * 9], tr=3.3)
    with pytest.raises(ValueError, match="TR=.*VPS"):
        build_timing_pool([one, two], tmp_path / "pool.npz")
