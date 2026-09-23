import numpy as np
import torch

from modules.module_05_signal_decoder.mlp_model import MdmSignalMLP
from modules.module_09_qc_benchmark.benchmark_signal_decoder import benchmark_signal_decoder
from online_helpers import write_functional_checkpoint


def test_benchmark_writes_separate_method_peak_fields_and_speedups(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"; write_functional_checkpoint(checkpoint)
    data = torch.load(checkpoint, weights_only=False); data.update({"functional_fixture": False, "scientific_checkpoint": True, "formal_candidate": True, "validation_status": "unvalidated"}); torch.save(data, checkpoint)
    pool = tmp_path / "pool.npz"
    np.savez_compressed(pool, timing9_ms=np.asarray([[60.0] * 9, [70.0] * 9, [80.0] * 9]), tr_ms=np.asarray(3.2), vps=np.asarray(32), source_id=np.asarray(["a", "b", "c"]), group_id=np.arange(3), stack_idx=np.zeros(3, dtype=np.int64), subject_id=np.asarray(["S1", "S2", "S3"]), functional_fixture=np.asarray(False))
    rows = benchmark_signal_decoder(checkpoint, pool, "configs/protocol_hhz_v1.yaml", tmp_path / "benchmark.json", "cpu", [2], repetitions=1)
    assert set(("trad_peak_gpu_bytes", "mlp_peak_gpu_bytes", "forward_speedup", "backward_speedup")).issubset(rows[0])
