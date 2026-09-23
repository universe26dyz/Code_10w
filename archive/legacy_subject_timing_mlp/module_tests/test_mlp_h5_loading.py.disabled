import h5py
import numpy as np
import torch

from modules.module_05_signal_decoder.train_mlp import load_h5_split


def test_h5_split_is_opened_once_and_loaded_to_cpu_tensor_dataset(tmp_path, monkeypatch):
    path = tmp_path / "train.h5"
    with h5py.File(path, "w") as handle:
        handle["input12"] = np.arange(48, dtype=np.float32).reshape(4, 12)
        handle["target_signal10"] = np.arange(40, dtype=np.float32).reshape(4, 10)
        handle["timing_id"] = np.zeros(4, dtype=np.int64)
    real_file, opens = h5py.File, 0

    def counted_file(*args, **kwargs):
        nonlocal opens
        opens += 1
        return real_file(*args, **kwargs)

    monkeypatch.setattr("modules.module_05_signal_decoder.train_mlp.h5py.File", counted_file)
    dataset = load_h5_split(path, "train")
    assert opens == 1
    inputs, targets = dataset.tensors
    assert inputs.device.type == "cpu" and targets.device.type == "cpu"
    assert inputs.dtype == torch.float32 and targets.dtype == torch.float32
    assert inputs.shape == (4, 12) and targets.shape == (4, 10)
