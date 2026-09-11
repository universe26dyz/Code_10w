import pytest
import torch

from modules.module_05_signal_decoder.checkpoint_loader import load_frozen_mlp_decoder
from online_helpers import write_functional_checkpoint, write_observations


def test_checkpoint_loader_rejects_online_timing_outside_training_range(tmp_path):
    dataset = write_observations(tmp_path / "observations.npz")
    checkpoint = tmp_path / "checkpoint.pth"; write_functional_checkpoint(checkpoint, 60.0, 69.0)
    with pytest.raises(ValueError, match="outside the MLP training range.*timing2"):
        load_frozen_mlp_decoder(checkpoint, dataset, "configs/protocol_hhz_v1.yaml", allow_functional_fixture=True, device=torch.device("cpu"))


def test_checkpoint_loader_accepts_float32_rounding_at_a_stored_timing_boundary(tmp_path):
    dataset = write_observations(tmp_path / "observations.npz")
    dataset.timing.fill_(699.93)  # prepared observations are float32
    checkpoint = tmp_path / "checkpoint.pth"; write_functional_checkpoint(checkpoint, 699.9300000000001, 699.9300000000001)
    decoder = load_frozen_mlp_decoder(checkpoint, dataset, "configs/protocol_hhz_v1.yaml", allow_functional_fixture=True, device=torch.device("cpu"))
    assert decoder.protocol is not None
