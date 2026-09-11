import torch
import pytest

from modules.module_05_signal_decoder.checkpoint_loader import load_frozen_mlp_decoder
from online_helpers import write_functional_checkpoint, write_observations


def test_unvalidated_formal_candidate_is_rejected_by_online_loader(tmp_path):
    dataset = write_observations(tmp_path / "observations.npz")
    checkpoint = tmp_path / "candidate.pth"; write_functional_checkpoint(checkpoint)
    data = torch.load(checkpoint, weights_only=False)
    data.update({"functional_fixture": False, "scientific_checkpoint": True, "formal_candidate": True, "validation_status": "unvalidated"})
    torch.save(data, checkpoint)
    with pytest.raises(ValueError, match="validation_status=.*awaiting_manual_review"):
        load_frozen_mlp_decoder(checkpoint, dataset, "configs/protocol_hhz_v1.yaml", allow_functional_fixture=False, device=torch.device("cpu"))
