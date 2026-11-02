import copy

import pytest
import torch

from mlp.modules.module_05_signal_decoder.checkpoint_loader import load_frozen_mlp_decoder
from online_helpers import write_functional_checkpoint, write_observations


def _formal_rr_checkpoint(path, *, train_min=60.0, train_max=80.0, overall_min=60.0, overall_max=80.0):
    write_functional_checkpoint(path, overall_min, overall_max)
    payload = torch.load(path, weights_only=False)
    payload.update({
        "functional_fixture": False,
        "formal_candidate": True,
        "validation_status": "approved_by_manual_review",
        "dataset_schema": "mlp_rr_synthetic/v1",
        "dataset_split_mode": "rhythm",
        "train_timing9_min_ms": [train_min] * 9,
        "train_timing9_max_ms": [train_max] * 9,
    })
    torch.save(payload, path)


def test_online_loader_rejects_timing_inside_overall_range_but_outside_training_domain(tmp_path):
    dataset = write_observations(tmp_path / "observations.npz")
    checkpoint = tmp_path / "candidate.pth"
    _formal_rr_checkpoint(checkpoint, train_max=69.0, overall_max=80.0)
    with pytest.raises(ValueError, match="training timing domain"):
        load_frozen_mlp_decoder(checkpoint, dataset, "mlp/configs/protocol_hhz_v1.yaml", allow_functional_fixture=False, device=torch.device("cpu"))


@pytest.mark.parametrize("field,value", [("dataset_schema", None), ("dataset_schema", "legacy_subject_timing/v1"), ("dataset_split_mode", "subject")])
def test_online_loader_rejects_legacy_or_non_rr_checkpoint_contract(tmp_path, field, value):
    dataset = write_observations(tmp_path / "observations.npz")
    checkpoint = tmp_path / "candidate.pth"
    _formal_rr_checkpoint(checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    if value is None:
        payload.pop(field)
    else:
        payload[field] = value
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="RR synthetic|dataset_schema|dataset_split_mode"):
        load_frozen_mlp_decoder(checkpoint, dataset, "mlp/configs/protocol_hhz_v1.yaml", allow_functional_fixture=False, device=torch.device("cpu"))


def test_online_loader_accepts_approved_active_rr_checkpoint(tmp_path):
    dataset = write_observations(tmp_path / "observations.npz")
    checkpoint = tmp_path / "candidate.pth"
    _formal_rr_checkpoint(checkpoint)
    decoder = load_frozen_mlp_decoder(checkpoint, dataset, "mlp/configs/protocol_hhz_v1.yaml", allow_functional_fixture=False, device=torch.device("cpu"))
    assert decoder.training is False
