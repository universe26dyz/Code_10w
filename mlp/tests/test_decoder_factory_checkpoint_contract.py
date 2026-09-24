import json

import numpy as np
import torch

from mlp.modules.module_05_signal_decoder.decoder_factory import frozen_mlp_decoder_factory
from mlp.modules.module_05_signal_decoder.provenance import sha256_file
from mlp.scripts.approve_formal_checkpoint import approve_formal_checkpoint
from online_helpers import write_functional_checkpoint, write_observations
from trad.modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset


def _formal_dataset(path):
    write_observations(path)
    with np.load(path, allow_pickle=False) as data:
        observed = {key: data[key] for key in data.files}
    observed["tr_ms"] = np.full(10, 2.61, dtype=np.float64)
    observed["vps"] = np.full(10, 87, dtype=np.int64)
    np.savez_compressed(path, **observed)
    return QuantPointDataset([path])


def _formal_candidate(path):
    write_functional_checkpoint(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint.update({
        "functional_fixture": False, "formal_candidate": True, "validation_status": "unvalidated",
        "dataset_schema": "mlp_rr_synthetic/v1", "dataset_split_mode": "rhythm",
        "protocol_hhz_v1": {"tr_ms": 2.61, "vps": 87, "fa_deg": [45.0] * 3,
                            "ti_ms": [50.0, 150.0], "t2prep_ms": [35.0, 45.0, 55.0], "n_ramp_up": 10},
    })
    checkpoint.pop("scientific_checkpoint")
    torch.save(checkpoint, path)


def _approve(candidate, approved):
    validation = candidate.with_name("validation.json")
    validation.write_text(json.dumps({"checkpoint_sha256": sha256_file(candidate), "validation_status": "awaiting_manual_review"}), encoding="utf-8")
    real = candidate.with_name("real_timing.json")
    real.write_text(json.dumps({"schema": "rr_real_vps87_validation/v1", "checkpoint_sha256": sha256_file(candidate),
                                "training_domain_coverage_status": "PASS", "approval_recommendation": "eligible_for_human_review",
                                "validation_status": "awaiting_manual_review"}), encoding="utf-8")
    approve_formal_checkpoint(candidate, validation, real, approved, "reviewed")


def test_approved_rr_checkpoint_without_legacy_scientific_field_loads_through_factory(tmp_path):
    dataset = _formal_dataset(tmp_path / "formal_observations.npz")
    candidate, approved = tmp_path / "candidate.pth", tmp_path / "approved.pth"
    _formal_candidate(candidate); _approve(candidate, approved)
    approved_payload = torch.load(approved, map_location="cpu", weights_only=False)
    assert "scientific_checkpoint" not in approved_payload

    decoder, metadata = frozen_mlp_decoder_factory(dataset, {"decoder": {"checkpoint": str(approved), "allow_functional_fixture_checkpoint": False, "protocol_yaml": "mlp/configs/protocol_hhz_v1.yaml"}}, object(), torch.device("cpu"))

    assert metadata["decoder_type"] == "FrozenMLP"
    assert metadata["scientific_checkpoint"] is True
    assert metadata["source_mlp_checkpoint_metadata"]["scientific_checkpoint"] is True
    assert all(not parameter.requires_grad for parameter in decoder.parameters())
    assert "scientific_checkpoint" not in torch.load(approved, map_location="cpu", weights_only=False)


def test_functional_fixture_factory_metadata_is_not_scientific(tmp_path):
    dataset = write_observations(tmp_path / "functional_observations.npz")
    checkpoint = tmp_path / "functional.pth"; write_functional_checkpoint(checkpoint)
    _, metadata = frozen_mlp_decoder_factory(dataset, {"decoder": {"checkpoint": str(checkpoint), "allow_functional_fixture_checkpoint": True, "protocol_yaml": "mlp/configs/protocol_hhz_v1.yaml"}}, object(), torch.device("cpu"))
    assert metadata["scientific_checkpoint"] is False
