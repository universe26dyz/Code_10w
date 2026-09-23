import json

import numpy as np
import pytest
import torch

from mlp.modules.module_05_signal_decoder.mlp_model import MdmSignalMLP
from mlp.modules.module_05_signal_decoder.provenance import sha256_file
from mlp.modules.module_05_signal_decoder.validate_real_timings import validate_real_timings


def _rr_dataset(tmp_path, low=60.0, high=80.0):
    rr = tmp_path / "rr"; rr.mkdir()
    metadata = {
        "schema": "mlp_rr_synthetic/v1", "split_mode": "rhythm",
        "protocol": {"tr_ms": 2.61, "vps": 87, "fa_deg": [45.0] * 3,
                     "ti_ms": [50.0, 150.0], "t2prep_ms": [35.0, 45.0, 55.0], "n_ramp_up": 10},
        "train_timing9_min_ms": [low] * 9, "train_timing9_max_ms": [high] * 9,
    }
    (rr / "dataset_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return rr


def _prepared(tmp_path, tr_values=(2.61,) * 10, vps_values=(87,) * 10, timing=70.0):
    prepared = tmp_path / "prepared"
    for stack in ("sax", "2ch", "4ch"):
        target = prepared / "CYJ" / stack; target.mkdir(parents=True)
        count = len(tr_values)
        np.savez_compressed(target / "observations.npz", group_idx=np.zeros(count, dtype=np.int64),
                            timing9_ms=np.full((count, 9), timing, dtype=np.float32),
                            tr_ms=np.asarray(tr_values), vps=np.asarray(vps_values, dtype=np.int64))
    return prepared


def _candidate(tmp_path):
    checkpoint = tmp_path / "candidate.pth"
    torch.save({"state_dict": MdmSignalMLP().state_dict(), "dataset_schema": "mlp_rr_synthetic/v1", "dataset_split_mode": "rhythm"}, checkpoint)
    return checkpoint


def _validate(tmp_path, prepared, rr):
    checkpoint = _candidate(tmp_path)
    report = validate_real_timings(checkpoint, prepared, ["CYJ"], rr, "mlp/configs/protocol_hhz_v1.yaml", tmp_path / "real_timing_validation.json", samples_per_timing=2)
    return checkpoint, report


def test_real_timing_validation_accepts_fixed_hhz_protocol_and_records_provenance(tmp_path):
    rr = _rr_dataset(tmp_path); checkpoint, report = _validate(tmp_path, _prepared(tmp_path), rr)
    assert report["training_domain_coverage_status"] == "PASS"
    assert report["approval_recommendation"] == "eligible_for_human_review"
    assert report["expected_protocol"] == {"tr_ms": 2.61, "vps": 87}
    assert report["checkpoint_sha256"] == sha256_file(checkpoint)
    assert report["rr_dataset_metadata_sha256"] == sha256_file(rr / "dataset_metadata.json")
    assert (tmp_path / "real_timing_validation_per_source.csv").is_file()


@pytest.mark.parametrize(("tr_values", "vps_values", "message"), [
    ((2.61,) * 10, (85,) * 10, "VPS=85"),
    ((2.62,) * 10, (87,) * 10, "TR=2.62 ms"),
    ((2.61, 2.62) + (2.61,) * 8, (87,) * 10, "inconsistent TR"),
    ((2.61,) * 10, (87, 85) + (87,) * 8, "inconsistent VPS"),
])
def test_real_timing_validation_rejects_protocol_or_within_group_inconsistency(tmp_path, tr_values, vps_values, message):
    with pytest.raises(ValueError, match=message):
        _validate(tmp_path, _prepared(tmp_path, tr_values, vps_values), _rr_dataset(tmp_path))


def test_real_timing_validation_marks_outside_training_domain_not_approval_eligible(tmp_path):
    _, report = _validate(tmp_path, _prepared(tmp_path, timing=90.0), _rr_dataset(tmp_path))
    assert report["training_domain_coverage_status"] == "OUTSIDE_TRAINING_DOMAIN"
    assert report["approval_recommendation"] == "do_not_approve"
