import json

import numpy as np
import torch

from mlp.modules.module_05_signal_decoder.mlp_model import MdmSignalMLP
from mlp.modules.module_05_signal_decoder.validate_real_timings import validate_real_timings


def test_real_timing_validation_marks_outside_training_domain_not_approval_eligible(tmp_path):
    prepared = tmp_path / "prepared"
    for stack in ("sax", "2ch", "4ch"):
        target = prepared / "CYJ" / stack
        target.mkdir(parents=True)
        np.savez_compressed(target / "observations.npz", group_idx=np.zeros(10, dtype=np.int64), timing9_ms=np.full((10, 9), 90.0, dtype=np.float32), tr_ms=np.full(10, 2.61), vps=np.full(10, 87, dtype=np.int64))
    rr = tmp_path / "rr"; rr.mkdir()
    rr_meta = {"schema": "mlp_rr_synthetic/v1", "split_mode": "rhythm", "protocol": {"tr_ms": 2.61, "vps": 87, "fa_deg": [45.0] * 3, "ti_ms": [50.0, 150.0], "t2prep_ms": [35.0, 45.0, 55.0], "n_ramp_up": 10}, "train_timing9_min_ms": [60.0] * 9, "train_timing9_max_ms": [80.0] * 9}
    (rr / "dataset_metadata.json").write_text(json.dumps(rr_meta), encoding="utf-8")
    checkpoint = tmp_path / "candidate.pth"
    torch.save({"state_dict": MdmSignalMLP().state_dict(), "dataset_schema": "mlp_rr_synthetic/v1", "dataset_split_mode": "rhythm"}, checkpoint)

    report = validate_real_timings(checkpoint, prepared, ["CYJ"], rr, "mlp/configs/protocol_hhz_v1.yaml", tmp_path / "real_timing_validation.json", samples_per_timing=2)

    assert report["training_domain_coverage_status"] == "OUTSIDE_TRAINING_DOMAIN"
    assert report["approval_recommendation"] == "do_not_approve"
    assert (tmp_path / "real_timing_validation_per_source.csv").is_file()
