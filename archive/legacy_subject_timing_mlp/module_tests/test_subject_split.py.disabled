import json

import numpy as np

from modules.module_05_signal_decoder.build_timing_pool import build_timing_pool_from_manifest
from modules.module_05_signal_decoder.synthetic_dataset import split_subject_ids


def _obs(path, timing):
    np.savez_compressed(path, group_idx=np.zeros(10, dtype=np.int64), weight_idx=np.arange(10), stack_idx=np.zeros(10, dtype=np.int64), timing9_ms=np.full((10, 9), timing), tr_ms=np.full(10, 3.2), vps=np.full(10, 32, dtype=np.int64))


def test_manifest_preserves_subject_provenance_and_subject_split_is_disjoint(tmp_path):
    sources = []
    for subject, timing in (("S001", 60.0), ("S002", 60.0), ("S003", 80.0), ("S004", 90.0), ("S005", 100.0)):
        path = tmp_path / f"{subject}.npz"; _obs(path, timing)
        sources.append({"subject_id": subject, "stack": "sax", "observations": str(path)})
    manifest = tmp_path / "sources.json"; manifest.write_text(json.dumps({"sources": sources}))
    pool = tmp_path / "timing_pool.npz"
    build_timing_pool_from_manifest(manifest, pool)
    with np.load(pool, allow_pickle=False) as data:
        assert data["timing9_ms"].shape == (5, 9)  # S001/S002 numeric duplicate is retained
        assert data["subject_id"].tolist() == ["S001", "S002", "S003", "S004", "S005"]
    split = split_subject_ids(np.asarray(["S001", "S002", "S003", "S004", "S005"]), {"mode": "subject", "train_subjects": ["S001", "S002", "S003"], "valid_subjects": ["S004"], "test_subjects": ["S005"]})
    assert not set(split["train"]).intersection(split["valid"]) and not set(split["train"]).intersection(split["test"])
