import numpy as np
import h5py

from modules.module_05_signal_decoder.synthetic_rr_domain import RRDomainConfig, generate_rhythm_timing9, split_rhythm_ids
from modules.module_05_signal_decoder.synthetic_dataset import generate_rr_synthetic_dataset


def test_rr_domain_is_reproducible_and_uses_canonical_timing_contract():
    config = RRDomainConfig(n_rhythms=6, seed=7, tr_ms=2.61, vps=87, ti_ms=(50.0, 150.0), t2prep_ms=(35.0, 45.0, 55.0))
    first = generate_rhythm_timing9(config)
    second = generate_rhythm_timing9(config)
    assert np.array_equal(first["rr_ms"], second["rr_ms"])
    assert np.array_equal(first["timing9_ms"], second["timing9_ms"])
    assert first["rr_ms"].shape == (6, 9)
    assert first["timing9_ms"].shape == (6, 9)
    assert np.all(first["timing9_ms"] > 0)


def test_rhythm_split_is_disjoint_and_exhaustive():
    split = split_rhythm_ids(20, seed=11, counts={"train": 14, "valid": 3, "test": 3})
    all_ids = np.concatenate(tuple(split.values()))
    assert set(all_ids.tolist()) == set(range(20))
    assert len(set(all_ids.tolist())) == 20


def test_signal_only_rr_dataset_skips_jacobian_storage(tmp_path):
    metadata = generate_rr_synthetic_dataset(
        tmp_path / "dataset",
        {"n_rhythms": 8, "tissue_samples_per_rhythm": 2, "split_rhythms": {"train": 4, "valid": 2, "test": 2}, "seed": 3, "store_jacobian": False, "teacher_chunk_size": 4, "teacher_device": "cpu"},
    )
    assert metadata["split_mode"] == "rhythm" and metadata["store_jacobian"] is False
    with h5py.File(tmp_path / "dataset" / "train.h5", "r") as handle:
        assert "target_jacobian10x3" not in handle
        assert handle["input12"].shape == (8, 12)
