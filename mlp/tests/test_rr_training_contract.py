import numpy as np

from mlp.modules.module_05_signal_decoder import synthetic_dataset
from mlp.modules.module_05_signal_decoder.synthetic_dataset import generate_rr_synthetic_dataset
from mlp.modules.module_05_signal_decoder.train_mlp import load_h5_split, validate_dataset_provenance


def test_rr_schema_uses_rhythm_id_and_needs_no_timing_pool(tmp_path):
    dataset = tmp_path / "rr"
    generate_rr_synthetic_dataset(dataset, {"n_rhythms": 8, "tissue_samples_per_rhythm": 2, "split_rhythms": {"train": 4, "valid": 2, "test": 2}, "seed": 3, "store_jacobian": False, "teacher_chunk_size": 4, "teacher_device": "cpu"})
    metadata, protocol = validate_dataset_provenance(dataset, "mlp/configs/protocol_hhz_v1.yaml")
    assert metadata["schema"] == "mlp_rr_synthetic/v1" and protocol.vps == 87
    assert len(load_h5_split(dataset / "train.h5", "train")) == 8


def test_rr_metadata_separates_training_timing_domain_from_full_dataset(monkeypatch, tmp_path):
    rhythms = {
        "rhythm_id": np.arange(3, dtype=np.int64),
        "rr_ms": np.full((3, 9), 1000.0, dtype=np.float32),
        "timing9_ms": np.asarray([[100.0] * 9, [150.0] * 9, [250.0] * 9], dtype=np.float32),
        "mean_hr_bpm": np.full(3, 60.0, dtype=np.float32),
        "rr_cv": np.zeros(3, dtype=np.float32),
    }
    split = {"train": np.asarray([1]), "valid": np.asarray([0]), "test": np.asarray([2])}
    monkeypatch.setattr(synthetic_dataset, "generate_rhythm_timing9", lambda _: rhythms)
    monkeypatch.setattr(synthetic_dataset, "split_rhythm_ids", lambda *_args, **_kwargs: split)

    metadata = generate_rr_synthetic_dataset(
        tmp_path / "rr",
        {"n_rhythms": 3, "tissue_samples_per_rhythm": 1, "split_rhythms": {"train": 1, "valid": 1, "test": 1}, "seed": 4, "store_jacobian": False, "teacher_chunk_size": 1, "teacher_device": "cpu"},
    )

    ids = [set(metadata["rhythm_split_ids"][name]) for name in ("train", "valid", "test")]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    assert set.union(*ids) == {0, 1, 2}
    assert metadata["timing9_min_ms"] == [100.0] * 9
    assert metadata["timing9_max_ms"] == [250.0] * 9
    assert metadata["train_timing9_min_ms"] == [150.0] * 9
    assert metadata["train_timing9_max_ms"] == [150.0] * 9
    assert metadata["timing9_max_ms"] != metadata["train_timing9_max_ms"]
