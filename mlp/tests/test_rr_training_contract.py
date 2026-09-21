from modules.module_05_signal_decoder.synthetic_dataset import generate_rr_synthetic_dataset
from modules.module_05_signal_decoder.train_mlp import load_h5_split, validate_dataset_provenance


def test_rr_schema_uses_rhythm_id_and_needs_no_timing_pool(tmp_path):
    dataset = tmp_path / "rr"
    generate_rr_synthetic_dataset(dataset, {"n_rhythms": 8, "tissue_samples_per_rhythm": 2, "split_rhythms": {"train": 4, "valid": 2, "test": 2}, "seed": 3, "store_jacobian": False, "teacher_chunk_size": 4, "teacher_device": "cpu"})
    metadata, pool, protocol = validate_dataset_provenance(dataset, None, "configs/protocol_hhz_v1.yaml")
    assert metadata["schema"] == "mlp_rr_synthetic/v1" and pool is None and protocol.vps == 87
    assert len(load_h5_split(dataset / "train.h5", "train")) == 8
