import h5py
import numpy as np
import pytest
import torch

from modules.module_05_signal_decoder.synthetic_dataset import (
    FORMAL_MLP_SPLIT,
    generate_mlp_dataset,
    validate_formal_mlp_subject_split,
)
from modules.module_05_signal_decoder.mlp_model import MdmSignalMLP
from modules.module_05_signal_decoder.train_mlp import mlp_training_loss


def _pool(path, subjects=("CYJ", "DYZ", "HHZ", "HJL")):
    timing = np.asarray([[60.0 + index] * 9 for index in range(len(subjects))])
    np.savez_compressed(path, timing9_ms=timing, tr_ms=np.asarray(3.2), vps=np.asarray(87), source_id=np.asarray(subjects), group_id=np.arange(len(subjects)), stack_idx=np.zeros(len(subjects), dtype=np.int64), subject_id=np.asarray(subjects), functional_fixture=np.asarray(False))


def test_offline_teacher_h5_contains_normalized_parameter_jacobian_and_provenance(tmp_path):
    pool = tmp_path / "pool.npz"; _pool(pool)
    result = generate_mlp_dataset(pool, "configs/protocol_hhz_v1.yaml", tmp_path / "h5", {"train": 4, "valid": 2, "test": 2, "split": {"mode": "subject", "train_subjects": ["CYJ", "DYZ"], "valid_subjects": ["HHZ"], "test_subjects": ["HJL"], "formal_mlp": True}}, seed=3, chunk_size=2)
    with h5py.File(tmp_path / "h5" / "train.h5", "r") as handle:
        assert handle["input12"].shape == (4, 12)
        assert handle["target_signal10"].shape == (4, 10)
        assert handle["target_jacobian10x3"].shape == (4, 10, 3)
        assert handle["target_jacobian10x3"].attrs["parameter_space"] == "T1/1000,T2/1000,B1"
    assert result["jacobian_target_units"] == "normalized_signal_per_normalized_T1_T2_B1"


def test_formal_mlp_split_is_fixed_and_dyl_is_rejected():
    assert validate_formal_mlp_subject_split(np.asarray(["CYJ", "DYZ", "HHZ", "HJL"]), FORMAL_MLP_SPLIT) == {"train": ["CYJ", "DYZ"], "valid": ["HHZ"], "test": ["HJL"]}
    with pytest.raises(ValueError, match="DYL"):
        validate_formal_mlp_subject_split(np.asarray(["CYJ", "DYZ", "HHZ", "DYL"]), FORMAL_MLP_SPLIT)


def test_zero_jacobian_and_cosine_weights_are_exact_signal_mse_only():
    model = MdmSignalMLP().eval()
    input12 = torch.rand(3, 12)
    target = torch.rand(3, 10); jacobian = torch.rand(3, 10, 3)
    total, terms = mlp_training_loss(model, input12, target, jacobian, {"signal_mse_weight": 1.0, "jacobian_weight": 0.0, "cosine_weight": 0.0})
    torch.testing.assert_close(total, (model(input12) - target).pow(2).mean())
    assert terms["jacobian_mse"].item() == 0 and terms["cosine"].item() == 0
