from types import SimpleNamespace

import torch
import pytest

from mlp.modules.module_05_signal_decoder.checkpoint_loader import load_frozen_mlp_decoder
from mlp.modules.module_05_signal_decoder.mlp_model import MdmSignalMLP


def _checkpoint(path):
    model = MdmSignalMLP()
    torch.save({"state_dict": model.state_dict(), "architecture": "12-200-200-200-10", "input_normalization": "T1/1000,T2/1000,B1,timing9/1000", "output_normalization": "raw_l2_normalized", "protocol_hhz_v1": {"tr_ms": 3.2, "vps": 32, "fa_deg": [45.0, 45.0, 45.0], "ti_ms": [50.0, 150.0], "t2prep_ms": [35.0, 45.0, 55.0], "n_ramp_up": 10}, "parameter_ranges": {"t1_ms": [20, 2500], "t2_ms": [5, 200], "b1": [0.1, 1.2], "constraint": "T1>T2"}, "functional_fixture": True, "formal_candidate": False, "validation_status": "functional_smoke", "scientific_checkpoint": False, "timing9_min_ms": [60.0] * 9, "timing9_max_ms": [80.0] * 9, "train_timing9_min_ms": [60.0] * 9, "train_timing9_max_ms": [80.0] * 9}, path)


def test_functional_checkpoint_requires_explicit_smoke_allowance(tmp_path):
    path = tmp_path / "functional.pth"; _checkpoint(path)
    dataset = SimpleNamespace(timing=torch.full((10, 9), 70.0), validated_tr_vps=lambda: (3.2, 32))
    with pytest.raises(ValueError, match="functional-fixture"):
        load_frozen_mlp_decoder(path, dataset, "mlp/configs/protocol_hhz_v1.yaml", allow_functional_fixture=False, device=torch.device("cpu"))
    decoder = load_frozen_mlp_decoder(path, dataset, "mlp/configs/protocol_hhz_v1.yaml", allow_functional_fixture=True, device=torch.device("cpu"))
    assert not any(parameter.requires_grad for parameter in decoder.parameters())
