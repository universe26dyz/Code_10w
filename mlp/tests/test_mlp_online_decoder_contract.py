import pytest
import torch

from modules.module_05_signal_decoder.frozen_decoder import FrozenMLPSignalDecoder
from modules.module_05_signal_decoder.mlp_model import MdmSignalMLP
from modules.module_05_signal_decoder.trad_teacher.trad_signal_simulator import TradProtocol


def test_frozen_decoder_accepts_trad_forward_signature_and_keeps_tissue_gradient():
    protocol = TradProtocol(3.2, 32)
    decoder = FrozenMLPSignalDecoder(MdmSignalMLP(), protocol)
    t1 = torch.full((2,), 1000.0, requires_grad=True); t2 = torch.full((2,), 80.0, requires_grad=True)
    b1 = torch.full((2,), 1.0, requires_grad=True); timing = torch.full((2, 9), 70.0)
    output = decoder(t1, t2, b1, timing, protocol, normalize=True)
    torch.testing.assert_close(output.norm(dim=-1), torch.ones(2), atol=1e-5, rtol=1e-5)
    output.sum().backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in (t1, t2, b1))
    with pytest.raises(ValueError, match="only provides L2-normalized"):
        decoder(t1, t2, b1, timing, protocol, normalize=False)
