import torch

from modules.module_05_signal_decoder.frozen_decoder import FrozenMLPSignalDecoder
from modules.module_05_signal_decoder.mlp_model import MdmSignalMLP


def test_frozen_decoder_has_no_parameter_grad_but_preserves_tissue_input_gradients():
    decoder = FrozenMLPSignalDecoder(MdmSignalMLP())
    t1 = torch.tensor([900.0, 1200.0], requires_grad=True); t2 = torch.tensor([45.0, 65.0], requires_grad=True); b1 = torch.tensor([0.9, 1.1], requires_grad=True)
    decoder(t1, t2, b1, torch.full((2, 9), 70.0)).sum().backward()
    assert all(parameter.requires_grad is False and parameter.grad is None for parameter in decoder.mlp.parameters())
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in (t1, t2, b1))
