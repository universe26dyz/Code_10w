import torch

from modules.module_04_quantitative_inr.quantitative_inr import QuantitativeINR, QuantitativeINRConfig
from modules.module_05_signal_decoder.trad_signal_simulator import TradProtocol, TradSignalSimulator
from modules.module_06_rigid_psf.rigid_psf_forward import GroupRigidPSF, TradQuantitativeForward


def test_psf_forward_simulates_before_sample_averaging_with_finite_gradients():
    torch.manual_seed(7)
    inr = QuantitativeINR(
        torch.tensor([[-25.0, -25.0, -25.0], [25.0, 25.0, 25.0]]),
        QuantitativeINRConfig(log2_hashmap_size=8, latent_features=8, width=12, depth=1),
    )
    rigid = GroupRigidPSF(torch.zeros((1, 6)), torch.tensor([[1.2, 1.8, 7.0]]))
    forward = TradQuantitativeForward(inr, TradSignalSimulator(), rigid, TradProtocol(tr_ms=3.1, vps=32))
    output = forward(
        torch.tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 0.0]]),
        torch.zeros(2, dtype=torch.long),
        torch.tensor([0, 9], dtype=torch.long),
        torch.full((2, 9), 70.0),
        n_psf_samples=2,
    )
    assert output.shape == (2,) and torch.isfinite(output).all()
    output.sum().backward()
    assert torch.isfinite(rigid.axisangle.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in inr.parameters())
