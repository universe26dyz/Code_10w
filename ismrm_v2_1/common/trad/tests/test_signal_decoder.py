import torch

from modules.module_05_signal_decoder.trad_signal_simulator import TradProtocol, TradSignalSimulator


def test_hhz_signal_decoder_returns_normalized_ten_weights_and_finite_gradients():
    simulator = TradSignalSimulator()
    t1 = torch.tensor([900.0, 1200.0], requires_grad=True)
    t2 = torch.tensor([45.0, 70.0], requires_grad=True)
    b1 = torch.tensor([0.9, 1.05], requires_grad=True)
    timing = torch.tensor([[70.0] * 9, [90.0] * 9])
    raw = simulator(t1, t2, b1, timing, TradProtocol(tr_ms=3.2, vps=32), normalize=False)
    normalized = simulator(t1, t2, b1, timing, TradProtocol(tr_ms=3.2, vps=32), normalize=True)
    assert raw.shape == normalized.shape == (2, 10)
    assert torch.isfinite(raw).all() and torch.isfinite(normalized).all()
    torch.testing.assert_close(torch.linalg.vector_norm(normalized, dim=1), torch.ones(2), atol=1e-6, rtol=0)
    normalized.sum().backward()
    assert all(item.grad is not None and torch.isfinite(item.grad).all() for item in (t1, t2, b1))
