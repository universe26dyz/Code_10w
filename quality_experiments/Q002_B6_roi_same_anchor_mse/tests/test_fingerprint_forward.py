import torch
from torch import nn
import pytest

from trad.modules.module_06_rigid_psf.rigid_psf_forward import TradQuantitativeForward


class _PSF(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def sample_local_then_transform(self, local, group, n_samples):
        self.calls += 1
        return local[:, None, :].expand(-1, n_samples, -1)


class _INR(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(2.0))

    def forward(self, xyz):
        value = xyz[:, 0] * self.gain
        return {"t1_ms": value, "t2_ms": value, "b1": value, "amplitude": value + 3.0}


class _Decoder(nn.Module):
    def forward(self, t1, t2, b1, timing, protocol, normalize=True):
        return t1[:, None] + torch.arange(10, dtype=t1.dtype, device=t1.device)[None, :]


def test_fingerprint_forward_shares_one_k8_psf_and_keeps_amplitude_per_sample():
    psf, inr = _PSF(), _INR()
    forward = TradQuantitativeForward(inr, _Decoder(), psf, protocol=object())
    local = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], requires_grad=True)
    result = forward.forward_fingerprint(local, torch.tensor([0, 0]), torch.zeros((2, 9)), n_psf_samples=8)

    assert psf.calls == 1 and result.shape == (2, 10)
    expected = (torch.tensor([2.0, 4.0]) + 3.0)[:, None] * (torch.tensor([2.0, 4.0])[:, None] + torch.arange(10.0)[None])
    assert torch.allclose(result, expected)
    result.sum().backward()
    assert torch.isfinite(inr.gain.grad) and torch.isfinite(local.grad).all()


def test_fingerprint_k1_is_the_full_vector_without_weight_gather():
    forward = TradQuantitativeForward(_INR(), _Decoder(), _PSF(), protocol=object())
    result = forward.forward_fingerprint(torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([0]), torch.zeros((1, 9)), n_psf_samples=1)
    assert torch.allclose(result[0], (2.0 + 3.0) * (2.0 + torch.arange(10.0)))


def test_real_frozen_mlp_decoder_uses_bk_by_10_contract():
    from mlp.modules.module_05_signal_decoder.frozen_decoder import FrozenMLPSignalDecoder
    from mlp.modules.module_05_signal_decoder.mlp_model import MdmSignalMLP

    decoder = FrozenMLPSignalDecoder(MdmSignalMLP())
    direct = decoder(torch.full((16,), 1000.0), torch.full((16,), 50.0), torch.ones(16), torch.zeros((16, 9)))
    assert direct.shape == (16, 10)
    result = TradQuantitativeForward(_INR(), decoder, _PSF(), protocol=object()).forward_fingerprint(torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), torch.tensor([0, 0]), torch.zeros((2, 9)), n_psf_samples=8)
    assert result.shape == (2, 10)


def test_fingerprint_rejects_noncontract_singleton_dimension():
    class BadDecoder(nn.Module):
        def forward(self, t1, t2, b1, timing, protocol, normalize=True):
            return torch.ones((t1.numel(), 1, 10))
    with pytest.raises(ValueError, match="\\[B\\*K,10\\]"):
        TradQuantitativeForward(_INR(), BadDecoder(), _PSF(), protocol=object()).forward_fingerprint(torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([0]), torch.zeros((1, 9)), n_psf_samples=1)
