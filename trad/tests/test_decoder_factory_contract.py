import torch


def test_decoder_interface_preserves_batch_shape_dtype_and_gradient():
    from trad.modules.module_05_signal_decoder.decoder_factory import build_bloch_decoder
    from trad.modules.module_05_signal_decoder.trad_signal_simulator import TradProtocol
    from mlp.modules.module_05_signal_decoder.frozen_decoder import FrozenMLPSignalDecoder
    from mlp.modules.module_05_signal_decoder.mlp_model import MdmSignalMLP

    protocol = TradProtocol(3.2, 32)
    t1 = torch.full((2, 3), 1000.0, requires_grad=True)
    t2 = torch.full((2, 3), 50.0, requires_grad=True)
    b1 = torch.ones((2, 3), requires_grad=True)
    timing = torch.full((2, 3, 9), 70.0)
    outputs = []
    for decoder in (build_bloch_decoder(), FrozenMLPSignalDecoder(MdmSignalMLP(), protocol)):
        value = decoder(t1, t2, b1, timing, protocol, normalize=True)
        assert value.shape == (2, 3, 10)
        assert value.dtype == torch.float32
        assert torch.isfinite(value).all()
        value.square().mean().backward(retain_graph=True)
        outputs.append(value)
        if isinstance(decoder, FrozenMLPSignalDecoder):
            assert all(not parameter.requires_grad and parameter.grad is None for parameter in decoder.parameters())
    assert t1.grad is not None and t2.grad is not None and b1.grad is not None
    assert torch.isfinite(torch.sqrt(torch.mean((outputs[0] - outputs[1]).square())))
