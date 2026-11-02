import torch
from torch import nn

from mlp.modules.module_05_signal_decoder.frozen_decoder import FrozenMLPSignalDecoder
from mlp.modules.module_05_signal_decoder.mlp_model import MdmSignalMLP


def test_frozen_decoder_keeps_batchnorm_eval_under_outer_train_and_is_batch_invariant():
    decoder = FrozenMLPSignalDecoder(MdmSignalMLP()).train()
    assert all(module.training is False for module in decoder.mlp.modules() if isinstance(module, nn.BatchNorm1d))
    t1, t2, b1, timing = torch.tensor([900.0]), torch.tensor([45.0]), torch.tensor([0.9]), torch.full((1, 9), 70.0)
    single = decoder(t1, t2, b1, timing)
    batch = decoder(t1.repeat(3), t2.repeat(3), b1.repeat(3), timing.repeat(3, 1))
    torch.testing.assert_close(single[0], batch[0])
