import torch

from modules.module_05_signal_decoder.mlp_model import MdmSignalMLP


def test_mdm_mlp_has_exact_12_200_200_200_10_normalized_contract():
    model = MdmSignalMLP()
    output = model(torch.randn(4, 12))
    assert output.shape == (4, 10)
    torch.testing.assert_close(torch.linalg.vector_norm(output, dim=1), torch.ones(4), atol=1e-6, rtol=0)
    assert [block[0].out_features for block in (model.mlp_block1, model.mlp_block2, model.mlp_block3)] == [200, 200, 200]
    assert model.mlp_block4.out_features == 10
