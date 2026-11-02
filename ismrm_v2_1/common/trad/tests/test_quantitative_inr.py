import torch

from modules.module_04_quantitative_inr.quantitative_inr import QuantitativeINR, QuantitativeINRConfig


def test_quantitative_inr_uses_bounded_continuous_fields_with_finite_gradients():
    model = QuantitativeINR(
        torch.tensor([[-20.0, -20.0, -20.0], [20.0, 20.0, 20.0]]),
        QuantitativeINRConfig(log2_hashmap_size=8, latent_features=8, width=12, depth=1),
    )
    xyz = torch.tensor([[0.0, 0.0, 0.0], [2.0, -3.0, 5.0]], requires_grad=True)
    fields = model(xyz)
    assert torch.all((fields["t2_ms"] > 5.0) & (fields["t2_ms"] < 200.0))
    assert torch.all((fields["t1_ms"] > fields["t2_ms"]) & (fields["t1_ms"] < 2500.0))
    assert torch.all((fields["b1"] > 0.1) & (fields["b1"] < 1.2))
    assert torch.all(fields["amplitude"] > 0)
    sum(value.sum() for value in fields.values()).backward()
    assert torch.isfinite(xyz.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
