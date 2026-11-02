import pytest
import torch
from torch import nn

from modules.module_07_objective_training.trad_trainer import _quantitative_regularization


class _ZeroSpatialGradientINR(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = nn.Parameter(torch.tensor(1.0))

    def forward(self, points):
        # The values are connected to points, but their spatial derivative is
        # exactly zero and still retains a graph back to parameter.
        constant = self.parameter + (points[:, :1] * self.parameter * 0.0)
        return {"t1_ms": constant * 1000.0, "t2_ms": constant * 50.0, "b1": constant}


class _RegularizationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.inr = _ZeroSpatialGradientINR()


def test_zero_spatial_gradient_regularization_backward_is_finite():
    model = _RegularizationModel()
    points = torch.zeros(4, 3)
    regularization = _quantitative_regularization(
        model, points, spatial_scaling=30.0, weights={"t1": 1.0, "t2": 1.0, "b1": 1.0}
    )
    total = sum(regularization.values())
    assert torch.isfinite(total)
    total.backward()
    assert torch.isfinite(model.inr.parameter.grad).all()


def test_nonfinite_gradient_aborts_before_optimizer_step():
    from modules.module_07_objective_training.trad_trainer import _assert_finite_gradients

    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.5)
    parameter.grad = torch.tensor(float("nan"))
    before = parameter.detach().clone()
    with pytest.raises(FloatingPointError, match="stage A.*iteration 7.*parameter"):
        _assert_finite_gradients(nn.ModuleList([nn.ParameterList([parameter])]), "A", 7)
    torch.testing.assert_close(parameter, before)


def test_nonfinite_parameter_guard_reports_parameter_name():
    from modules.module_07_objective_training.trad_trainer import _assert_finite_parameters

    parameter = nn.Parameter(torch.tensor(float("inf")))
    with pytest.raises(FloatingPointError, match="stage B.*iteration 8.*parameter"):
        _assert_finite_parameters(nn.ModuleList([nn.ParameterList([parameter])]), "B", 8)
