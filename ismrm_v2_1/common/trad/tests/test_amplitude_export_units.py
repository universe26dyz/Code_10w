from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from modules.module_08_inference_export.export_quantitative import sample_quantitative_fields


class _ConstantINR(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, points):
        count = points.shape[0]
        return {
            "t1_ms": torch.full((count,), 1000.0, device=points.device),
            "t2_ms": torch.full((count,), 50.0, device=points.device),
            "b1": torch.full((count,), 0.9, device=points.device),
            "amplitude": torch.full((count,), 2.0, device=points.device),
        }


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.inr = _ConstantINR()
        self.register_buffer("intensity_scale", torch.tensor(100.0))


def test_sampled_amplitude_is_restored_to_original_intensity_units():
    space = SimpleNamespace(
        physical_bbox_ras_mm=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        physical_ras_to_train=lambda points: points,
    )
    fields, _ = sample_quantitative_fields(_Model(), space, output_resolution_mm=1.0, batch_size=1)
    assert fields["amplitude"].item() == 200.0
    assert fields["t1_ms"].item() == 1000.0
    assert fields["t2_ms"].item() == 50.0
    np.testing.assert_allclose(fields["b1"].item(), 0.9, rtol=0, atol=1e-6)
