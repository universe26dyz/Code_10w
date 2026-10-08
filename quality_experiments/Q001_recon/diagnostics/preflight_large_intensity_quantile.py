"""Exercise the exact full-FOV quantile fallback above PyTorch's 2**24 limit."""

from __future__ import annotations

import json

import torch

from trad.modules.module_03_dataset_geometry.quantitative_point_dataset import (
    _exact_linear_quantile_kthvalue,
)


N_VALUES = 2**24 + 1


def run(n_values: int = N_VALUES) -> dict[str, object]:
    if n_values <= 2**24:
        raise ValueError("Large-quantile preflight requires more than 2**24 values.")
    values = torch.arange(n_values, dtype=torch.float32, device="cpu")
    original_quantile = torch.quantile

    def forbidden_quantile(*args, **kwargs):
        raise AssertionError("Large-tensor preflight must not call torch.quantile.")

    torch.quantile = forbidden_quantile
    try:
        q10 = _exact_linear_quantile_kthvalue(values, 0.1)
        q90 = _exact_linear_quantile_kthvalue(values, 0.9)
    finally:
        torch.quantile = original_quantile
    if not torch.isfinite(q10) or not torch.isfinite(q90) or not q10 < q90:
        raise RuntimeError("Exact large-tensor kthvalue preflight produced invalid quantiles.")
    return {
        "status": "PASS",
        "algorithm": "order_statistic_kthvalue_linear",
        "input_numel": n_values,
        "dtype": str(values.dtype),
        "device": values.device.type,
        "q10": float(q10),
        "q90": float(q90),
        "torch_quantile_called": False,
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))
