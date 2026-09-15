import pytest
import torch

from modules.module_03_dataset_geometry.quantitative_point_dataset import robust_trimmed_mean_intensity


def test_trimmed_mean_intensity_matches_strict_q10_q90_mask():
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0])
    scale = robust_trimmed_mean_intensity(values, lower_quantile=0.1, upper_quantile=0.9)
    expected = values[(values > torch.quantile(values, 0.1)) & (values < torch.quantile(values, 0.9))].mean()
    torch.testing.assert_close(scale, expected)
    assert torch.isfinite(scale) and scale > 0


def test_trimmed_mean_intensity_is_scale_invariant_and_rejects_empty_trim():
    values, factor = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0]), 17.0
    scale_one = robust_trimmed_mean_intensity(values, lower_quantile=0.1, upper_quantile=0.9)
    scale_two = robust_trimmed_mean_intensity(values * factor, lower_quantile=0.1, upper_quantile=0.9)
    torch.testing.assert_close(scale_two, scale_one * factor)
    torch.testing.assert_close(values / scale_one, values * factor / scale_two)
    with pytest.raises(ValueError, match="trimmed intensity set is empty"):
        robust_trimmed_mean_intensity(torch.ones(8), lower_quantile=0.1, upper_quantile=0.9)
