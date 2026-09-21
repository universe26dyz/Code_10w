import numpy as np

from evaluation.scmr.metrics import signal_agreement_metrics


def test_signal_metrics_are_explicitly_in_input_intensity_units():
    result = signal_agreement_metrics(np.array([1.0, 3.0]), np.array([2.0, 5.0]), np.array([True, True]), min_pixels=1)
    assert result["units"] == "original_input_intensity"
    assert result["MAE_signal"] == 1.5
    assert "MAE_ms" not in result
