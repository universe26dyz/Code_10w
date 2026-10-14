from __future__ import annotations

import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.simulation.generate_clean_dataset import retired_phase1_1
from paper_experiments.joint_vs_2dfit.simulation.generate_phase1_2_xcat3d import acquisition_times_from_timing9
from paper_experiments.joint_vs_2dfit.simulation.simulate_hhz_signals import generate_signal_field


def test_retired_composite_entry_point_fails_loudly() -> None:
    with pytest.raises(RuntimeError, match="invalid for scientific use"):
        retired_phase1_1()


def test_signal_generation_is_exactly_zero_outside_support() -> None:
    signals = generate_signal_field(
        np.array([1410.0, 0.0], dtype=np.float32),
        np.array([50.0, 0.0], dtype=np.float32),
        np.ones(2, dtype=np.float32),
        np.arange(1, 10, dtype=np.float32),
        signal_support_mask=np.array([True, False]),
    )
    assert np.any(signals[0] != 0.0)
    assert np.array_equal(signals[1], np.zeros(10, dtype=np.float32))


def test_acquisition_times_round_trip_to_selected_timing() -> None:
    timing9 = np.arange(10.0, 100.0, 10.0)
    acquisition = acquisition_times_from_timing9(timing9)
    assert acquisition.shape == (10,)
    assert acquisition[0] == 0.0
    assert np.all(np.diff(acquisition) > 0.0)
