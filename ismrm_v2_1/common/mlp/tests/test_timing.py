import numpy as np

from modules.module_02_data_bridge.timing import compute_duration_before_acq


def test_duration_before_acq_matches_hhz_indexed_delays():
    duration = compute_duration_before_acq(
        acquisition_times_ms=np.arange(10, dtype=np.float64) * 100.0,
        tr_ms=4.0,
        n_ex=5,
        ti_ms=np.array([50.0, 150.0]),
        t2prep_ms=np.array([35.0, 45.0, 55.0]),
    )
    np.testing.assert_allclose(duration, [0.0, 80.0, 80.0, 80.0, -70.0, 80.0, 80.0, 45.0, 35.0, 25.0])
    np.testing.assert_allclose(duration[1:], [-0.0 + 80.0, 80.0, 80.0, -70.0, 80.0, 80.0, 45.0, 35.0, 25.0])
