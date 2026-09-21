import numpy as np

from evaluation_common.myocardium.build_native_bundle import (
    myocardium_core_1px,
    myocardium_core_legacy,
)


def _ring() -> np.ndarray:
    values = np.zeros((1, 13, 13), dtype=bool)
    values[0, 1:12, 1:12] = True
    values[0, 5:8, 5:8] = False
    return values


def test_legacy_edt_core_reproduces_full_at_125_mm_sampling():
    full = _ring()

    legacy = myocardium_core_legacy(full, (1.25, 1.25))

    assert np.array_equal(legacy, full)


def test_one_pixel_core_removes_in_plane_boundary_without_mutating_full():
    full = _ring()
    original = full.copy()

    core = myocardium_core_1px(full)

    assert np.array_equal(full, original)
    assert core.sum() < full.sum()
    assert core.sum() > 0
    assert not core[0, 1, 1]


def test_one_pixel_core_is_slice_wise_and_empty_core_has_no_fallback():
    full = np.zeros((2, 7, 7), dtype=bool)
    full[0, 1:6, 1:6] = True
    full[1, 3, 3] = True

    core = myocardium_core_1px(full)

    assert core[0].any()
    assert not core[1].any()
    assert not np.any(core[:, 0])
