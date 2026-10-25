from __future__ import annotations

import numpy as np


def test_sectorize_preserves_historical_six_segment_orientation() -> None:
    from paper_experiments.joint_vs_2dfit.aha.legacy_aha16 import aha16_labels, sectorize

    mask = np.ones((9, 9), dtype=bool)
    labels = sectorize(mask, anterior=(8, 4), inferior=(0, 4), n_sectors=6)

    assert set(np.unique(labels)) == set(range(1, 7))
    # This value is the historical anterior-to-inferior traversal result for
    # an unambiguous anterior/right, inferior/left geometry.
    assert labels[0, 4] == 6

    all_levels = aha16_labels(
        {name: mask for name in ("basal", "mid", "apical")},
        {name: ((8, 4), (0, 4)) for name in ("basal", "mid", "apical")},
    )
    assert set(np.unique(all_levels["basal"])) == set(range(1, 7))
    assert set(np.unique(all_levels["mid"])) == set(range(7, 13))
    assert set(np.unique(all_levels["apical"])) == set(range(13, 17))


def test_manual_roi_core_uses_the_historical_one_mm_erosion() -> None:
    from paper_experiments.joint_vs_2dfit.aha.legacy_aha16 import build_filled_masks

    outer = np.zeros((7, 7), bool); outer[1:6, 1:6] = True
    masks = build_filled_masks(outer, np.zeros((7, 7), bool), (1.0, 1.0))
    assert masks["myocardium"].sum() == 25
    assert masks["myocardium_core"].sum() == 9


def test_segment_statistics_use_one_shared_support_and_historical_sample_sd() -> None:
    from paper_experiments.joint_vs_2dfit.aha.legacy_aha16 import segment_statistics

    labels = np.array([[1, 1, 2, 2]], dtype=np.int16)
    reference = np.array([[10.0, 14.0, 20.0, 24.0]])
    prediction = np.array([[11.0, 13.0, 19.0, 27.0]])
    support = np.array([[True, True, True, False]])

    values = segment_statistics(reference, prediction, labels, support, segments=(1, 2))

    assert values[1]["n"] == 2
    assert values[1]["reference_mean"] == 12.0
    assert values[1]["reference_sd"] == np.sqrt(8.0)
    assert values[1]["prediction_mean"] == 12.0
    assert values[2]["n"] == 1
    assert values[2]["prediction_sd"] == 0.0


def test_single_bullseye_writes_individual_png_and_pdf_without_montage(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.legacy_aha16 import render_single_bullseye

    output = tmp_path / "CYJ_L100_proposed_T1_bullseye.png"
    render_single_bullseye(output, {segment: float(segment) for segment in range(1, 17)}, parameter="t1", title="CYJ L100 Proposed T1")

    assert output.is_file()
    assert output.with_suffix(".pdf").is_file()
