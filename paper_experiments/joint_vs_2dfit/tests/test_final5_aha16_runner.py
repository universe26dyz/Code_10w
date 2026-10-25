from __future__ import annotations

import numpy as np
import pytest


def test_paired_aha_plane_uses_one_anatomical_label_and_one_support() -> None:
    from paper_experiments.joint_vs_2dfit.aha.final5_aha16 import build_paired_aha16_plane

    anatomy = np.ones((9, 9), dtype=bool)
    reference = np.full((9, 9), 1000.0)
    proposed = reference + 10.0
    fit_first = reference - 20.0
    proposed[0, 0] = np.nan
    fit_first[0, 1] = np.nan
    result = build_paired_aha16_plane(
        level="basal", reference=reference, proposed=proposed, fit_first=fit_first,
        proposed_valid=np.ones_like(anatomy), fit_first_valid=np.ones_like(anatomy),
        myocardium_core=anatomy, anterior_xy=(8, 4), inferior_xy=(0, 4),
    )

    assert np.array_equal(result["proposed_support"], result["fit_first_support"])
    assert result["proposed_support"].sum() == 79
    assert set(np.unique(result["labels"])) == set(range(1, 7))
    assert all(values["prediction_mean"] == 1010.0 for values in result["proposed_statistics"].values())
    assert all(values["prediction_mean"] == 980.0 for values in result["fit_first_statistics"].values())


def test_aha_output_root_is_immutable(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.final5_aha16 import require_new_aha_output

    output = tmp_path / "aha"
    assert require_new_aha_output(output) == output
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        require_new_aha_output(output)
