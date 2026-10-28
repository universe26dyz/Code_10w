"""Contracts for the Final5 original-native-grid formal evaluator."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def nifti() -> object:
    return pytest.importorskip("nibabel")


@pytest.mark.parametrize("parameter", ("t1", "t2"))
def test_raw_sparse_checkpoint_mask_uses_its_own_valid_native_geometry(nifti: object, parameter: str) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.final_native_grid_metrics import (
        _validate_sparse_checkpoint_mask,
        reproject_binary_mask_to_native_plane,
    )

    raw = nifti.Nifti1Image(np.array([[[1]], [[0]], [[1]]], dtype=np.uint8), np.diag((2.0, 1.0, 1.0, 1.0)))
    endpoint = nifti.Nifti1Image(np.ones((5, 1, 1)), np.eye(4))
    assert raw.shape != endpoint.shape and not np.allclose(raw.affine, endpoint.affine)
    _validate_sparse_checkpoint_mask(raw, parameter)
    mapped = reproject_binary_mask_to_native_plane(raw.get_fdata(), raw.affine, np.diag((2.0, 1.0, 1.0, 1.0)), (3, 1))
    assert np.array_equal(mapped[:, 0], np.array([True, False, True]))


def test_effective_query_mask_geometry_must_match_fit_first_endpoint(nifti: object) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.final_native_grid_metrics import _validate_effective_query_mask

    fit = nifti.Nifti1Image(np.ones((2, 2, 2)), np.eye(4))
    query = nifti.Nifti1Image(np.ones((3, 2, 2)), np.eye(4))
    with pytest.raises(ValueError, match="effective query mask geometry"):
        _validate_effective_query_mask(query, fit, "t1")


@pytest.mark.parametrize(
    "values, affine, message",
    [
        (np.array([[[2.0]]]), np.eye(4), "not binary"),
        (np.array([[[np.nan]]]), np.eye(4), "not finite"),
        (np.ones((1, 1, 1)), np.diag((1.0, 1.0, 0.0, 1.0)), "not invertible"),
    ],
)
def test_sparse_checkpoint_mask_invalid_data_or_affine_fails_closed(nifti: object, values: np.ndarray, affine: np.ndarray, message: str) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.final_native_grid_metrics import _validate_sparse_checkpoint_mask

    image = nifti.Nifti1Image(values, affine) if message != "not invertible" else SimpleNamespace(shape=values.shape, affine=affine, get_fdata=lambda: values)
    with pytest.raises(ValueError, match=message):
        _validate_sparse_checkpoint_mask(image, "t2")


def test_formal_regions_keep_disconnected_l100_sampling_support() -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.final_native_grid_metrics import native_grid_regions

    source = np.ones((4, 5), bool)
    reference = np.full((4, 5), 1000.0)
    l100 = np.zeros((4, 5), bool)
    l100[(0, 0, 1, 3), (0, 4, 1, 2)] = True
    sparse = np.zeros_like(l100)
    sparse[0, 0] = True
    regions = native_grid_regions(source, reference, l100, sparse)
    assert regions["l100_full"].sum() == 4
    assert regions["sparse_region"].sum() == 1
    assert regions["recovered_only"].sum() == 3
    assert regions["l100_full"][1, 1]  # isolated sampling-support pixels are retained


def test_formal_regions_can_use_manual_outer_segment_for_slice_domain() -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.final_native_grid_metrics import native_grid_regions

    source = np.ones((4, 5), bool)
    reference = np.full((4, 5), 1000.0)
    l100 = np.ones((4, 5), bool)
    sparse = np.zeros_like(l100)
    outer = np.zeros_like(l100)
    outer[1:3, 1:4] = True
    regions = native_grid_regions(source, reference, l100, sparse, manual_outer_region=outer)
    assert regions["l100_full"].sum() == int(outer.sum())
    assert regions["l100_full"][1, 1]
    assert not regions["l100_full"][0, 0]


def test_exact_paired_support_is_shared_by_both_methods() -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.final_native_grid_metrics import exact_paired_support

    region = np.array([[True, True, True, True]])
    reference = np.array([[1.0, 1.0, 1.0, np.nan]])
    proposed = np.array([[1.0, np.nan, 1.0, 1.0]])
    fit = np.array([[1.0, 1.0, np.nan, 1.0]])
    support = exact_paired_support(region, reference, proposed, fit, np.ones_like(region), np.ones_like(region))
    assert np.array_equal(support, np.array([[True, False, False, False]]))


def test_fit_first_effective_query_mask_defines_l100_region_once() -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.final_native_grid_metrics import l100_region_mask, mask_overlap_counts

    query = np.array([[True, False, True]])
    assert np.array_equal(l100_region_mask(query), query)
    assert mask_overlap_counts(query, np.array([[True, True, False]])) == {
        "intersection_voxels": 1, "union_voxels": 3, "xor_voxels": 2, "iou": 1 / 3,
    }


def test_native_affine_uses_historical_lps_to_ras_conversion() -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.final_native_grid_metrics import native_ras_affine

    lps = np.eye(4); lps[:3, 3] = (2, 3, 4)
    affine = native_ras_affine(lps)
    assert np.array_equal(affine, np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps)


def test_formal_metric_evaluator_does_not_depend_on_publication_renderer() -> None:
    from pathlib import Path
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_native_grid_metrics as metrics

    source = Path(metrics.__file__).read_text(encoding="utf-8")
    assert "final5_publication_visualization" not in source
    assert "reproject_volume_to_native_plane" in source
    assert "exact_paired_support" in source


def test_l100_uses_all_native_groups_and_sparse_uses_manifest_heldouts() -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.final_native_grid_metrics import evaluation_group_ids

    all_groups = {"sax": np.arange(12), "2ch": np.arange(15), "4ch": np.arange(10)}
    assert evaluation_group_ids("L100", all_groups, {"sax": [], "2ch": [], "4ch": []}) == {"sax": list(range(12)), "2ch": list(range(15)), "4ch": list(range(10))}
    heldout = {"sax": [1, 3], "2ch": [0, 5], "4ch": [2]}
    assert evaluation_group_ids("SAX100_2CH0_4CH100", all_groups, heldout) == heldout
    with pytest.raises(ValueError, match="held-out"):
        evaluation_group_ids("L50_fold0", all_groups, {"sax": [], "2ch": [], "4ch": []})


def test_formal_output_is_immutable(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.final_native_grid_metrics import require_new_output

    output = tmp_path / "final_native_grid"; output.mkdir()
    with pytest.raises(FileExistsError):
        require_new_output(output)


def test_batch_formal_metric_has_no_current_component_mask_dependency(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final5_batch_runner as runner

    l100 = runner.CaseRecord("CYJ", "L100", "L100", None, 1, tmp_path / "l100.json", "a" * 64, "b" * 40, tmp_path)
    sparse = runner.CaseRecord("CYJ", "SAX100_2CH0_4CH0", "case", None, None, tmp_path / "sparse.json", "c" * 64, "b" * 40, tmp_path)
    records = {("CYJ", "L100"): l100, ("CYJ", sparse.case_id): sparse}
    assert {stage for stage, _record in runner._stage_dependencies("final_native_grid_metrics", l100, records)} == {"proposed_endpoint", "fit_first_endpoint"}
    assert {stage for stage, _record in runner._stage_dependencies("final_native_grid_metrics", sparse, records)} == {"proposed_endpoint", "fit_first_endpoint"}
