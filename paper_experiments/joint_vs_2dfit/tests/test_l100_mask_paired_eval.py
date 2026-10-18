"""CPU contracts for the standalone L100-mask paired post-hoc audit."""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.phase2_clean import l100_mask_paired_eval as audit


def test_binary_masks_reproject_with_nearest_neighbor() -> None:
    mask = np.array([[[1]], [[0]]], dtype=np.uint8)
    affine = np.eye(4); affine[0, 0] = 2.0
    native = np.eye(4)
    result = audit.reproject_binary_mask_to_native_plane(mask, affine, native, (3, 1))
    assert result[:, 0].tolist() == [True, False, False]
    source = Path(audit.__file__).read_text(encoding="utf-8").split("def reproject_binary_mask_to_native_plane", 1)[1].split("def native_support_regions", 1)[0]
    assert "order=0" in source


def test_predictions_keep_existing_linear_native_reprojection() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert "reproject_volume_to_native_plane" in source
    assert "order=0" not in source.split("proposed, p_valid = reproject_volume_to_native_plane", 1)[1].split("regions =", 1)[0]


def test_mask_defined_regions_never_use_prediction_fov() -> None:
    source = np.array([[True, True, True, False]])
    reference = np.array([[1.0, 2.0, 3.0, 4.0]])
    l100 = np.array([[True, True, False, True]])
    sparse = np.array([[True, False, True, True]])
    regions = audit.native_support_regions(source, reference, l100, sparse)
    assert regions["l100_full"].tolist() == [[True, True, False, False]]
    assert regions["recovered_only"].tolist() == [[False, True, False, False]]
    assert regions["sparse_region"].tolist() == [[True, False, True, False]]
    assert "fixed_valid & ~default_valid" not in Path(audit.__file__).read_text(encoding="utf-8")


def test_paired_support_is_exactly_shared_by_both_methods() -> None:
    region = np.array([[True, True, True]])
    proposed = np.array([[1.0, np.nan, 3.0]])
    fit = np.array([[1.0, 2.0, 3.0]])
    paired = audit.paired_support(region, proposed, fit, np.array([[True, True, False]]), np.ones((1, 3), bool))
    assert paired.tolist() == [[True, False, False]]
    p = audit.metrics_on_exact_support(np.array([[1., 2., 3.]]), proposed, paired, 3)
    f = audit.metrics_on_exact_support(np.array([[1., 2., 3.]]), fit, paired, 3)
    assert p["voxel_count"] == f["voxel_count"] == 1
    assert p["coverage_fraction"] == f["coverage_fraction"] == pytest.approx(1 / 3)


def test_pooled_metrics_are_voxel_pooled_not_plane_averaged() -> None:
    first = audit.metrics_on_exact_support(np.array([0.]), np.array([10.]), np.array([True]), 1)
    second = audit.metrics_on_exact_support(np.array([0., 0., 0.]), np.array([0., 0., 0.]), np.ones(3, bool), 3)
    pooled = audit._pooled_metrics([np.array([0.]), np.array([0., 0., 0.])], [np.array([10.]), np.array([0., 0., 0.])], 4)
    assert (first["rmse"] + second["rmse"]) / 2 == pytest.approx(5.0)
    assert pooled["rmse"] == pytest.approx(5.0)


def test_mask_coverage_uses_masks_not_prediction_validity() -> None:
    native = np.array([[True, True, True]])
    l100 = np.array([[True, True, False]])
    sparse = np.array([[True, False, True]])
    result = audit.mask_coverage(native, l100, sparse)
    assert result == {"native_source_voxels": 3, "l100_native_mask_voxels_within_source": 2,
                      "sparse_native_mask_voxels_within_source": 2, "recovered_only_voxels": 1,
                      "sparse_fraction_of_l100": .5, "l100_coverage_fraction_of_native": pytest.approx(2 / 3),
                      "sparse_not_l100_boundary_discrepancy_voxels": 1}


def test_subject_mask_coverage_is_pooled_not_plane_averaged() -> None:
    pooled = audit.pooled_mask_coverage([
        {"native_source_voxels": 1, "l100_native_mask_voxels_within_source": 1, "sparse_native_mask_voxels_within_source": 1, "recovered_only_voxels": 0, "sparse_not_l100_boundary_discrepancy_voxels": 0},
        {"native_source_voxels": 3, "l100_native_mask_voxels_within_source": 0, "sparse_native_mask_voxels_within_source": 0, "recovered_only_voxels": 0, "sparse_not_l100_boundary_discrepancy_voxels": 0},
    ])
    assert pooled["l100_coverage_fraction_of_native"] == pytest.approx(.25)


def test_recovered_distribution_is_on_the_paired_recovered_region() -> None:
    values = np.array([[0., 3., 8.]])
    support = np.array([[True, False, True]])
    result = audit.recovered_distribution(values, support)
    assert result["zero_fraction"] == .5
    assert result["median_prediction_ms"] == 4.0


def test_formal_proposed_resolver_selects_the_matching_export(tmp_path: Path) -> None:
    root = tmp_path / "real" / "R1" / "proposed" / "CYJ" / "L25_fold0"; root.mkdir(parents=True)
    for parameter in ("T1_3D.nii.gz", "T2_3D.nii.gz"):
        nib.save(nib.Nifti1Image(np.ones((1, 1, 1)), np.eye(4)), root / parameter)
    paths, provenance = audit.resolve_formal_proposed_volumes(tmp_path / "real", "R1", "CYJ", "L25_fold0")
    assert paths["t1"].name == "T1_3D.nii.gz" and paths["t2"].name == "T2_3D.nii.gz"
    assert provenance["selection"] == "matching_quantitative_export"
    assert "evaluate_real_sparse_pilot.sh" in provenance["selection_basis"]


def test_visualization_contract_has_common_l100_display_and_sparse_overlay() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")
    renderer = source.split("def render_plane_panels", 1)[1].split("def render_summary_triptych", 1)[0]
    assert "_masked(values, display_mask)" in renderer
    assert "linestyles=\"dashed\"" in renderer
    assert "magma" in renderer
    assert "_crop_bounds" not in source


def test_posthoc_module_does_not_train_refit_or_reconstruct() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert '"no_training": True' in source and '"no_refitting": True' in source and '"no_reconstruction": True' in source
    assert "nesvor reconstruct" not in source and "train(" not in source
