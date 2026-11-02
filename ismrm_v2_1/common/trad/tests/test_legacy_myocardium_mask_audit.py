import numpy as np
import pytest

from evaluation_common.myocardium.legacy_masks import (
    classify_affine_relation,
    reorient_labels_exact,
    summarize_voxels,
)


def _classification(transform, source_shape=(4, 5, 6), reference_shape=(4, 5, 6)):
    return classify_affine_relation(np.asarray(transform, float), np.eye(4), source_shape, reference_shape)


def test_identity_affine_is_identical_grid():
    result = _classification(np.eye(4))

    assert result["geometry_class"] == "IDENTICAL_GRID"
    assert result["status"] == "PASS"


def test_exact_z_reversal_is_exact_reorientable_grid():
    source = np.eye(4); source[2, 2] = -1; source[2, 3] = 5

    result = _classification(source)

    assert result["geometry_class"] == "EXACT_REORIENTABLE_GRID"
    assert result["status"] == "PASS"
    assert result["corner_mapping_pass"] is True
    assert result["physical_coordinate_error_max"] == pytest.approx(0.0)


def test_exact_xy_permutation_is_exact_reorientable_grid():
    transform = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    result = _classification(transform, source_shape=(4, 5, 6), reference_shape=(5, 4, 6))

    assert result["geometry_class"] == "EXACT_REORIENTABLE_GRID"


@pytest.mark.parametrize(
    "transform",
    [
        np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        np.array([[1, 0, 0, 0.5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        np.diag([2, 1, 1, 1]),
        np.array([[1, 0.1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
        np.array([[0.8, -0.6, 0, 0], [0.6, 0.8, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
    ],
)
def test_non_equivalent_lattice_relations_are_rejected(transform):
    result = _classification(transform)

    assert result["geometry_class"] == "NON_EQUIVALENT_GRID"
    assert result["status"] == "STOP"


def test_exact_reorientation_preserves_labels_counts_physics_and_is_invertible():
    source = np.zeros((3, 4, 5), dtype=np.int16)
    source[0, 1, 0] = 3; source[2, 3, 4] = 6
    transform = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 4], [0, 0, 0, 1]], float)
    relation = _classification(transform, source_shape=source.shape, reference_shape=(4, 3, 5))

    reoriented = reorient_labels_exact(source, relation, (4, 3, 5))
    inverse_relation = classify_affine_relation(np.linalg.inv(transform), np.eye(4), reoriented.shape, source.shape)
    restored = reorient_labels_exact(reoriented, inverse_relation, source.shape)

    assert set(np.unique(reoriented)) == set(np.unique(source))
    assert np.count_nonzero(reoriented) == np.count_nonzero(source)
    assert np.array_equal(restored, source)



def test_summarize_voxels_uses_range_not_unique_values_for_continuous_reference():
    result = summarize_voxels(np.array([0.1, 0.2, 0.3], dtype=np.float32), is_label_map=False)

    assert "label_values" not in result
    assert result == {"finite_min": 0.10000000149011612, "finite_max": 0.30000001192092896}
