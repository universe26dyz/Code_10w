import numpy as np

from evaluation_common.myocardium.legacy_masks import classify_affine_relation, summarize_voxels


def test_classify_affine_relation_rejects_z_shift_even_when_shape_matches():
    reference = np.eye(4)
    source = np.eye(4)
    source[2, 3] = -13.0

    result = classify_affine_relation(source, reference)

    assert result["status"] == "STOP"
    assert result["reason"] == "geometry_header_mismatch"
    assert result["voxel_transform_source_to_reference"] == [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -13.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_summarize_voxels_uses_range_not_unique_values_for_continuous_reference():
    result = summarize_voxels(np.array([0.1, 0.2, 0.3], dtype=np.float32), is_label_map=False)

    assert "label_values" not in result
    assert result == {"finite_min": 0.10000000149011612, "finite_max": 0.30000001192092896}
