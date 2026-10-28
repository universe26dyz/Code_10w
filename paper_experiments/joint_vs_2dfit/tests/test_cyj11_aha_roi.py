from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest


def test_scmr_exact_crop_admits_cyj_affine_serialization_drift() -> None:
    """CYJ's 1.76e-5 native-index drift is an exact SCMR integer crop."""
    from evaluation_common.myocardium.legacy_masks import exact_native_plane_crop

    transform = np.array([
        [0.0, 0.99999994, 0.00000010, 62.9999974],
        [1.0, -0.00000003, -0.00000030, 70.9999914],
        [0.0, -0.00000002, 0.99999994, 13.0000007],
        [0.0, 0.0, 0.0, 1.0],
    ])
    x, y, z, evidence = exact_native_plane_crop(
        source_affine=np.eye(4), target_affine=transform,
        source_shape=(256, 288, 14), target_shape=(145, 129),
    )

    assert (x[0, 0], y[0, 0], z[0, 0]) == (63, 71, 13)
    assert (x[-1, -1], y[-1, -1], z[-1, -1]) == (191, 215, 13)
    assert evidence["integer_transform"] == [[0, 1, 0, 63], [1, 0, 0, 71], [0, 0, 1, 13], [0, 0, 0, 1]]


def test_scmr_exact_crop_rejects_genuine_fractional_native_origin() -> None:
    from evaluation_common.myocardium.legacy_masks import exact_native_plane_crop

    transform = np.array([[0.0, 1.0, 0.0, 63.01], [1.0, 0.0, 0.0, 71.0],
                          [0.0, 0.0, 1.0, 13.0], [0.0, 0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="integer"):
        exact_native_plane_crop(source_affine=np.eye(4), target_affine=transform,
                                source_shape=(256, 288, 14), target_shape=(145, 129))


def test_cyj11_mapping_rejects_non_discrete_transform(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.cyj11_adapter import require_cyj11_mapping
    from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256

    prepared = tmp_path / "observations.npz"
    np.savez_compressed(prepared, group_idx=np.repeat([2, 4, 6], 10), weight_idx=np.tile(np.arange(10), 3))
    bundle = tmp_path / "roi_derivation_manifest.json"
    bundle.write_text(json.dumps({"schema": "cyj11_native_roi_bundle/v1", "subject": "DYZ", "planes": [
        {"native_group_idx": 2, "roi_slice_index": 0}, {"native_group_idx": 4, "roi_slice_index": 1}, {"native_group_idx": 6, "roi_slice_index": 2},
    ]}))
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({
        "schema": "cyj11_aha_native_roi_mapping/v1", "subject": "DYZ", "case": "L100_fold0",
        "prepared_sax": {"path": str(prepared), "sha256": sha256(prepared)},
        "roi_bundle_manifest": {"path": str(bundle), "sha256": sha256(bundle)},
        "planes": [
            {"level": "basal", "native_group_idx": 2, "roi_slice_index": 0, "roi_transform": "bilinear", "anterior_xy": [1, 2], "inferior_xy": [3, 4]},
            {"level": "mid", "native_group_idx": 4, "roi_slice_index": 1, "roi_transform": "identity", "anterior_xy": [1, 2], "inferior_xy": [3, 4]},
            {"level": "apical", "native_group_idx": 6, "roi_slice_index": 2, "roi_transform": "identity", "anterior_xy": [1, 2], "inferior_xy": [3, 4]},
        ],
    }))

    with pytest.raises(ValueError, match="discrete"):
        require_cyj11_mapping(mapping, subject="DYZ", case="L100_fold0")


def test_cyj11_mapping_binds_explicit_group_to_prepared_observations(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.cyj11_adapter import require_cyj11_mapping
    from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256

    prepared = tmp_path / "observations.npz"
    np.savez_compressed(prepared, group_idx=np.repeat([2, 4, 6], 10), weight_idx=np.tile(np.arange(10), 3))
    bundle = tmp_path / "roi_derivation_manifest.json"
    bundle.write_text(json.dumps({"schema": "cyj11_native_roi_bundle/v1", "subject": "DYZ", "planes": [
        {"native_group_idx": 2, "roi_slice_index": 0}, {"native_group_idx": 4, "roi_slice_index": 1}, {"native_group_idx": 6, "roi_slice_index": 2},
    ]}))
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({
        "schema": "cyj11_aha_native_roi_mapping/v1", "subject": "DYZ", "case": "L100_fold0",
        "prepared_sax": {"path": str(prepared), "sha256": sha256(prepared)},
        "roi_bundle_manifest": {"path": str(bundle), "sha256": sha256(bundle)},
        "planes": [
            {"level": "basal", "native_group_idx": 9, "roi_slice_index": 0, "roi_transform": "identity", "anterior_xy": [1, 2], "inferior_xy": [3, 4]},
            {"level": "mid", "native_group_idx": 4, "roi_slice_index": 1, "roi_transform": "identity", "anterior_xy": [1, 2], "inferior_xy": [3, 4]},
            {"level": "apical", "native_group_idx": 6, "roi_slice_index": 2, "roi_transform": "identity", "anterior_xy": [1, 2], "inferior_xy": [3, 4]},
        ],
    }))

    with pytest.raises(ValueError, match="absent"):
        require_cyj11_mapping(mapping, subject="DYZ", case="L100_fold0")


def test_cyj11_derives_historical_aha_mapping_from_six_landmark_labels(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.cyj11_adapter import derive_cyj11_historical_mapping
    from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256

    prepared = tmp_path / "observations.npz"
    np.savez_compressed(prepared, group_idx=np.repeat([5, 7, 9], 10), weight_idx=np.tile(np.arange(10), 3))
    derived = tmp_path / "derived"; derived.mkdir()
    derivation = derived / "roi_derivation_manifest.json"
    derivation.write_text(json.dumps({"schema": "cyj11_native_roi_bundle/v1", "subject": "CYJ", "planes": [
        {"native_group_idx": 5, "roi_slice_index": 8}, {"native_group_idx": 7, "roi_slice_index": 6},
        {"native_group_idx": 9, "roi_slice_index": 4},
    ]}))
    points = np.zeros((3, 7, 7), np.uint8)
    points[0, 2, 5], points[0, 2, 1] = 5, 6
    points[1, 3, 5], points[1, 3, 1] = 1, 2
    points[2, 4, 5], points[2, 4, 1] = 3, 4
    (derived / "manifest.json").write_text(json.dumps({"schema": "exact_native_myocardium_bundle/v2", "status": "PASS"}))
    np.savez_compressed(derived / "sax_myocardium_masks.npz", group_idx=np.array([5, 7, 9]), landmark_labels=points)

    mapping = derive_cyj11_historical_mapping(subject="CYJ", case="L100_fold0", prepared_sax=prepared,
                                               roi_bundle=derived, output=tmp_path / "mapping.json")
    payload = json.loads(mapping.read_text())
    assert [(entry["level"], entry["native_group_idx"], entry["roi_slice_index"])
            for entry in payload["planes"]] == [("basal", 5, 8), ("mid", 7, 6), ("apical", 9, 4)]
    assert payload["prepared_sax"]["sha256"] == sha256(prepared)


def test_cyj11_aha_and_roi_use_one_common_support() -> None:
    from paper_experiments.joint_vs_2dfit.aha.cyj11_aha16 import build_paired_aha16_plane
    from paper_experiments.joint_vs_2dfit.aha.cyj11_myocardial_roi_metrics import paired_roi_statistics

    anatomy = np.ones((9, 9), bool)
    reference = np.full((9, 9), 1000.0)
    proposed = reference + 10.0
    fit_first = reference - 20.0
    proposed[0, 0] = np.nan
    fit_first[0, 1] = np.nan
    aha = build_paired_aha16_plane(
        level="basal", reference=reference, proposed=proposed, fit_first=fit_first,
        proposed_valid=np.ones_like(anatomy), fit_first_valid=np.ones_like(anatomy),
        myocardium_core=anatomy, anterior_xy=(8, 4), inferior_xy=(0, 4),
    )
    roi = paired_roi_statistics(reference=reference, proposed=proposed, fit_first=fit_first,
                                proposed_valid=np.ones_like(anatomy), fit_first_valid=np.ones_like(anatomy), roi=anatomy)

    assert np.array_equal(aha["proposed_support"], aha["fit_first_support"])
    assert aha["proposed_support"].sum() == roi["common_support_voxels"] == 79


def test_cyj11_final_pose_plane_reads_current_native_bundles(tmp_path) -> None:
    """Final-pose AHA reads current bundle pixels, never endpoint reprojections."""
    from paper_experiments.joint_vs_2dfit.aha.cyj11_runner import (
        final_pose_sax_plane,
        load_cyj11_final_pose_bundles,
    )

    from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256

    direct, fit = tmp_path / "direct", tmp_path / "fit"
    (direct / "planes" / "sax").mkdir(parents=True)
    (fit / "planes" / "t1" / "sax").mkdir(parents=True)
    (fit / "planes" / "t2" / "sax").mkdir(parents=True)
    np.savez_compressed(direct / "planes" / "sax" / "group005.npz",
                        t1_ms=np.full((2, 3), 1100.0), t2_ms=np.full((2, 3), 55.0))
    for parameter, value in (("t1", 1000.0), ("t2", 50.0)):
        np.savez_compressed(fit / "planes" / parameter / "sax" / "group005.npz",
                            prediction_ms=np.full((2, 3), value),
                            volume_geometry_valid=np.ones((2, 3), bool),
                            query_effective_mask=np.ones((2, 3), bool))
    (direct / "manifest.json").write_text(json.dumps({
        "schema": "direct_sparse_native_prediction_bundle/v1", "subject": "CYJ", "case_id": "L50_fold0",
        "endpoint": "central", "planes": [{"stack": "sax", "original_group_idx": 5, "path": "planes/sax/group005.npz"}],
    }))
    endpoint = tmp_path / "fit_endpoint_manifest.json"
    endpoint.write_text(json.dumps({"subject": "CYJ", "condition": "L50_fold0", "endpoint": "central"}))
    (fit / "manifest.json").write_text(json.dumps({
        "schema": "fit_first_native_prediction_bundle/v6", "endpoint": "central",
        "endpoint_source_manifest": {"path": str(endpoint), "sha256": sha256(endpoint)},
    }))

    bundles = load_cyj11_final_pose_bundles(direct, fit, subject="CYJ", case="L50_fold0")
    native = {"group_idx": np.array([5]), "t1_ms": np.full((1, 2, 3), 1200.0),
              "t2_ms": np.full((1, 2, 3), 60.0)}
    plane = final_pose_sax_plane(native=native, group=5, parameter="t1", **bundles)

    assert np.all(plane["reference"] == 1200.0)
    assert np.all(plane["proposed"] == 1100.0)
    assert np.all(plane["fit_first"] == 1000.0)
    assert plane["proposed_valid"].all() and plane["fit_first_valid"].all()


def test_cyj11_final_pose_accepts_raw_current_fit_first_bundle(tmp_path) -> None:
    """HHZ raw NeSVoR volumes use their own final slices without a fabricated endpoint mask."""
    from paper_experiments.joint_vs_2dfit.aha.cyj11_runner import (
        final_pose_sax_plane,
        load_cyj11_final_pose_bundles,
    )

    direct, fit = tmp_path / "direct", tmp_path / "fit"
    (direct / "planes" / "sax").mkdir(parents=True)
    (fit / "planes" / "t1" / "sax").mkdir(parents=True)
    np.savez_compressed(direct / "planes" / "sax" / "group005.npz",
                        t1_ms=np.full((2, 3), 1100.0), t2_ms=np.full((2, 3), 55.0))
    np.savez_compressed(fit / "planes" / "t1" / "sax" / "group005.npz",
                        prediction_ms=np.full((2, 3), 1000.0),
                        volume_geometry_valid=np.asarray([[True, True, False], [True, True, True]]),
                        query_effective_mask=np.ones((2, 3), bool),
                        valid=np.asarray([[True, True, False], [True, True, True]]))
    (direct / "manifest.json").write_text(json.dumps({
        "schema": "direct_sparse_native_prediction_bundle/v1", "subject": "HHZ", "case_id": "L50_fold0",
        "endpoint": "central", "planes": [{"stack": "sax", "original_group_idx": 5, "path": "planes/sax/group005.npz"}],
    }))
    (fit / "manifest.json").write_text(json.dumps({
        "schema": "fit_first_raw_final_pose_native_bundle/v1", "subject": "HHZ", "case": "L50_fold0",
        "endpoint": "central", "support_policy": "volume_geometry_and_finite_prediction_only",
    }))

    bundles = load_cyj11_final_pose_bundles(direct, fit, subject="HHZ", case="L50_fold0")
    native = {"group_idx": np.array([5]), "t1_ms": np.full((1, 2, 3), 1200.0),
              "t2_ms": np.full((1, 2, 3), 60.0)}
    plane = final_pose_sax_plane(native=native, group=5, parameter="t1", **bundles)

    assert plane["fit_first_valid"].sum() == 5


def test_current_fit_first_final_pose_requires_current_case_manifest(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.current_fit_first_final_pose_bundle import _input_manifest

    (tmp_path / "input_manifest.json").write_text(json.dumps({
        "method": "fit_first", "subject_id": "HHZ", "case_id": "L25_fold0",
    }))

    assert _input_manifest(tmp_path, subject="HHZ", case="L25_fold0") == tmp_path / "input_manifest.json"
    with pytest.raises(ValueError, match="case mismatch"):
        _input_manifest(tmp_path, subject="HHZ", case="L50_fold0")


def test_cyj11_r3_final_slices_follow_current_manifest_group_order(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.cyj11_r3_fit_first_final_pose_bundle import r3_group_records

    root, source = tmp_path / "fit", tmp_path / "source_slices"
    for stack, groups in (("sax", [3, 2]), ("2ch", [4]), ("4ch", [5, 1])):
        folder = root / "nesvor_inputs" / stack
        folder.mkdir(parents=True)
        (folder / "manifest.json").write_text(json.dumps({"group_idx": groups}))
    source.mkdir()
    for index in range(5):
        (source / f"{index}.nii.gz").write_bytes(str(index).encode())
    result = root / "nesvor" / "t1" / "result.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({"registration": "stack", "no_transformation_optimization": False,
                                  "no_slice_scale": True, "output_slices": str(source),
                                  "input_stacks": ["/old/sax/t1/stack.nii.gz", "/old/2ch/t1/stack.nii.gz", "/old/4ch/t1/stack.nii.gz"]}))

    records = r3_group_records(root, "t1")

    assert [(item["stack"], item["original_group_idx"], item["output_slice_index"])
            for item in records] == [("sax", 3, 0), ("sax", 2, 1), ("2ch", 4, 2), ("4ch", 5, 3), ("4ch", 1, 4)]


def test_cyj11_r3_endpoint_accepts_migrated_v1_local_copy(tmp_path) -> None:
    """The r3 v1 manifest retains old paths, so the local copy is hash-bound."""
    from paper_experiments.joint_vs_2dfit.aha.cyj11_r3_fit_first_final_pose_bundle import (
        r3_effective_query_mask,
        r3_endpoint_volume,
    )
    from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256

    root = tmp_path / "endpoint"
    records: dict[str, dict[str, dict[str, str]]] = {}
    for parameter in ("t1", "t2"):
        folder = root / parameter; folder.mkdir(parents=True)
        volume, mask = folder / "volume_ms.nii.gz", folder / "query_effective_mask.nii.gz"
        volume.write_bytes((parameter + "-volume").encode())
        mask.write_bytes((parameter + "-mask").encode())
        records[parameter] = {
            "output_volume_ms": {"path": f"/old/{parameter}/volume_ms.nii.gz", "sha256": sha256(volume)},
            "effective_query_mask": {"path": f"/old/{parameter}/query_effective_mask.nii.gz", "sha256": sha256(mask)},
        }
    (root / "manifest.json").write_text(json.dumps({
        "schema": "phase4r_fixed_l100_fit_first_export_bundle/v1", "parameters": records,
    }))

    assert r3_endpoint_volume(root, "t1")[0] == root / "t1" / "volume_ms.nii.gz"
    assert r3_effective_query_mask(root, "t2")[0] == root / "t2" / "query_effective_mask.nii.gz"


def test_cyj11_r3_direct_contract_accepts_migrated_profile_by_digest(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.cyj11_r3_direct_final_pose_bundle import r3_sparse_prepared_contract
    from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256

    prepared = tmp_path / "migrated"
    stacks = {}
    for stack, original_groups in (("sax", [4, 2]), ("2ch", [3]), ("4ch", [8])):
        path = prepared / stack / "observations.npz"; path.parent.mkdir(parents=True)
        np.savez_compressed(path, group_idx=np.repeat(np.arange(len(original_groups)), 2), images=np.zeros((2 * len(original_groups), 3, 4)), pixel_spacing_rc_mm=np.ones((2 * len(original_groups), 2)))
        stacks[stack] = {
            "training_stack_present": True,
            "derived_observation": {"path": f"/old/{stack}/observations.npz", "sha256": sha256(path)},
            "derived_to_original_group_idx": [
                {"derived_group_idx": index, "original_group_idx": group}
                for index, group in enumerate(original_groups)
            ],
        }
    manifest = tmp_path / "sparse_manifest.json"
    manifest.write_text(json.dumps({"schema_version": "phase4r_sparse_profile_v2", "status": "ELIGIBLE",
                                    "subject": "CYJ", "case_id": "case", "stacks": stacks}))

    _payload, paths, mappings = r3_sparse_prepared_contract(
        prepared_root=prepared, sparse_manifest=manifest, subject="CYJ", case="case")

    assert [path.parent.name for path in paths] == ["sax", "2ch", "4ch"]
    assert mappings["sax"] == [(0, 4), (1, 2)]


def test_cyj11_r3_direct_contract_accepts_legacy_l50_by_declared_mapping(tmp_path) -> None:
    """Legacy L50 observations are renumbered; the manifest is the identity authority."""
    from paper_experiments.joint_vs_2dfit.aha.cyj11_r3_direct_final_pose_bundle import r3_sparse_prepared_contract
    from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256

    prepared = tmp_path / "legacy"
    stacks = {}
    for stack, original_groups in (("sax", [11, 10]), ("2ch", [15, 13]), ("4ch", [9, 7])):
        path = prepared / stack / "observations.npz"; path.parent.mkdir(parents=True)
        np.savez_compressed(path, group_idx=np.repeat(np.arange(len(original_groups)), 2),
                            images=np.zeros((2 * len(original_groups), 3, 4)),
                            pixel_spacing_rc_mm=np.ones((2 * len(original_groups), 2)))
        stacks[stack] = {
            "source_observation": {"path": f"/old/{stack}/observations.npz", "sha256": "source"},
            "derived_observation": {"path": f"/old/{stack}/derived.npz", "sha256": sha256(path)},
            "retained_original_group_ids": original_groups,
            "derived_to_original_group_idx": [
                {"derived_group_idx": index, "original_group_idx": group}
                for index, group in enumerate(original_groups)
            ],
        }
    manifest = tmp_path / "sparse_manifest.json"
    manifest.write_text(json.dumps({"schema_version": "phase4r_sparse_v1", "status": "ELIGIBLE",
                                    "subject": "HHZ", "condition": "L50", "phase": 0,
                                    "stacks": stacks}))

    _payload, paths, mappings = r3_sparse_prepared_contract(
        prepared_root=prepared, sparse_manifest=manifest, subject="HHZ", case="L50_fold0")

    assert [path.parent.name for path in paths] == ["sax", "2ch", "4ch"]
    assert mappings["sax"] == [(0, 11), (1, 10)]
    assert mappings["2ch"] == [(0, 15), (1, 13)]


def test_cyj11_r3_direct_contract_maps_legacy_l100_derived_groups(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.cyj11_r3_direct_final_pose_bundle import r3_sparse_prepared_contract
    from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256

    prepared = tmp_path / "legacy"
    stacks = {}
    for stack, original_groups in (("sax", [11, 10]), ("2ch", [3]), ("4ch", [2])):
        path = prepared / stack / "observations.npz"; path.parent.mkdir(parents=True)
        np.savez_compressed(path, group_idx=np.repeat(np.arange(len(original_groups)), 2),
                            images=np.zeros((2 * len(original_groups), 3, 4)),
                            pixel_spacing_rc_mm=np.ones((2 * len(original_groups), 2)))
        stacks[stack] = {
            "source_observation": {"path": f"/old/{stack}/source.npz", "sha256": "source"},
            "derived_observation": {"path": f"/old/{stack}/derived.npz", "sha256": sha256(path)},
            "retained_original_group_ids": original_groups,
        }
    manifest = tmp_path / "sparse_manifest.json"
    manifest.write_text(json.dumps({"schema_version": "phase4r_sparse_v1", "status": "ELIGIBLE",
                                    "subject": "HHZ", "condition": "L100", "phase": 0,
                                    "stacks": stacks}))

    _payload, _paths, mappings = r3_sparse_prepared_contract(
        prepared_root=prepared, sparse_manifest=manifest, subject="HHZ", case="L100_fold0")

    assert mappings["sax"] == [(0, 11), (1, 10)]


def test_current_fit_first_mapping_uses_sparse_manifest_original_groups(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.current_fit_first_final_pose_bundle import _sparse_group_mapping
    sparse = tmp_path / "sparse_manifest.json"
    sparse.write_text(json.dumps({
        "schema_version": "phase4r_sparse_v1",
        "stacks": {
            "sax": {"derived_observation": {}, "retained_original_group_ids": [11, 10]},
            "2ch": {"derived_observation": {}, "retained_original_group_ids": [15]},
            "4ch": {"derived_observation": {}, "retained_original_group_ids": [9]},
        },
    }))
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "input_manifest.json").write_text(json.dumps({
        "method": "fit_first", "sparse_manifest": {"path": str(sparse)},
    }))
    assert _sparse_group_mapping(baseline) == {
        "sax": {0: 11, 1: 10}, "2ch": {0: 15}, "4ch": {0: 9},
    }


def test_current_fit_first_mapping_allows_legacy_full_l100_identity(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.current_fit_first_final_pose_bundle import _sparse_group_mapping
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "input_manifest.json").write_text(json.dumps({
        "method": "fit_first", "case_id": "L100_fold0",
    }))
    assert _sparse_group_mapping(baseline, case="L100_fold0") == {"sax": {}, "2ch": {}, "4ch": {}}


def test_paired_through_plane_staging_creates_figure_directory(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.paired_final_pose_through_plane import _figure_dir
    staging = tmp_path / ".figures.staging"
    target = _figure_dir(staging)
    assert target == staging / "figures"
    assert target.is_dir()


def test_through_plane_sampling_applies_one_shared_native_mask() -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.visualization import sample_same_world_through_plane

    native = np.arange(27, dtype=float).reshape(3, 3, 3)
    support = np.zeros((3, 3, 3), bool)
    support[:, 1, :] = True
    result = sample_same_world_through_plane(
        native, np.eye(4), native + 100, np.eye(4), native + 200, np.eye(4),
        cut_column=1, common_support=support,
    )

    assert result["common_support"].shape == result["native"].shape
    assert result["common_support"].all()


def test_historical_scmr_aha_validation_requires_all_supported_segments() -> None:
    from paper_experiments.joint_vs_2dfit.aha.legacy_aha16 import validate_aha16_statistics

    rows = [{"parameter": parameter, "segment": segment, "n": 10}
            for parameter in ("t1", "t2") for segment in range(1, 17)]
    assert validate_aha16_statistics(rows)["status"] == "PASS"
    rows[0]["n"] = 9
    assert validate_aha16_statistics(rows)["status"] == "FAIL"


def test_historical_scmr_figure3_adapter_writes_two_by_two_bullseye(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.legacy_aha16 import render_scmr_figure3

    values = {parameter: {segment: float(segment) for segment in range(1, 17)}
              for parameter in ("t1", "t2")}
    output = render_scmr_figure3(tmp_path / "figure3.png", reference=values, prediction=values,
                                 subject="CYJ", case="L100_fold0", method="proposed")
    assert output.is_file() and output.with_suffix(".pdf").is_file()


def test_cyj11_native_sax_requires_manifest_bound_to_prepared_input(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.cyj11_adapter import load_cyj11_native_sax
    from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256

    prepared = tmp_path / "prepared.npz"
    np.savez_compressed(prepared, group_idx=np.repeat([3], 10), weight_idx=np.arange(10), images=np.zeros((10, 2, 3)))
    cache = tmp_path / "cache"; cache.mkdir()
    np.savez_compressed(cache / "native_maps.npz", group_idx=np.array([3]), t1_ms=np.zeros((1, 2, 3)), t2_ms=np.zeros((1, 2, 3)), b1=np.zeros((1, 2, 3)))
    np.savez_compressed(cache / "geometry.npz", affine_lps_rc=np.eye(4)[None], pixel_spacing_rc_mm=np.ones((1, 2)), slice_thickness_mm=np.ones(1))
    (cache / "manifest.json").write_text(json.dumps({
        "source_observations": {"path": str(prepared), "sha256": sha256(prepared)},
        "native_maps_sha256": sha256(cache / "native_maps.npz"), "geometry_sha256": sha256(cache / "geometry.npz"),
    }))

    native = load_cyj11_native_sax(cache, prepared)
    assert native["group_idx"].tolist() == [3]


def test_cyj11_native_sax_accepts_historical_unsealed_dictionary_manifest(tmp_path) -> None:
    """Unified imported CYJ cache predates per-file digest fields but binds its prepared SAX source."""
    from paper_experiments.joint_vs_2dfit.aha.cyj11_adapter import load_cyj11_native_sax
    from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import BASELINE_POSE_POLICY, sha256
    from paper_experiments.joint_vs_2dfit.phase2_clean.fit_first import NATIVE_FIT_METHOD

    prepared = tmp_path / "prepared.npz"
    np.savez_compressed(prepared, group_idx=np.repeat([3], 10), weight_idx=np.arange(10), images=np.zeros((10, 2, 3)))
    cache = tmp_path / "cache"; cache.mkdir()
    np.savez_compressed(cache / "native_maps.npz", group_idx=np.array([3]), t1_ms=np.zeros((1, 2, 3)), t2_ms=np.zeros((1, 2, 3)), b1=np.zeros((1, 2, 3)))
    np.savez_compressed(cache / "geometry.npz", affine_lps_rc=np.eye(4)[None], pixel_spacing_rc_mm=np.ones((1, 2)), slice_thickness_mm=np.ones(1))
    (cache / "manifest.json").write_text(json.dumps({
        "method": NATIVE_FIT_METHOD,
        "source_observations": {"path": str(prepared), "sha256": sha256(prepared)},
        "dictionary_cache_entries": 1,
        "baseline_pose_policy": BASELINE_POSE_POLICY,
        "next_formal_step": "independent registration",
    }))

    native = load_cyj11_native_sax(cache, prepared)
    assert native["group_idx"].tolist() == [3]


def test_cyj11_runner_uses_the_unified_native_maps_cache_layout(tmp_path, monkeypatch) -> None:
    """CYJ historical and r2/r3 imports share native_cache/<subject>/native_maps/<stack>."""
    from paper_experiments.joint_vs_2dfit.aha import cyj11_runner

    captured: dict[str, object] = {}
    monkeypatch.setattr(cyj11_runner, "load_cyj11_native_sax", lambda cache, prepared: captured.update(cache=cache, prepared=prepared) or {})
    monkeypatch.setattr(cyj11_runner, "load_cyj11_endpoint_images", lambda *args: {})

    cyj11_runner.load_cyj11_run_inputs(tmp_path, "CYJ", "L100_fold0")

    assert captured["cache"] == tmp_path / "native_cache" / "CYJ" / "native_maps" / "sax"
    assert captured["prepared"] == tmp_path / "prepared" / "CYJ" / "sax" / "observations.npz"


def test_cyj11_native_stack_root_accepts_direct_subject_cache_layout(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.cyj11_adapter import cyj11_native_stack_root

    direct = tmp_path / "native_cache" / "DYZ" / "sax"
    direct.mkdir(parents=True)
    for name in ("native_maps.npz", "geometry.npz", "manifest.json"):
        (direct / name).write_bytes(b"fixture")

    assert cyj11_native_stack_root(tmp_path, "DYZ") == direct


def test_cyj11_roi_plane_uses_only_explicit_discrete_mapping() -> None:
    from paper_experiments.joint_vs_2dfit.aha.cyj11_legacy_roi import mapped_manual_roi_plane

    outer = np.zeros((2, 3, 1), np.uint8); outer[:, :, 0] = [[1, 0, 0], [1, 1, 0]]
    inner = np.zeros_like(outer)
    result = mapped_manual_roi_plane(outer, inner, roi_slice_index=0, roi_transform="transpose", expected_shape=(3, 2))

    assert np.array_equal(result["outer"], np.array([[True, True], [False, True], [False, False]]))
    assert not result["inner"].any()


def test_cyj11_mapping_requires_the_declared_immutable_roi_bundle(tmp_path) -> None:
    """A mapping cannot silently point at an unrelated manual ROI bundle."""
    from paper_experiments.joint_vs_2dfit.aha.cyj11_adapter import require_cyj11_mapping
    from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256

    prepared = tmp_path / "observations.npz"
    np.savez_compressed(prepared, group_idx=np.repeat([2, 4, 6], 10), weight_idx=np.tile(np.arange(10), 3))
    bundle = tmp_path / "roi_derivation_manifest.json"
    bundle.write_text(json.dumps({"schema": "cyj11_native_roi_bundle/v1", "subject": "DYZ", "planes": [
        {"native_group_idx": 2, "roi_slice_index": 7},
        {"native_group_idx": 4, "roi_slice_index": 8},
        {"native_group_idx": 6, "roi_slice_index": 9},
    ]}))
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({
        "schema": "cyj11_aha_native_roi_mapping/v1", "subject": "DYZ", "case": "L100_fold0",
        "prepared_sax": {"path": str(prepared), "sha256": sha256(prepared)},
        "roi_bundle_manifest": {"path": str(bundle), "sha256": "wrong"},
        "planes": [
            {"level": "basal", "native_group_idx": 2, "roi_slice_index": 7, "roi_transform": "identity", "anterior_xy": [1, 2], "inferior_xy": [3, 4]},
            {"level": "mid", "native_group_idx": 4, "roi_slice_index": 8, "roi_transform": "identity", "anterior_xy": [1, 2], "inferior_xy": [3, 4]},
            {"level": "apical", "native_group_idx": 6, "roi_slice_index": 9, "roi_transform": "identity", "anterior_xy": [1, 2], "inferior_xy": [3, 4]},
        ],
    }))

    with pytest.raises(ValueError, match="ROI bundle"):
        require_cyj11_mapping(mapping, subject="DYZ", case="L100_fold0")


def test_cyj11_aha_bundle_is_immutable_and_uses_mapped_common_support(tmp_path, monkeypatch) -> None:
    """The downstream writer emits all four method/parameter bullseyes without endpoint mutation."""
    from paper_experiments.joint_vs_2dfit.aha import cyj11_aha16

    mapping = {"planes": [
        {"level": "basal", "native_group_idx": 1, "roi_slice_index": 0, "roi_transform": "identity", "anterior_xy": [5, 2], "inferior_xy": [1, 2]},
        {"level": "mid", "native_group_idx": 2, "roi_slice_index": 1, "roi_transform": "identity", "anterior_xy": [5, 2], "inferior_xy": [1, 2]},
        {"level": "apical", "native_group_idx": 3, "roi_slice_index": 2, "roi_transform": "identity", "anterior_xy": [5, 2], "inferior_xy": [1, 2]},
    ]}
    native = {"group_idx": np.array([1, 2, 3]), "t1_ms": np.full((3, 7, 7), 1000.0),
              "t2_ms": np.full((3, 7, 7), 80.0), "pixel_spacing_rc_mm": np.ones((3, 2))}
    endpoint_images = {"t1": {}, "t2": {}}
    monkeypatch.setattr(cyj11_aha16, "require_cyj11_mapping", lambda *args, **kwargs: mapping)
    monkeypatch.setattr(cyj11_aha16, "load_cyj11_run_inputs", lambda *args, **kwargs: (native, endpoint_images))
    monkeypatch.setattr(cyj11_aha16, "load_cyj11_roi_plane", lambda *args, **kwargs: {
        "outer": np.ones((7, 7), bool), "inner": np.zeros((7, 7), bool), "points": np.zeros((7, 7), np.uint8)})
    monkeypatch.setattr(cyj11_aha16, "native_sax_plane", lambda **kwargs: {
        "reference": native[f"{kwargs['parameter']}_ms"][kwargs["group"] - 1],
        "proposed": native[f"{kwargs['parameter']}_ms"][kwargs["group"] - 1] + 1,
        "fit_first": native[f"{kwargs['parameter']}_ms"][kwargs["group"] - 1] - 1,
        "proposed_valid": np.ones((7, 7), bool), "fit_first_valid": np.ones((7, 7), bool)})
    monkeypatch.setattr(cyj11_aha16, "_validate_mapping_landmarks", lambda *args, **kwargs: ((5.0, 2.0), (1.0, 2.0)))
    monkeypatch.setattr(cyj11_aha16, "render_single_bullseye", lambda path, *args, **kwargs: (path.parent.mkdir(parents=True, exist_ok=True), path.write_text("plot"))[1])
    monkeypatch.setattr(cyj11_aha16, "_record", lambda path: {"path": str(path), "sha256": "fixture"})

    output = tmp_path / "aha"
    assert cyj11_aha16.build_cyj11_aha16(analysis_root=tmp_path, subject="DYZ", case="L100_fold0",
                                          mapping_path=tmp_path / "mapping.json", roi_bundle=tmp_path,
                                          output=output) == output
    assert (output / "aha16_segment_statistics.csv").is_file()
    assert len(list((output / "figures").rglob("*.png"))) == 4
    with pytest.raises(FileExistsError):
        cyj11_aha16.build_cyj11_aha16(analysis_root=tmp_path, subject="DYZ", case="L100_fold0",
                                      mapping_path=tmp_path / "mapping.json", roi_bundle=tmp_path, output=output)


def test_cyj11_myocardial_bundle_writes_per_plane_and_pooled_statistics(tmp_path, monkeypatch) -> None:
    from paper_experiments.joint_vs_2dfit.aha import cyj11_myocardial_roi_metrics as metrics

    mapping = {"planes": [
        {"level": "basal", "native_group_idx": 1, "roi_slice_index": 0, "roi_transform": "identity"},
        {"level": "mid", "native_group_idx": 2, "roi_slice_index": 1, "roi_transform": "identity"},
        {"level": "apical", "native_group_idx": 3, "roi_slice_index": 2, "roi_transform": "identity"},
    ]}
    native = {"group_idx": np.array([1, 2, 3, 4]), "t1_ms": np.full((4, 7, 7), 1000.0),
              "t2_ms": np.full((4, 7, 7), 80.0), "pixel_spacing_rc_mm": np.ones((4, 2))}
    (tmp_path / "manifest.json").write_text(json.dumps({"schema": "exact_native_myocardium_bundle/v2", "status": "PASS"}))
    np.savez_compressed(
        tmp_path / "sax_myocardium_masks.npz", group_idx=native["group_idx"],
        myocardium_full=np.ones((4, 7, 7), np.uint8),
        myocardium_core_legacy=np.ones((4, 7, 7), np.uint8),
        myocardium_core_1px=np.ones((4, 7, 7), np.uint8),
    )
    monkeypatch.setattr(metrics, "require_cyj11_mapping", lambda *args, **kwargs: mapping)
    monkeypatch.setattr(metrics, "load_cyj11_run_inputs", lambda *args, **kwargs: (native, {"t1": {}, "t2": {}}))
    monkeypatch.setattr(metrics, "native_sax_plane", lambda **kwargs: {
        "reference": native[f"{kwargs['parameter']}_ms"][kwargs["group"] - 1],
        "proposed": native[f"{kwargs['parameter']}_ms"][kwargs["group"] - 1] + 2,
        "fit_first": native[f"{kwargs['parameter']}_ms"][kwargs["group"] - 1] - 3,
        "proposed_valid": np.ones((7, 7), bool), "fit_first_valid": np.ones((7, 7), bool)})
    monkeypatch.setattr(metrics, "_record", lambda path: {"path": str(path), "sha256": "fixture"})

    output = tmp_path / "myocardial"
    assert metrics.build_cyj11_myocardial_roi_metrics(analysis_root=tmp_path, subject="DYZ", case="L100_fold0",
                                                       mapping_path=tmp_path / "mapping.json", roi_bundle=tmp_path,
                                                       output=output) == output
    per_plane = json.loads((output / "per_plane_statistics.json").read_text())
    pooled = json.loads((output / "pooled_sax_statistics.json").read_text())
    assert [plane["native_group_idx"] for plane in per_plane["planes"]] == [1, 2, 3, 4]
    assert pooled["pooled_sax"]["t1"]["myocardium_full"]["proposed"]["bias"] == 2.0


def test_cyj11_roi_derivation_copies_only_integer_native_lattice_pixels(tmp_path, monkeypatch) -> None:
    """Synthetic native geometry exercises the no-resampling cropped ROI bundle path."""
    from paper_experiments.joint_vs_2dfit.aha.cyj11_legacy_roi import (
        ROI_FILENAMES, derive_cyj11_native_roi_bundle, load_cyj11_roi_plane,
    )
    from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256

    source_root = tmp_path / "source"; source_root.mkdir()
    arrays = {name: np.zeros((3, 4, 3), np.uint8) for name in ROI_FILENAMES}
    arrays["Segmentation_1-out-label.nii.gz"][:, :, :] = 1
    arrays["Segmentation_3-label_point.nii.gz"][1, 1, 0] = 1
    for name in ROI_FILENAMES: (source_root / name).write_bytes(name.encode())
    composite_affine = np.eye(4); composite_affine[2, 2] = -1; composite_affine[2, 3] = 2
    images = {
        str(source_root / name): types.SimpleNamespace(
            shape=array.shape, affine=composite_affine if name == "sax_composite_reference.nii.gz" else np.eye(4), dataobj=array)
        for name, array in arrays.items()
    }
    monkeypatch.setitem(sys.modules, "nibabel", types.SimpleNamespace(load=lambda path: images[str(path)]))

    prepared = tmp_path / "observations.npz"
    np.savez_compressed(prepared, group_idx=np.repeat([0, 1, 2], 10), weight_idx=np.tile(np.arange(10), 3), images=np.zeros((30, 3, 4)))
    native = tmp_path / "native"; native.mkdir()
    np.savez_compressed(native / "native_maps.npz", group_idx=np.array([0, 1, 2]),
                        t1_ms=np.zeros((3, 3, 4)), t2_ms=np.zeros((3, 3, 4)), b1=np.zeros((3, 3, 4)))
    affine_lps = np.repeat(np.diag([-1.0, -1.0, 1.0, 1.0])[None], 3, axis=0)
    for index in range(3): affine_lps[index, 2, 3] = index
    np.savez_compressed(native / "geometry.npz", affine_lps_rc=affine_lps,
                        pixel_spacing_rc_mm=np.ones((3, 2)), slice_thickness_mm=np.ones(3))
    (native / "manifest.json").write_text(json.dumps({"source_observations": {"path": str(prepared), "sha256": sha256(prepared)},
                                                         "native_maps_sha256": sha256(native / "native_maps.npz"), "geometry_sha256": sha256(native / "geometry.npz")}))
    config = tmp_path / "roi_config.json"
    config.write_text(json.dumps({"schema": "cyj11_native_roi_derivation_config/v1", "subject": "DYZ",
                                  "prepared_sax": {"path": str(prepared), "sha256": sha256(prepared)},
                                  "source_roi": {"path": str(source_root)}, "native_sax_root": str(native),
                                  "planes": [{"native_group_idx": index, "roi_slice_index": index} for index in range(3)]}))

    bundle = derive_cyj11_native_roi_bundle(config=config, output=tmp_path / "derived")
    plane = load_cyj11_roi_plane(bundle, group=2, expected_shape=(3, 4))
    assert plane["outer"].all() and not plane["points"].any()
    with np.load(bundle / "sax_myocardium_masks.npz", allow_pickle=False) as masks:
        assert masks["group_idx"].tolist() == [0, 1, 2]
        assert masks["myocardium_full"].shape == (3, 3, 4)
        assert not masks["landmark_labels"][1].any()
    assert json.loads((bundle / "manifest.json").read_text())["schema"] == "exact_native_myocardium_bundle/v2"


def test_cyj11_config_derivation_writes_all_exact_native_plane_mappings(tmp_path, monkeypatch) -> None:
    """The reusable config records every native group and its exact manual slice."""
    from paper_experiments.joint_vs_2dfit.aha.cyj11_legacy_roi import (
        ROI_FILENAMES, derive_cyj11_native_roi_config,
    )
    from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256

    root, source = tmp_path / "analysis", tmp_path / "source"
    source.mkdir()
    arrays = {name: np.zeros((3, 4, 3), np.uint8) for name in ROI_FILENAMES}
    composite_affine = np.eye(4); composite_affine[2, 2] = -1; composite_affine[2, 3] = 2
    for name in ROI_FILENAMES:
        (source / name).write_bytes(name.encode())
    images = {
        str(source / name): types.SimpleNamespace(
            shape=array.shape,
            affine=composite_affine if name == "sax_composite_reference.nii.gz" else np.eye(4),
            dataobj=array,
        )
        for name, array in arrays.items()
    }
    monkeypatch.setitem(sys.modules, "nibabel", types.SimpleNamespace(load=lambda path: images[str(path)]))

    prepared = root / "prepared/DYZ/sax/observations.npz"; prepared.parent.mkdir(parents=True)
    np.savez_compressed(prepared, group_idx=np.repeat([0, 1, 2], 10), weight_idx=np.tile(np.arange(10), 3), images=np.zeros((30, 3, 4)))
    native = root / "native_cache/DYZ/native_maps/sax"; native.mkdir(parents=True)
    np.savez_compressed(native / "native_maps.npz", group_idx=np.array([0, 1, 2]),
                        t1_ms=np.zeros((3, 3, 4)), t2_ms=np.zeros((3, 3, 4)), b1=np.zeros((3, 3, 4)))
    affine_lps = np.repeat(np.diag([-1.0, -1.0, 1.0, 1.0])[None], 3, axis=0)
    for index in range(3):
        affine_lps[index, 2, 3] = 2 - index
    np.savez_compressed(native / "geometry.npz", affine_lps_rc=affine_lps,
                        pixel_spacing_rc_mm=np.ones((3, 2)), slice_thickness_mm=np.ones(3))
    (native / "manifest.json").write_text(json.dumps({
        "source_observations": {"path": str(prepared), "sha256": sha256(prepared)},
        "native_maps_sha256": sha256(native / "native_maps.npz"),
        "geometry_sha256": sha256(native / "geometry.npz"),
    }))

    output = root / "roi/config/DYZ/config.json"
    config = derive_cyj11_native_roi_config(analysis_root=root, subject="DYZ", source_roi=source, output=output)
    payload = json.loads(config.read_text())
    assert payload["schema"] == "cyj11_native_roi_derivation_config/v1"
    assert payload["prepared_sax"]["sha256"] == sha256(prepared)
    assert [plane["native_group_idx"] for plane in payload["planes"]] == [0, 1, 2]
    assert [plane["roi_slice_index"] for plane in payload["planes"]] == [2, 1, 0]
