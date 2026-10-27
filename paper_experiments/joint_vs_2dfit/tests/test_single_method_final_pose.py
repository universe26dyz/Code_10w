from __future__ import annotations

import json

import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256


def _native() -> dict[str, np.ndarray]:
    return {
        "group_idx": np.asarray([3]),
        "t1_ms": np.asarray([[[1000.0, 1100.0], [1200.0, 1300.0]]]),
        "t2_ms": np.asarray([[[40.0, 45.0], [50.0, 55.0]]]),
    }


def _fit_bundle(root, *, shape=(2, 2)):
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({
        "schema": "fit_first_native_prediction_bundle/v2", "endpoint": "central",
    }), encoding="utf-8")
    plane = root / "planes/t1/sax"; plane.mkdir(parents=True)
    np.savez_compressed(plane / "group003.npz",
                        prediction_ms=np.full(shape, 999.0),
                        volume_geometry_valid=np.ones(shape, bool),
                        query_effective_mask=np.asarray([[True, False], [True, True]]),
                        valid=np.asarray([[True, False], [True, True]]))


def _proposed_bundle(root, *, shape=(2, 2)):
    root.mkdir()
    plane = root / "planes/group003.npz"; plane.parent.mkdir()
    np.savez_compressed(plane, t1_ms=np.full(shape, 1001.0), t2_ms=np.full(shape, 51.0))
    (root / "manifest.json").write_text(json.dumps({
        "schema": "direct_sparse_native_prediction_bundle/v1", "endpoint": "central",
        "subject": "HHZ", "case_id": "L100_fold0",
        "planes": [{"stack": "sax", "original_group_idx": 3, "path": "planes/group003.npz"}],
    }), encoding="utf-8")


def test_load_fit_first_plane_uses_bundle_validity(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.single_method_final_pose import load_single_method_plane

    bundle = tmp_path / "fit"; _fit_bundle(bundle)
    result = load_single_method_plane(native=_native(), bundle_root=bundle, method="fit_first", group=3, parameter="t1")

    assert np.array_equal(result["prediction"], np.full((2, 2), 999.0))
    assert np.array_equal(result["valid"], np.asarray([[True, False], [True, True]]))


def test_load_proposed_plane_requires_native_shape_match(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.single_method_final_pose import load_single_method_plane

    bundle = tmp_path / "proposed"; _proposed_bundle(bundle, shape=(2, 3))
    with pytest.raises(ValueError, match="geometry mismatch"):
        load_single_method_plane(native=_native(), bundle_root=bundle, method="proposed", group=3, parameter="t1")


def _full_proposed_input(tmp_path):
    prepared = tmp_path / "prepared.npz"
    groups = np.asarray([3, 4, 5])
    np.savez_compressed(prepared, group_idx=np.repeat(groups, 2), images=np.zeros((6, 2, 7, 7), np.float32))
    native = tmp_path / "native"; native.mkdir()
    maps = native / "native_maps.npz"; geometry = native / "geometry.npz"
    t1 = np.stack([np.full((7, 7), 1100.0 + group) for group in groups])
    t2 = np.stack([np.full((7, 7), 50.0 + group) for group in groups])
    np.savez_compressed(maps, group_idx=groups, t1_ms=t1, t2_ms=t2, b1=np.ones_like(t1))
    np.savez_compressed(geometry, affine_lps_rc=np.repeat(np.eye(4)[None], 3, axis=0),
                        pixel_spacing_rc_mm=np.ones((3, 2)), slice_thickness_mm=np.ones(3))
    (native / "manifest.json").write_text(json.dumps({
        "source_observations": {"sha256": sha256(prepared)},
        "native_maps_sha256": sha256(maps), "geometry_sha256": sha256(geometry),
    }), encoding="utf-8")

    roi = tmp_path / "roi"; (roi / "planes").mkdir(parents=True)
    entries = []
    full = np.zeros((7, 7), bool); full[1:6, 1:6] = True; full[3, 3] = False
    points = np.zeros((7, 7), np.uint8); points[1, 3] = 1; points[5, 3] = 2
    for group in groups:
        path = roi / "planes" / f"group{group:03d}.npz"
        np.savez_compressed(path, outer=full | (np.indices(full.shape)[0] == 3) & (np.indices(full.shape)[1] == 3),
                            inner=np.zeros((7, 7), bool), points=points)
        entries.append({"native_group_idx": int(group), "roi_slice_index": int(group - 3),
                        "plane": {"path": str(path), "sha256": sha256(path)}})
    derivation = roi / "roi_derivation_manifest.json"
    derivation.write_text(json.dumps({"schema": "cyj11_native_roi_bundle/v1", "subject": "HHZ", "planes": entries}), encoding="utf-8")
    np.savez_compressed(roi / "sax_myocardium_masks.npz", group_idx=groups,
                        myocardium_full=np.stack([full] * 3), myocardium_core_legacy=np.stack([full] * 3),
                        myocardium_core_1px=np.stack([full] * 3))
    (roi / "manifest.json").write_text(json.dumps({"schema": "exact_native_myocardium_bundle/v2", "status": "PASS"}), encoding="utf-8")

    mapping = tmp_path / "mapping.json"
    levels = ("basal", "mid", "apical")
    mapping.write_text(json.dumps({
        "schema": "cyj11_aha_native_roi_mapping/v1", "subject": "HHZ", "case": "L100_fold0",
        "prepared_sax": {"path": str(prepared), "sha256": sha256(prepared)},
        "roi_bundle_manifest": {"path": str(derivation), "sha256": sha256(derivation)},
        "planes": [{"level": level, "native_group_idx": int(group), "roi_slice_index": int(group - 3),
                    "roi_transform": "identity", "anterior_xy": [3.0, 1.0], "inferior_xy": [3.0, 5.0]}
                   for level, group in zip(levels, groups)],
    }), encoding="utf-8")

    bundle = tmp_path / "proposed"; (bundle / "planes").mkdir(parents=True)
    records = []
    for group, t1_plane, t2_plane in zip(groups, t1, t2):
        path = bundle / "planes" / f"group{group:03d}.npz"
        np.savez_compressed(path, t1_ms=t1_plane + 10, t2_ms=t2_plane + 2)
        records.append({"stack": "sax", "original_group_idx": int(group), "path": str(path.relative_to(bundle))})
    (bundle / "manifest.json").write_text(json.dumps({
        "schema": "direct_sparse_native_prediction_bundle/v1", "endpoint": "central",
        "subject": "HHZ", "case_id": "L100_fold0", "planes": records,
    }), encoding="utf-8")
    return native, prepared, mapping, roi, bundle


def _full_fit_bundle(tmp_path, groups) -> object:
    bundle = tmp_path / "fit"; bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps({
        "schema": "fit_first_native_prediction_bundle/v2", "endpoint": "central",
    }), encoding="utf-8")
    for parameter, base in (("t1", 1110.0), ("t2", 55.0)):
        folder = bundle / "planes" / parameter / "sax"; folder.mkdir(parents=True)
        for group in groups:
            prediction = np.full((7, 7), base + group)
            valid = np.ones((7, 7), bool); valid[0, 0] = False
            np.savez_compressed(folder / f"group{group:03d}.npz", prediction_ms=prediction,
                                volume_geometry_valid=valid, query_effective_mask=np.ones((7, 7), bool), valid=valid)
    return bundle


def test_single_method_writer_marks_diagnostic_and_generates_aha_and_roi(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.single_method_final_pose import build_single_method_final_pose

    native, prepared, mapping, roi, bundle = _full_proposed_input(tmp_path)
    output = build_single_method_final_pose(subject="HHZ", case="L100_fold0", method="proposed",
                                            native_sax_root=native, prepared_sax=prepared, mapping_path=mapping,
                                            roi_bundle=roi, native_bundle=bundle, output=tmp_path / "result")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    rows = (output / "aha16" / "aha16_segment_statistics.csv").read_text(encoding="utf-8").splitlines()

    assert manifest["diagnostic_only"] is True
    assert manifest["sampling"] == "final_pose"
    assert manifest["method"] == "proposed"
    assert len(rows) == 33
    assert (output / "myocardial_roi" / "pooled_sax_statistics.json").is_file()


def test_single_method_writer_accepts_fit_first_bundle(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.aha.single_method_final_pose import build_single_method_final_pose

    native, prepared, mapping, roi, _proposed = _full_proposed_input(tmp_path)
    output = build_single_method_final_pose(subject="HHZ", case="L100_fold0", method="fit_first",
                                            native_sax_root=native, prepared_sax=prepared, mapping_path=mapping,
                                            roi_bundle=roi, native_bundle=_full_fit_bundle(tmp_path, [3, 4, 5]),
                                            output=tmp_path / "fit_result")

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["method"] == "fit_first"
    assert manifest["method_bundle_schema"] == "fit_first_native_prediction_bundle/v2"
