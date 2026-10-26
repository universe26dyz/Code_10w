from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest


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
    native = {"group_idx": np.array([1, 2, 3]), "t1_ms": np.full((3, 7, 7), 1000.0),
              "t2_ms": np.full((3, 7, 7), 80.0), "pixel_spacing_rc_mm": np.ones((3, 2))}
    monkeypatch.setattr(metrics, "require_cyj11_mapping", lambda *args, **kwargs: mapping)
    monkeypatch.setattr(metrics, "load_cyj11_run_inputs", lambda *args, **kwargs: (native, {"t1": {}, "t2": {}}))
    monkeypatch.setattr(metrics, "load_cyj11_roi_plane", lambda *args, **kwargs: {
        "outer": np.ones((7, 7), bool), "inner": np.zeros((7, 7), bool), "points": np.zeros((7, 7), np.uint8)})
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
    assert len(per_plane["planes"]) == 3
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
    arrays["Segmentation_3-label_point.nii.gz"][1, 1, :] = 1
    for name in ROI_FILENAMES: (source_root / name).write_bytes(name.encode())
    images = {str(source_root / name): types.SimpleNamespace(shape=array.shape, affine=np.eye(4), dataobj=array)
              for name, array in arrays.items()}
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
    assert plane["outer"].all() and plane["points"][1, 1]
