import json

import numpy as np
import pytest


def _write_pair_eval(root, *, decoder_type, seed=7, native_offset=0.0, final_pose_source=None):
    root.mkdir()
    manifest = {
        "schema": "map_domain_psf_reprojection/v1",
        "subject_id": "CYJ",
        "method": "shared_svr",
        "reconstruction_method": "shared_svr",
        "geometry_reprojection_path": "shared_svr_prepared_geometry_final_poses",
        "geometry_source": "/prepared/CYJ",
        "psf_implementation": "map_domain_psf.reproject_volume_to_native_map_psf",
        "psf_source": "nesvor.svr.reconstruction.simulate_slices",
        "physical_resolution_thickness_source": "prepared observations",
        "pose_source": final_pose_source or "/runs/final_rigid_poses.json",
        "domain": "map",
        "n_samples": 32,
        "seed": seed,
        "decoder_type": decoder_type,
        "native_reference": {"manifest_sha256": "native-reference-sha"},
    }
    (root / "map_domain_psf_manifest.json").write_text(json.dumps(manifest))
    shapes = {"sax": (2, 9, 9), "2ch": (1, 7, 11), "4ch": (1, 5, 8)}
    for parameter, base in (("T1", 1000.0), ("T2", 50.0)):
        arrays = {
            "schema": np.asarray("map_domain_psf_comparison_stackwise/v1"),
            "stack_names": np.asarray(("sax", "2ch", "4ch")),
        }
        for index, (stack, shape) in enumerate(shapes.items()):
            native = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + base + native_offset
            prediction = native + (1.0 if decoder_type == "Bloch" else 2.0)
            support = np.ones(shape, dtype=bool)
            arrays.update({
                f"{stack}_predicted_ms": prediction,
                f"{stack}_native_reference_ms": native,
                f"{stack}_residual_ms": prediction - native,
                f"{stack}_common_support": support,
            })
        np.savez_compressed(root / f"{parameter}_native_map_domain_psf_comparison.npz", **arrays)
    return root


def test_load_decoder_pair_accepts_heterogeneous_native_stacks_and_distinct_final_poses(tmp_path):
    from trad.evaluation.scmr.decoder_pair import load_decoder_pair

    trad = _write_pair_eval(tmp_path / "trad", decoder_type="Bloch", final_pose_source="/trad/final_rigid_poses.json")
    mlp = _write_pair_eval(tmp_path / "mlp", decoder_type="FrozenMLP", final_pose_source="/mlp/final_rigid_poses.json")

    pair = load_decoder_pair("CYJ", trad, mlp)

    assert pair.trad.manifest["pose_source"] != pair.mlp.manifest["pose_source"]
    assert pair.trad.arrays["T1"]["2ch"]["native"].shape == (1, 7, 11)
    assert pair.mlp.arrays["T2"]["4ch"]["native"].shape == (1, 5, 8)


def test_pair_metrics_use_intersected_support_and_slice_ssim(tmp_path):
    from trad.evaluation.scmr.decoder_pair import evaluate_pair_metrics, load_decoder_pair

    trad = _write_pair_eval(tmp_path / "trad", decoder_type="Bloch")
    mlp = _write_pair_eval(tmp_path / "mlp", decoder_type="FrozenMLP")
    pair = load_decoder_pair("CYJ", trad, mlp)
    pair.mlp.arrays["T1"]["sax"]["support"][0, :2, :] = False

    metrics = evaluate_pair_metrics(pair)

    trad_row = next(row for row in metrics["paired_common_support"]["per_slice"] if row["parameter"] == "T1" and row["stack"] == "sax" and row["group_idx"] == 0 and row["method"] == "Bloch")
    mlp_row = next(row for row in metrics["paired_common_support"]["per_slice"] if row["parameter"] == "T1" and row["stack"] == "sax" and row["group_idx"] == 0 and row["method"] == "FrozenMLP")
    assert trad_row["N"] == mlp_row["N"] == 63
    assert np.isfinite(trad_row["SSIM"])
    assert np.isfinite(mlp_row["SSIM"])
    global_t1 = metrics["method_specific_global"]["T1"]
    assert global_t1["Bloch"]["N"] == 279
    assert global_t1["FrozenMLP"]["N"] == 261
    assert np.isnan(global_t1["Bloch"]["SSIM"])
    assert np.isfinite(global_t1["Bloch"]["SSIM_slice_mean"])


def test_load_decoder_pair_rejects_native_reference_value_mismatch(tmp_path):
    from trad.evaluation.scmr.decoder_pair import load_decoder_pair

    trad = _write_pair_eval(tmp_path / "trad", decoder_type="Bloch")
    mlp = _write_pair_eval(tmp_path / "mlp", decoder_type="FrozenMLP", native_offset=1.0)

    with pytest.raises(ValueError, match="native-reference value mismatch for T1/sax"):
        load_decoder_pair("CYJ", trad, mlp)


@pytest.mark.parametrize("field", ("psf_implementation", "psf_source"))
def test_load_decoder_pair_rejects_missing_required_psf_contract_field(tmp_path, field):
    from trad.evaluation.scmr.decoder_pair import load_decoder_pair

    trad = _write_pair_eval(tmp_path / "trad", decoder_type="Bloch")
    mlp = _write_pair_eval(tmp_path / "mlp", decoder_type="FrozenMLP")
    for root in (trad, mlp):
        manifest_path = root / "map_domain_psf_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.pop(field)
        manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=field):
        load_decoder_pair("CYJ", trad, mlp)


def test_load_decoder_pair_rejects_different_psf_source(tmp_path):
    from trad.evaluation.scmr.decoder_pair import load_decoder_pair

    trad = _write_pair_eval(tmp_path / "trad", decoder_type="Bloch")
    mlp = _write_pair_eval(tmp_path / "mlp", decoder_type="FrozenMLP")
    manifest_path = mlp / "map_domain_psf_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["psf_source"] = "other.psf.source"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="psf_source"):
        load_decoder_pair("CYJ", trad, mlp)


@pytest.mark.parametrize("field, value", [("seed", 8), ("n_samples", 128), ("native_reference.manifest_sha256", "other-native-reference")])
def test_load_decoder_pair_rejects_psf_provenance_mismatch(tmp_path, field, value):
    from trad.evaluation.scmr.decoder_pair import load_decoder_pair

    trad = _write_pair_eval(tmp_path / "trad", decoder_type="Bloch")
    mlp = _write_pair_eval(tmp_path / "mlp", decoder_type="FrozenMLP")
    manifest_path = mlp / "map_domain_psf_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if field.startswith("native_reference"):
        manifest["native_reference"]["manifest_sha256"] = value
    else:
        manifest[field] = value
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError):
        load_decoder_pair("CYJ", trad, mlp)


def _write_full_pair_fixture(tmp_path):
    h5py = pytest.importorskip("h5py")
    nib = pytest.importorskip("nibabel")
    from trad.evaluation.scmr.reference_2d import sha256

    subject = "CYJ"
    trad_eval = _write_pair_eval(tmp_path / "trad_eval", decoder_type="Bloch")
    mlp_eval = _write_pair_eval(tmp_path / "mlp_eval", decoder_type="FrozenMLP")
    reference_root, preprocessed_root = tmp_path / "reference", tmp_path / "preprocessed"
    reference_root.mkdir()
    shapes = {"sax": (2, 9, 9), "2ch": (1, 7, 11), "4ch": (1, 5, 8)}
    hashes, maps = {}, {}
    for stack_index, (stack, shape) in enumerate(shapes.items()):
        mat = preprocessed_root / subject / stack / "preprocessed.mat"
        mat.parent.mkdir(parents=True)
        with h5py.File(mat, "w") as handle:
            handle.create_dataset("proof", data=np.array([stack_index]))
        hashes[stack] = sha256(mat)
        t1 = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1000.0
        archive = reference_root / f"{stack}.npz"
        t2 = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 50.0
        np.savez_compressed(archive, t1_ms=t1, t2_ms=t2, valid_mask=np.ones(shape, dtype=bool), group_idx=np.arange(shape[0]))
        maps[stack] = archive.name
    affine = np.eye(4)
    t1_native, t2_native = reference_root / "sax_T1_stack_ms.nii.gz", reference_root / "sax_T2_stack_ms.nii.gz"
    sax_t1 = np.arange(162, dtype=np.float32).reshape(9, 9, 2) + 1000.0
    nib.save(nib.Nifti1Image(sax_t1, affine), t1_native)
    nib.save(nib.Nifti1Image(sax_t1 / 20.0, affine), t2_native)
    manifest_path = reference_root / "native_reference_manifest.json"
    manifest_path.write_text(json.dumps({"subject_id": subject, "preprocessing_semantics": "MP-PCA(MIND_mag_reg)", "map_units": "ms", "preprocessed_mat_sha256": hashes, "maps": maps, "figure1_native_stacks": {"t1": t1_native.name, "t2": t2_native.name}}))
    reference_sha = sha256(manifest_path)
    for root in (trad_eval, mlp_eval):
        manifest = json.loads((root / "map_domain_psf_manifest.json").read_text())
        manifest["native_reference"]["manifest_sha256"] = reference_sha
        (root / "map_domain_psf_manifest.json").write_text(json.dumps(manifest))
    for label, offset in (("trad_run", 1.0), ("mlp_run", 2.0)):
        run = tmp_path / label
        run.mkdir()
        for parameter, volume in (("T1", sax_t1 + offset), ("T2", sax_t1 / 20.0 + offset), ("B1", np.full((9, 9, 2), 0.8, dtype=np.float32)), ("amplitude", np.full((9, 9, 2), 75.0, dtype=np.float32))):
            nib.save(nib.Nifti1Image(volume.astype(np.float32), affine), run / f"{parameter}_3D.nii.gz")
        (run / "timing_profile_summary.json").write_text(json.dumps({"decoder_type": "Bloch" if label == "trad_run" else "FrozenMLP", "run_level_ms": {"optimization_total": 100.0 if label == "trad_run" else 50.0, "total_runtime": 120.0 if label == "trad_run" else 60.0}}))
    mask_bundle = tmp_path / "masks"
    mask_bundle.mkdir()
    (mask_bundle / "manifest.json").write_text('{"schema":"exact_native_myocardium_bundle/v2","status":"PASS"}')
    np.savez_compressed(mask_bundle / "sax_myocardium_masks.npz", **{name: np.ones((2, 9, 9), dtype=bool) for name in ("myocardium_core_1px", "myocardium_full", "myocardium_core_legacy")})
    return subject, trad_eval, mlp_eval, tmp_path / "trad_run", tmp_path / "mlp_run", reference_root, preprocessed_root, mask_bundle


def test_decoder_pair_entry_writes_read_only_metrics_and_paired_visualizations(tmp_path):
    from argparse import Namespace

    from trad.evaluation.scmr.run_decoder_pair_evaluation import run

    subject, trad_eval, mlp_eval, trad_run, mlp_run, reference_root, preprocessed_root, mask_bundle = _write_full_pair_fixture(tmp_path)
    output = tmp_path / "decoder_pair"
    result = run(Namespace(subject_id=subject, trad_eval=trad_eval, mlp_eval=mlp_eval, trad_run=trad_run, mlp_run=mlp_run, native_reference=reference_root, preprocessed_root=preprocessed_root, mask_bundle=mask_bundle, legacy_source="/home/universe/SVR/multimap_postprogramming/MultiMapCode/method_repositories/2D_fit_first", output=output, plane_axis="x", plane_index=4, figure1_reference_group=0))

    assert result == output
    assert (output / "figure2" / "Figure2_T1_native_bloch_mlp_psf.png").is_file()
    assert (output / "all_slices" / "T2" / "T2_4CH_all_slices_pair.png").is_file()
    assert (output / "figure1" / "Figure1_native_bloch_mlp_through_plane.png").is_file()
    assert (output / "range_audit" / "pair_range_summary.json").is_file()
    summary = json.loads((output / "metrics" / "decoder_pair_summary.json").read_text())
    assert summary["read_only_scope"].startswith("existing reconstruction volumes")
    assert summary["selected_representative_groups"] == {"sax": 0, "2ch": 0, "4ch": 0}
    assert summary["timing_summary_sources"]["optimization_speedup_bloch_over_frozen_mlp"] == 2.0
    figure1 = summary["figure1"]
    assert figure1["scientific_panels"] == ["Native SAX + cut line", "Native through-plane (nearest)", "Bloch 3-D through-plane (linear)", "FrozenMLP 3-D through-plane (linear)"]
    for field in ("world_plane_affine_source", "world_point_count", "plane_shape", "physical_extent_mm", "native_interpolation", "reconstruction_interpolation", "same_world_plane_check", "display_ranges_ms", "colorbars"):
        assert field in figure1
