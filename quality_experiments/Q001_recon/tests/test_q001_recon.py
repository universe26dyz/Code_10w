import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from quality_experiments.Q001_recon.config import build_route_config, load_b6_resolved_config
from quality_experiments.Q001_recon.evaluate_q001 import fixed_baseline_map_pair
from quality_experiments.Q001_recon.run_controlled_reconstruction import checkpoint_provenance
from quality_experiments.Q001_recon.contracts import verify_q001_input_bundle
from quality_experiments.Q001_recon.evaluation import ROI_BY_STACK, evaluation_plan, fixed_roi
from quality_experiments.Q001_recon.metrics import legacy_macro_rows, strict_common_support, true_pooled_rows
from quality_experiments.Q001_recon.pose_transfer import compose_stack_delta
from quality_experiments.Q001_recon.routes import build_route_plan, registration_inputs_for_training
from quality_experiments.Q001_recon.verify_full_fov_reference import verify_full_fov_reference


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path):
    root = tmp_path / "inputs"; records = {}
    for stack, groups, shape in (("sax", 14, [288, 256]), ("2ch", 15, [288, 256]), ("4ch", 12, [256, 288])):
        paths = {}
        for key, relative in (("preprocessed_mat", f"full_fov_preprocessed/CYJ/{stack}/preprocessed.mat"), ("prepared_observations", f"full_fov_prepared/CYJ/{stack}/observations.npz"), ("prepared_manifest", f"full_fov_prepared/CYJ/{stack}/manifest.json"), ("timing", f"full_fov_prepared/CYJ/{stack}/timing.npy"), ("qc_summary", f"full_fov_prepared/CYJ/{stack}/qc_summary.json"), ("cropped_vs_full_qc", f"qc/CYJ/{stack}.json")):
            path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(key.encode()); paths[key] = {"path": str(path), "sha256": _sha(path)}
        records[stack] = {**paths, "qc_passed": True, "group_count": groups, "observation_count": groups * 10, "full_image_shape_rows_cols": shape}
    manifest = {"status": "READY_FOR_Q001_RECON_IMPLEMENTATION", "formal_preprocessing_semantics": "MP-PCA(full-FOV MIND_mag_reg)", "stacks": records}
    (root / "q001_input_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_ready_input_bundle_validates_hashes_and_shape_contract(tmp_path):
    bundle = _bundle(tmp_path)
    assert verify_q001_input_bundle(bundle)["status"] == "READY_FOR_Q001_RECON_IMPLEMENTATION"
    (bundle / "qc/CYJ/sax.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"): verify_q001_input_bundle(bundle)


def test_server_copy_verifies_relative_artifacts_not_local_manifest_paths(tmp_path):
    bundle = _bundle(tmp_path)
    manifest_path = bundle / "q001_input_manifest.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest["stacks"].values():
        for artifact in ("preprocessed_mat", "prepared_observations", "prepared_manifest", "timing", "qc_summary", "cropped_vs_full_qc"):
            record[artifact]["path"] = "/retired-local-root/" + artifact
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_q001_input_bundle(bundle)["status"] == "READY_FOR_Q001_RECON_IMPLEMENTATION"


def test_b6_config_is_loaded_from_checkpoint_and_route_metadata_only(tmp_path):
    checkpoint = tmp_path / "model.pt"; config = {"training": {"seed": 7}, "decoder": {"checkpoint": "/signal.pth"}, "stack_initialization": {"enabled": True}}
    torch.save({"resolved_config": config, "model_state": {"inr.weight": torch.ones(1)}}, checkpoint)
    loaded = load_b6_resolved_config(checkpoint)
    route = build_route_config(loaded, "Q001A_full_fov_full_recon", full_prepared_root="/full", cropped_prepared_root=None)
    assert route["resolved_config"] == config
    assert route["route"]["training_prepared_root"] == "/full"
    with pytest.raises(ValueError, match="warm-start"): build_route_config(loaded, "Q001A_full_fov_full_recon", full_prepared_root="/full", cropped_prepared_root=None, reconstruction_checkpoint="/b6/model.pt")


def test_q001_routes_keep_full_data_out_of_q001b_optimization():
    q001a = build_route_plan("Q001A", ["/full/sax", "/full/2ch", "/full/4ch"], ["/crop/sax", "/crop/2ch", "/crop/4ch"])
    q001b = build_route_plan("Q001B", ["/full/sax", "/full/2ch", "/full/4ch"], ["/crop/sax", "/crop/2ch", "/crop/4ch"])
    assert q001a.training_inputs == q001a.stack_initialization_inputs
    assert q001b.training_inputs == ("/crop/sax", "/crop/2ch", "/crop/4ch")
    assert q001b.stack_initialization_inputs == ("/full/sax", "/full/2ch", "/full/4ch")
    assert q001b.stack_registration_calls == 1


def test_rigid_delta_transfer_preserves_world_point_center_and_normal():
    initial = torch.tensor([[0.0, 0.0, 0.1, 2.0, -1.0, 0.5]])
    registered = torch.tensor([[0.0, 0.0, 0.2, 4.0, -3.0, 1.0]])
    dicom = torch.tensor([[0.1, 0.0, 0.0, -2.0, 1.0, 3.0]])
    post, delta = compose_stack_delta(dicom, initial, registered)
    assert post.shape == dicom.shape and delta.shape == (1, 6)
    assert torch.isfinite(post).all()


@pytest.mark.parametrize(("stack", "expected"), [("sax", (slice(71, 216), slice(63, 192))), ("2ch", (slice(71, 216), slice(63, 192))), ("4ch", (slice(63, 192), slice(71, 216)))])
def test_fixed_baseline_rois_are_exact_and_do_not_resize(stack, expected):
    assert fixed_roi(stack) == expected
    assert ROI_BY_STACK[stack] == expected
    image = np.zeros((288, 256) if stack != "4ch" else (256, 288))
    assert image[expected].shape == ((145, 129) if stack != "4ch" else (129, 145))


@pytest.mark.parametrize(("stack", "shape"), (("sax", (288, 256)), ("2ch", (288, 256)), ("4ch", (256, 288))))
def test_fixed_endpoint_pairs_full_prediction_roi_with_entire_cropped_reference(stack, shape):
    cropped_shape = (145, 129) if stack != "4ch" else (129, 145)
    prediction = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    reference = np.full(cropped_shape, 77.0, np.float32)
    pred, ref, support = fixed_baseline_map_pair(stack, prediction, reference, np.ones(shape, bool), np.ones(cropped_shape, bool))
    assert pred.shape == ref.shape == support.shape == cropped_shape
    assert np.array_equal(pred, prediction[fixed_roi(stack)])
    assert np.array_equal(ref, reference) and np.all(ref == 77.0)


def test_b6_and_signal_simulator_provenance_are_distinct(tmp_path):
    b6, signal = tmp_path / "b6_model.pt", tmp_path / "signal_simulator.pth"
    b6.write_bytes(b"b6"); signal.write_bytes(b"signal")
    provenance = checkpoint_provenance(b6, signal)
    assert provenance["b6_model_sha256"] == _sha(b6)
    assert provenance["frozen_signal_simulator_sha256"] == _sha(signal)
    assert provenance["b6_model_sha256"] != provenance["frozen_signal_simulator_sha256"]


def test_evaluation_plan_has_central_k32_k8_same_checkpoint():
    plan = evaluation_plan("abc123", "Q001A")
    assert plan["mapping_central_no_psf"]["psf_samples"] == 1
    assert plan["mapping_map_psf_K32"] == {"psf_samples": 32, "seed": 20260911, "checkpoint_sha256": "abc123"}
    assert plan["signal_psf_K8"]["psf_samples"] == 8


def _reference(tmp_path, bundle):
    root = tmp_path / "reference"; root.mkdir(parents=True); stacks = {}
    for stack, groups, shape in (("sax", 14, (288, 256)), ("2ch", 15, (288, 256)), ("4ch", 12, (256, 288))):
        path = root / f"{stack}.npz"; array = np.ones((groups, *shape), np.float32)
        np.savez_compressed(path, t1_ms=array, t2_ms=array * 2, valid_mask=np.ones_like(array, bool), group_idx=np.arange(groups))
        stacks[stack] = {"map": path.name, "map_sha256": _sha(path), "source_preprocessed_mat_sha256": json.loads((bundle / "q001_input_manifest.json").read_text())["stacks"][stack]["preprocessed_mat"]["sha256"], "shape_group_row_col": [groups, *shape]}
    (root / "native_reference_manifest.json").write_text(json.dumps({"schema": "q001_full_fov_native_reference/v1", "subject_id": "CYJ", "spatial_mode": "full_fov", "preprocessing_semantics": "MP-PCA(full-FOV MIND_mag_reg)", "map_units": "ms", "stacks": stacks}), encoding="utf-8")
    return root


def test_full_fov_reference_verifier_checks_map_and_input_provenance(tmp_path):
    bundle = _bundle(tmp_path)
    reference = _reference(tmp_path / "again", bundle)
    report = verify_full_fov_reference(reference, bundle)
    assert report["stacks"]["4ch"]["shape_group_row_col"] == [12, 256, 288]
    with (reference / "sax.npz").open("ab") as handle: handle.write(b"tamper")
    with pytest.raises(ValueError, match="SHA256"): verify_full_fov_reference(reference, bundle)


def test_full_fov_reference_rejects_source_mat_hash_mismatch(tmp_path):
    bundle = _bundle(tmp_path); reference = _reference(tmp_path / "reference_fixture", bundle)
    manifest_path = reference / "native_reference_manifest.json"; manifest = json.loads(manifest_path.read_text()); manifest["stacks"]["sax"]["source_preprocessed_mat_sha256"] = "bad"; manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source preprocessed MAT"): verify_full_fov_reference(reference, bundle)


def test_q001_true_pooled_metrics_are_not_legacy_macro_and_support_is_strict():
    samples = [
        {"stack": "sax", "weight_idx": 0, "support_provenance": "fixture", "reference": np.array([0., 10.]), "prediction": np.array([0., 20.]), "support": np.array([True, True])},
        {"stack": "sax", "weight_idx": 0, "support_provenance": "fixture", "reference": np.array([0., 1.]), "prediction": np.array([1., 1.]), "support": np.array([True, True])},
    ]
    keys = ("stack", "weight_idx", "support_provenance")
    pooled = true_pooled_rows(samples, keys, domain="signal")[0]
    macro = legacy_macro_rows(samples, keys, domain="signal")[0]
    assert pooled["RMSE_signal"] != macro["RMSE_signal"]
    assert np.isclose(pooled["Pearson_r"], pooled["NCC"], equal_nan=True)
    support = strict_common_support(np.array([1., np.nan]), np.array([1., 2.]), np.array([1., 2.]), np.array([True, True]), np.array([True, True]))
    assert support.tolist() == [True, False]


def test_stack_identity_ordering_prevents_list_order_pairing():
    crop = ["/crop/CYJ/4ch/observations.npz", "/crop/CYJ/sax/observations.npz", "/crop/CYJ/2ch/observations.npz"]
    full = ["/full/CYJ/sax/observations.npz", "/full/CYJ/2ch/observations.npz", "/full/CYJ/4ch/observations.npz"]
    assert registration_inputs_for_training(crop, full) == tuple(["/full/CYJ/4ch/observations.npz", "/full/CYJ/sax/observations.npz", "/full/CYJ/2ch/observations.npz"])


def test_q001_migration_scripts_refuse_nonempty_before_rsync_and_diagnostic_has_grep_fallback():
    root = Path(__file__).resolve().parents[1]
    for script in (root.parent / "server_commands" / "migrate_Q001_inputs_CYJ.sh", root.parent / "server_commands" / "migrate_Q001_reference_CYJ.sh"):
        text = script.read_text(); assert "Refusing non-empty remote" in text and text.index("Refusing non-empty remote") < text.index("\nrsync -a")
    diagnostic = (root / "diagnostics" / "run_native_reference_group0_diagnostic.sh").read_text()
    assert "command -v rg" in diagnostic and "grep -Ei" in diagnostic


def test_diagnostic_grid_contract_is_explicit_and_matches_authoritative_counts():
    matlab = (Path(__file__).resolve().parents[1] / "matlab" / "q001_dictionary_grid_report.m").read_text()
    assert "20:20:500,505:5:1500,1520:20:2500" in matlab
    assert "grid.num_T1==275" in matlab and "grid.num_T2==30" in matlab and "grid.num_B1==23" in matlab
    assert "grid.num_valid_dictionary_entries==186990" in matlab


def _hb1_observations(path):
    images = np.ones((10, 3, 4), np.float32); masks = np.ones_like(images, bool)
    affine = np.repeat(np.eye(4)[None], 10, axis=0); affine[:, :3, 0] = [0., 1., 0.]; affine[:, :3, 1] = [1., 0., 0.]; affine[:, :3, 2] = [0., 0., 5.]
    np.savez_compressed(path, images=images, masks=masks, group_idx=np.zeros(10, dtype=np.int64), weight_idx=np.arange(10), affine_lps_rc=affine, pixel_spacing_rc_mm=np.repeat([[1., 1.]], 10, axis=0), slice_thickness_mm=np.full(10, 5.), stack_idx=np.zeros(10, dtype=np.int64), acquisition_time_ms=np.zeros(10), timing9_ms=np.zeros((10, 9)), tr_ms=np.ones(10), vps=np.ones(10))


def test_shared_hb1_init_uses_identity_ordered_full_route_once(monkeypatch, tmp_path):
    """Exercise the real shared initializer with synthetic full-FOV registration stacks."""
    from trad.modules.module_06_rigid_psf import hb1_stack_adapter as adapter
    from trad.third_party.nesvor.nesvor.transform import RigidTransform
    full = []
    for stack in ("sax", "2ch", "4ch"):
        directory = tmp_path / "full" / "CYJ" / stack; directory.mkdir(parents=True); path = directory / "observations.npz"; _hb1_observations(path); full.append(str(path))
    crop_order = [str(tmp_path / "crop" / "CYJ" / stack / "observations.npz") for stack in ("4ch", "sax", "2ch")]
    ordered = registration_inputs_for_training(crop_order, full)
    calls = []
    def register(nested, args_registration=None):
        calls.append(nested)
        result = []
        for index, stack in enumerate(nested[0]):
            registered = stack.clone()
            axis = registered.transformation.axisangle(trans_first=True).clone()
            axis[:, 3] += float(index + 1)
            registered.transformation = RigidTransform(axis, trans_first=True)
            result.append(registered)
        return result
    monkeypatch.setattr(adapter, "stack_registration", register)
    dicom = torch.tensor([[0., 0., 0., 40., 0., 0.], [0., 0., 0., 10., 0., 0.], [0., 0., 0., 20., 0., 0.]])
    result = adapter.initialize_group_poses_from_hb1(dicom, list(ordered), device="cpu")
    assert len(calls) == 1 and [Path(record["prepared_input"]).parent.name for record in result.stack_pose_records] == ["4ch", "sax", "2ch"]
    # Deltas follow the full stacks matched by cropped stack identity, with no second registration pass.
    expected = []
    for index in range(3):
        initial = calls[0][0][index].transformation.mean()
        registered_axis = initial.axisangle(trans_first=True).clone(); registered_axis[:, 3] += float(index + 1)
        delta = RigidTransform(registered_axis, trans_first=True).compose(initial.inv())
        expected.append(delta.compose(RigidTransform(dicom[index:index + 1], trans_first=True)).axisangle(trans_first=True)[0])
    assert torch.allclose(result.post_stack_init_axisangle_physical, torch.stack(expected), atol=1e-5)
