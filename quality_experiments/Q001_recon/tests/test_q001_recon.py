import hashlib
import json

import numpy as np
import pytest
import torch

from quality_experiments.Q001_recon.config import build_route_config, load_b6_resolved_config
from quality_experiments.Q001_recon.contracts import verify_q001_input_bundle
from quality_experiments.Q001_recon.evaluation import ROI_BY_STACK, evaluation_plan, fixed_roi
from quality_experiments.Q001_recon.pose_transfer import compose_stack_delta
from quality_experiments.Q001_recon.routes import build_route_plan


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


def test_evaluation_plan_has_central_k32_k8_same_checkpoint():
    plan = evaluation_plan("abc123", "Q001A")
    assert plan["mapping_central_no_psf"]["psf_samples"] == 1
    assert plan["mapping_map_psf_K32"] == {"psf_samples": 32, "seed": 20260911, "checkpoint_sha256": "abc123"}
    assert plan["signal_psf_K8"]["psf_samples"] == 8
