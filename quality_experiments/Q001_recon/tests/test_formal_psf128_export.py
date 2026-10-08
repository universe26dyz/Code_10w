import json
from pathlib import Path

import numpy as np
import torch

from trad.modules.module_08_inference_export import export_quantitative as exporter


class _Space:
    def __init__(self, scaling: float):
        self.center_ras_mm = torch.zeros(3)
        self.spatial_scaling = scaling
        self.physical_bbox_ras_mm = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    def physical_ras_to_train(self, points):
        return points / self.spatial_scaling

    def train_ras_to_physical(self, points):
        return points * self.spatial_scaling


class _INR(torch.nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode
        self.calls = []

    def forward(self, points):
        self.calls.append(points.detach().cpu().clone())
        value = torch.ones_like(points[..., 0]) if self.mode == "constant" else points[..., 0].square()
        return {
            "t1_ms": value,
            "t2_ms": value + 1,
            "b1": value + 2,
            "amplitude": value + 3,
        }


class _Model(torch.nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.inr = _INR(mode)
        self.intensity_scale = torch.tensor(2.0)


def _psf_capture(scaling: float, *, seed: int = 20260911, batch_size: int = 2, mode: str = "nonconstant"):
    model, space = _Model(mode), _Space(scaling)
    fields, affine = exporter.sample_quantitative_fields_psf(
        model, space, bbox_ras_mm=space.physical_bbox_ras_mm, batch_size=batch_size, seed=seed
    )
    return fields, affine, torch.cat(model.inr.calls), space


def test_psf128_samples_physical_mm_before_training_scaling(monkeypatch):
    sigma_mm = 0.125
    monkeypatch.setattr(exporter, "resolution2sigma", lambda resolution, isotropic: sigma_mm)
    _, _, train_at_30, space30 = _psf_capture(30.0)
    _, _, train_at_1, space1 = _psf_capture(1.0)
    physical_at_30 = space30.train_ras_to_physical(train_at_30)
    physical_at_1 = space1.train_ras_to_physical(train_at_1)
    centers, _, _ = exporter._physical_grid(space30.physical_bbox_ras_mm, 1.0)
    centers = centers[:, None, :]
    offsets_mm = physical_at_30 - centers
    # The same physical Gaussian samples are used regardless of spatial scaling.
    assert torch.allclose(physical_at_30, physical_at_1)
    assert torch.allclose(train_at_30 - centers / 30.0, offsets_mm / 30.0)
    assert np.isclose(float(offsets_mm.std()), sigma_mm, rtol=0.2)


def test_psf128_seed_and_chunk_partition_are_deterministic_and_shared():
    fields_a, _, samples_a, _ = _psf_capture(30.0, batch_size=1)
    fields_b, _, samples_b, _ = _psf_capture(30.0, batch_size=4)
    _, _, samples_c, _ = _psf_capture(30.0, seed=20260912, batch_size=4)
    assert samples_a.shape[1:] == (128, 3)
    assert torch.equal(samples_a, samples_b)
    assert not torch.equal(samples_a, samples_c)
    assert all(np.array_equal(fields_a[key], fields_b[key]) for key in fields_a)
    # One INR call receives the complete K=128 coordinate set for all fields.
    assert fields_a["t2_ms"].shape == fields_a["t1_ms"].shape
    assert np.allclose(fields_a["t2_ms"] - fields_a["t1_ms"], 1.0)
    assert np.allclose(fields_a["b1"] - fields_a["t1_ms"], 2.0)
    assert np.allclose(fields_a["amplitude"] / 2.0 - fields_a["t1_ms"], 3.0)


def test_psf128_constant_invariant_nonconstant_changes_and_raw_grid_matches():
    constant, psf_affine, _, space = _psf_capture(30.0, mode="constant")
    raw_constant, raw_affine = exporter.sample_quantitative_fields(
        _Model("constant"), space, 1.0, 3, bbox_ras_mm=space.physical_bbox_ras_mm
    )
    nonconstant, _, _, _ = _psf_capture(30.0, mode="nonconstant")
    raw_nonconstant, _ = exporter.sample_quantitative_fields(
        _Model("nonconstant"), space, 1.0, 3, bbox_ras_mm=space.physical_bbox_ras_mm
    )
    assert np.array_equal(psf_affine, raw_affine)
    assert all(constant[key].shape == raw_constant[key].shape for key in constant)
    assert all(np.allclose(constant[key], raw_constant[key]) for key in constant)
    assert not np.allclose(nonconstant["t1_ms"], raw_nonconstant["t1_ms"])


def test_psf128_chunk_limit_is_explicit_and_never_exceeds_point_budget():
    assert exporter.psf128_effective_voxel_chunk_size(65536, 1_048_576) == 8192
    assert exporter.psf128_effective_voxel_chunk_size(10, 128) == 1
    with np.testing.assert_raises(ValueError):
        exporter.psf128_effective_voxel_chunk_size(1, 127)


def test_psf128_manifest_records_actual_operator_and_shared_bbox_contract(tmp_path):
    model, space = _Model("constant"), _Space(30.0)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    (tmp_path / "final_rigid_poses.json").write_text("{}", encoding="utf-8")
    paths = exporter.export_quantitative_outputs_psf128(
        model,
        space,
        tmp_path,
        dataset=None,
        export_config={"bbox": {"mode": "training_bbox"}},
        batch_size=65536,
        experiment_id="Q001A_full_fov_full_recon",
        subject_id="CYJ",
        source_checkpoint=checkpoint,
        reconstruction_code_git_commit="7f5cf71",
    )
    manifest = json.loads((tmp_path / "formal_export_psf128/formal_export_manifest.json").read_text())
    resolved = exporter.resolve_quantitative_export_bbox(model, space, dataset=None, export_config={"bbox": {"mode": "training_bbox"}})
    _, raw_affine = exporter.sample_quantitative_fields(model, space, 1.0, 3, bbox_ras_mm=resolved)
    assert Path(paths["t1_ms"]).is_file()
    assert manifest["output_resolution_mm"] == 1.0
    assert manifest["output_psf_factor"] == 1.0
    assert manifest["n_inference_samples"] == 128
    assert manifest["export_seed"] == 20260911
    assert manifest["spatial_scaling"] == 30.0
    assert manifest["sigma_physical_mm"] == exporter.resolution2sigma(1.0, isotropic=True)
    assert manifest["requested_output_batch_size"] == 65536
    assert manifest["effective_voxel_chunk_size"] == 8192
    assert manifest["reconstruction_code_git_commit"] == "7f5cf71"
    assert manifest["vendored_nesvor_commit"] == "7f5cf71"
    assert manifest["vendored_nesvor_tree_sha"]
    assert isinstance(manifest["git_dirty"], bool)
    assert manifest["source_checkpoint_sha256"]
    assert manifest["final_rigid_poses_sha256"]
    assert "physical RAS-mm" in manifest["psf_implementation"]
    assert np.array_equal(manifest["affine_ras_mm"], raw_affine.tolist())


def test_q001a_runner_records_code_sha_and_rejects_untracked_worktree():
    root = Path(__file__).resolve().parents[1]
    runner = (root / "run_controlled_reconstruction.py").read_text(encoding="utf-8")
    server = (root.parent / "server_commands" / "run_Q001A_CYJ.sh").read_text(encoding="utf-8")
    assert '"reconstruction_code_git_commit": reconstruction_code_git_commit' in runner
    assert "git status --porcelain" in server
    assert "Refusing non-empty output" in server
