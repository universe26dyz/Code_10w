"""Focused Phase-2.1 contract tests; all inputs are small non-formal fixtures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import (
    historical_reference_is_current_compatible,
    real_observation_identities,
)
from paper_experiments.joint_vs_2dfit.phase2_clean.evaluation import (
    reproject_volume_to_native_plane,
    resample_volume_to_grid,
    synthetic_metrics,
)
from paper_experiments.joint_vs_2dfit.phase2_clean.fit_first import (
    build_independent_nesvor_commands,
    export_nesvor_native_inputs,
)
from paper_experiments.joint_vs_2dfit.phase2_clean import proposed
from paper_experiments.joint_vs_2dfit.phase2_clean import native_reference
from paper_experiments.joint_vs_2dfit.phase2_clean.runner import preflight


def _obs(path: Path, *, value: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, images=np.full((10, 2, 2), value, np.float32))
    return path


def test_real_identity_is_subject_specific_and_rejects_mismatch(tmp_path: Path) -> None:
    for sid, value in (("CYJ", 1), ("DYZ", 2)):
        for stack in ("sax", "2ch", "4ch"):
            _obs(tmp_path / sid / stack / "observations.npz", value=value)
    identities = real_observation_identities(tmp_path, subjects=("CYJ", "DYZ"))
    assert identities["CYJ"]["sax"]["path"].endswith("CYJ/sax/observations.npz")
    shared = {"real_observation_sha256": identities}
    method = {"real_observation_sha256": json.loads(json.dumps(identities))}
    assert preflight("fit_first", "real_full", shared, method, "DYZ")["status"] == "preflight_pass"
    method["real_observation_sha256"]["DYZ"]["sax"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="real observation hash mismatch"):
        preflight("fit_first", "real_full", shared, method, "DYZ")


def test_historical_ti_10_100_is_rejected_as_current_reference() -> None:
    assert not historical_reference_is_current_compatible({"TI": [10, 100]})
    assert historical_reference_is_current_compatible({"TI": [100, 180, 260, 340, 420, 500, 580, 660, 740]})


def test_fresh_native_reference_records_current_protocol_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("nibabel")
    observation = _obs(tmp_path / "prepared" / "sax" / "observations.npz")
    def fake_fit(source: Path, output: Path) -> Path:
        output.mkdir(parents=True)
        np.savez(output / "native_maps.npz", t1_ms=np.ones((2, 3, 4)), t2_ms=np.ones((2, 3, 4)), b1=np.ones((2, 3, 4)), group_idx=np.array([0, 1]))
        affine0 = np.eye(4); affine1 = np.eye(4); affine1[2, 3] = 8.0
        np.savez(output / "geometry.npz", affine_lps_rc=np.stack((affine0, affine1)), pixel_spacing_rc_mm=np.ones((2, 2)), slice_thickness_mm=np.array([8.0, 8.0]))
        return output
    monkeypatch.setattr(native_reference, "fit_native", fake_fit)
    out = native_reference.build_current_native_reference(observation, tmp_path / "reference", legacy_config={"TI": [10, 100]})
    manifest = json.loads((out / "native_reference_manifest.json").read_text())
    assert manifest["legacy_reference_status"] == "historical_protocol_rejected_TI_10_100"
    assert manifest["current_protocol_source_observations"]["sha256"]


def test_native_export_preserves_geometry_and_commands_are_independent(tmp_path: Path) -> None:
    pytest.importorskip("nibabel")
    maps = tmp_path / "maps"; maps.mkdir()
    np.savez(maps / "native_maps.npz", t1_ms=np.ones((2, 3, 4)), t2_ms=np.full((2, 3, 4), 2), b1=np.ones((2, 3, 4)), group_idx=np.array([4, 9]))
    affine0 = np.eye(4); affine1 = np.eye(4); affine1[2, 3] = 4.0
    np.savez(maps / "geometry.npz", affine_lps_rc=np.stack((affine0, affine1)), pixel_spacing_mm=np.array([[1.2, 1.3], [1.2, 1.3]]), slice_thickness_mm=np.array([8.0, 8.0]))
    out = tmp_path / "nesvor_input"
    export_nesvor_native_inputs(maps, out)
    assert (out / "t1" / "stack.nii.gz").is_file()
    assert json.loads((out / "manifest.json").read_text())["group_idx"] == [4, 9]
    commands = build_independent_nesvor_commands(out, tmp_path / "recon")
    assert len(commands) == 2 and commands[0] != commands[1]
    assert all("--initial-transforms" not in cmd for cmd in commands)


def test_proposed_command_passes_sha_resolved_decoder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    decoder = tmp_path / "approved.pth"; decoder.write_bytes(b"decoder")
    monkeypatch.setattr(proposed, "APPROVED_DECODER_SHA256", proposed.sha256(decoder))
    command = proposed.build_proposed_command(tmp_path / "prepared", decoder, tmp_path / "b6.pt", tmp_path / "out")
    assert str(decoder) in command and "run_q002_reconstruction" in command
    assert "--anchor-batch-size 640" in command and "--experiment-id Q002S640_B6_roi_same_anchor_mse" in command


def test_affine_resampling_and_native_plane_reprojection() -> None:
    volume = np.zeros((4, 4, 4), np.float32)
    volume[1, 2, 3] = 9.0
    affine = np.array([[0, -1, 0, 10], [1, 0, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]], float)
    grid = np.zeros((4, 4, 4), np.float32)
    sampled, valid = resample_volume_to_grid(volume, affine, grid.shape, affine)
    assert valid.all() and sampled[1, 2, 3] == pytest.approx(9.0)
    plane, plane_valid = reproject_volume_to_native_plane(volume, affine, affine, (4, 4))
    assert plane_valid.all() and plane[1, 2] == pytest.approx(volume[1, 2, 0])


def test_constant_gt_uses_fixed_value_nrmse_and_no_fake_correlation() -> None:
    reference = np.full((2, 2, 2), 1410.0); prediction = reference + 14.1; mask = np.ones_like(reference, bool)
    report = synthetic_metrics(reference, prediction, mask, mask, normalization=1410.0)
    assert report["relative_nrmse"] == pytest.approx(0.01)
    assert report["pearson_r"] is None and report["ncc"] is None
    assert report["correlation_status"] == "not_applicable_constant_reference"


def test_phase2_server_scripts_are_commands_not_echo_templates() -> None:
    root = Path(__file__).resolve().parents[1] / "phase2_clean" / "server_commands"
    for name in ("run_phase2_synthetic_proposed.sh", "run_phase2_synthetic_fit_first.sh", "run_phase2_real_proposed.sh", "run_phase2_real_fit_first.sh", "evaluate_phase2.sh"):
        text = (root / name).read_text()
        assert "TEMPLATE ONLY" not in text
        if name == "evaluate_phase2.sh":
            assert "python -m " in text
        else:
            assert '"$CR_PYTHON" -m ' in text
