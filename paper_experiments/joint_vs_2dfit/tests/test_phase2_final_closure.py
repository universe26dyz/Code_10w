"""Non-formal regression fixtures for the Phase-2 local closure contract."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256
from paper_experiments.joint_vs_2dfit.phase2_clean.fit_first import build_independent_nesvor_commands
from paper_experiments.joint_vs_2dfit.phase2_clean import preflight_actual
from paper_experiments.joint_vs_2dfit.phase2_clean.finalize_phase2_server_readiness import finalize_server_readiness
from paper_experiments.joint_vs_2dfit.phase2_clean.historical_nesvor_runtime import verify_runtime


REPO = Path(__file__).resolve().parents[3]
PHASE2 = REPO / "paper_experiments/joint_vs_2dfit/phase2_clean"


def _write_observations(root: Path, value: int = 1) -> None:
    for stack in ("sax", "2ch", "4ch"):
        path = root / stack / "observations.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, fixture=np.full((1,), value, np.uint8))


def _approved_decoder_audit(path: Path, digest: str, *, declared_path: str = "/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/CYJ_decoder_comparison_20260924_clean_v1/inputs/mlp/signal_simulator_approved.pth") -> Path:
    path.write_text(json.dumps({
        "approved_for_paper": True,
        "approved_by_manual_review": True,
        "validation_status": "approved_by_manual_review",
        "sha256": digest,
        "resolved_from_provenance": {"declared_decoder_path": declared_path},
    }))
    return path


def test_canonical_decoder_authority_rejects_drift_but_allows_duplicate_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = tmp_path / "data"
    relative = preflight_actual.CANONICAL_DECODER_RELATIVE
    canonical = data_root / relative
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"approved")
    duplicate = data_root / "Code_10w_runs/duplicate/approved_copy.pth"
    duplicate.parent.mkdir(parents=True); duplicate.write_bytes(b"approved")
    monkeypatch.setattr(preflight_actual, "APPROVED_DECODER_SHA256", sha256(canonical))
    audit = _approved_decoder_audit(tmp_path / "approved_decoder.json", sha256(canonical))
    approved = preflight_actual.verify_approved_decoder(canonical, audit, data_root)
    assert approved["authority"] == "canonical_path_plus_sha256"
    assert approved["path"] == str(canonical)
    assert duplicate.is_file()  # an identical non-canonical copy is not an ambiguity
    with pytest.raises(FileNotFoundError, match="canonical approved decoder"):
        preflight_actual.verify_approved_decoder(data_root / relative.parent / "missing.pth", audit, data_root)
    canonical.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="approved decoder SHA mismatch"):
        preflight_actual.verify_approved_decoder(canonical, audit, data_root)
    canonical.write_bytes(b"approved")
    bad_audit = _approved_decoder_audit(tmp_path / "bad_decoder.json", sha256(canonical), declared_path="/wrong/provenance.pth")
    with pytest.raises(ValueError, match="declared canonical path"):
        preflight_actual.verify_approved_decoder(canonical, bad_audit, data_root)


def test_final_reference_manifest_is_relocatable_and_complete() -> None:
    manifest = json.loads((PHASE2 / "final_real_reference_manifest.json").read_text())
    assert set(manifest["subjects"]) == {"CYJ", "DYZ", "HHZ", "HJL"}
    for subject, record in manifest["subjects"].items():
        assert record["subject_id"] == subject
        assert record["protocol"] == "current HHZ/TRAD"
        assert record["sax_observation_sha256"]
        assert record["dictionary_grid_provenance"]
        assert record["roi_migration_provenance"]
        for key in ("native_reference", "myocardium_bundle"):
            assert "relative_path" in record[key]
            assert not Path(record[key]["relative_path"]).is_absolute()
            assert record[key]["sha256"]
    assert "new current-HHZ reference" in manifest["subjects"]["CYJ"]["roi_migration_provenance"]


def test_preflight_actual_reads_files_and_rejects_each_integrity_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    synthetic, prepared, reference_root = tmp_path / "synthetic", tmp_path / "prepared", tmp_path / "reference_root"
    _write_observations(synthetic, 1)
    real = {}
    for i, subject in enumerate(("CYJ", "DYZ", "HHZ", "HJL"), start=2):
        subject_root = prepared / subject
        _write_observations(subject_root, i)
        real[subject] = {stack: {"sha256": sha256(subject_root / stack / "observations.npz")} for stack in ("sax", "2ch", "4ch")}
    refs = {"subjects": {}}
    for subject, stacks in real.items():
        native = reference_root / "reference_data/phase2_clean" / subject / "native_reference_2d/native_reference_manifest.json"
        mask = reference_root / "reference_data/phase2_clean" / subject / "myocardium_masks_v1/manifest.json"
        native.parent.mkdir(parents=True, exist_ok=True); mask.parent.mkdir(parents=True, exist_ok=True)
        native.write_text(json.dumps({"current_protocol_source_observations": {"sha256": stacks["sax"]["sha256"]}})); mask.write_text("mask")
        refs["subjects"][subject] = {
            "sax_observation_sha256": stacks["sax"]["sha256"],
            "native_reference": {"relative_path": str(native.relative_to(reference_root)), "sha256": sha256(native)},
            "myocardium_bundle": {"relative_path": str(mask.relative_to(reference_root)), "sha256": sha256(mask)},
        }
    manifest = {"observation_sha256": {stack: sha256(synthetic / stack / "observations.npz") for stack in ("sax", "2ch", "4ch")}, "real_observation_sha256": real}
    manifest_path, refs_path = tmp_path / "manifest.json", tmp_path / "refs.json"
    manifest_path.write_text(json.dumps(manifest)); refs_path.write_text(json.dumps(refs))
    decoder = reference_root / preflight_actual.CANONICAL_DECODER_RELATIVE; decoder.parent.mkdir(parents=True, exist_ok=True); decoder.write_bytes(b"decoder")
    b6 = tmp_path / "b6.pt"; b6.write_bytes(b"b6")
    monkeypatch.setattr(preflight_actual, "APPROVED_DECODER_SHA256", sha256(decoder))
    monkeypatch.setattr(preflight_actual, "B6_CONFIG_SOURCE_SHA256", sha256(b6))
    audit = _approved_decoder_audit(tmp_path / "approved_decoder.json", sha256(decoder))
    result = preflight_actual.verify(manifest_path, prepared, synthetic, decoder, b6, refs_path, reference_root, audit)
    assert result["actual_observation_hash_gate"] and result["references_ready"]
    (synthetic / "sax/observations.npz").write_bytes(b"mismatch")
    with pytest.raises(ValueError, match="synthetic sax SHA mismatch"):
        preflight_actual.verify(manifest_path, prepared, synthetic, decoder, b6, refs_path, reference_root, audit)
    _write_observations(synthetic, 1)
    _write_observations(prepared / "CYJ", 99)
    with pytest.raises(ValueError, match="CYJ/sax SHA mismatch"):
        preflight_actual.verify(manifest_path, prepared, synthetic, decoder, b6, refs_path, reference_root, audit)
    _write_observations(prepared / "CYJ", 2)
    decoder.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="approved decoder SHA mismatch"):
        preflight_actual.verify(manifest_path, prepared, synthetic, decoder, b6, refs_path, reference_root, audit)
    decoder.write_bytes(b"decoder"); b6.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="B6 source SHA mismatch"):
        preflight_actual.verify(manifest_path, prepared, synthetic, decoder, b6, refs_path, reference_root, audit)
    b6.write_bytes(b"b6")
    refs["subjects"]["CYJ"]["sax_observation_sha256"] = "wrong"
    refs_path.write_text(json.dumps(refs))
    with pytest.raises(ValueError, match="reference source SAX SHA mismatch"):
        preflight_actual.verify(manifest_path, prepared, synthetic, decoder, b6, refs_path, reference_root, audit)
    refs["subjects"]["CYJ"]["sax_observation_sha256"] = real["CYJ"]["sax"]["sha256"]
    refs_path.write_text(json.dumps(refs))
    native_path = reference_root / refs["subjects"]["CYJ"]["native_reference"]["relative_path"]
    native_path.write_text(json.dumps({"current_protocol_source_observations": {"sha256": "wrong"}}))
    refs["subjects"]["CYJ"]["native_reference"]["sha256"] = sha256(native_path)
    refs_path.write_text(json.dumps(refs))
    with pytest.raises(ValueError, match="native-reference internal source SAX SHA mismatch"):
        preflight_actual.verify(manifest_path, prepared, synthetic, decoder, b6, refs_path, reference_root, audit)
    refs["subjects"]["CYJ"]["native_reference"]["sha256"] = "not-the-file-hash"
    refs_path.write_text(json.dumps(refs))
    with pytest.raises(ValueError, match="CYJ native_reference"):
        preflight_actual.verify(manifest_path, prepared, synthetic, decoder, b6, refs_path, reference_root, audit)


def test_real_evaluator_cli_writes_subject_level_aggregate(tmp_path: Path) -> None:
    nib = pytest.importorskip("nibabel")
    reference, masks = tmp_path / "reference", tmp_path / "masks"
    reference.mkdir(); masks.mkdir()
    maps = np.array([[[1000.0, 1010.0], [1020.0, 1030.0]], [[1040.0, 1050.0], [1060.0, 1070.0]]])
    np.savez_compressed(reference / "sax.npz", t1_ms=maps, t2_ms=maps / 20.0)
    np.savez_compressed(masks / "sax_myocardium_masks.npz", myocardium_full=np.ones_like(maps, bool))
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(np.moveaxis(maps, 0, 2), affine), reference / "sax_T1_stack_ms.nii.gz")
    nib.save(nib.Nifti1Image(np.moveaxis(maps / 20.0, 0, 2), affine), reference / "sax_T2_stack_ms.nii.gz")
    t1, t2 = tmp_path / "t1.nii.gz", tmp_path / "t2.nii.gz"
    nib.save(nib.Nifti1Image(np.moveaxis(maps + 10.0, 0, 2), affine), t1)
    nib.save(nib.Nifti1Image(np.moveaxis(maps / 20.0 + 1.0, 0, 2), affine), t2)
    output = tmp_path / "real_metrics.json"
    completed = subprocess.run([sys.executable, "-m", "paper_experiments.joint_vs_2dfit.phase2_clean.evaluation", "--mode", "real", "--subject", "CYJ", "--real-t1", str(t1), "--real-t2", str(t2), "--native-reference", str(reference), "--myocardium-bundle", str(masks), "--output", str(output)], cwd=REPO, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text())
    for parameter in ("t1_ms", "t2_ms"):
        aggregate = result["metrics"][parameter]["subject_level"]
        assert aggregate["support_voxels"] == 8
        assert aggregate["coverage_fraction"] == 1.0
        assert aggregate["endpoint"] == "native-reference agreement/consistency, not accuracy versus ground truth"


def test_synthetic_evaluator_cli_writes_fixed_value_nrmse(tmp_path: Path) -> None:
    nib = pytest.importorskip("nibabel")
    t1_gt = np.full((2, 2, 2), 1410.0)
    t2_gt = np.full((2, 2, 2), 50.0)
    gt, mesh = tmp_path / "gt.npz", tmp_path / "mesh.npz"
    np.savez_compressed(gt, t1_ms=t1_gt, t2_ms=t2_gt)
    np.savez_compressed(mesh, spacing_mm=np.ones(3), origin_mm=np.zeros(3), myocardium_mask=np.ones((2, 2, 2), bool))
    t1, t2 = tmp_path / "t1.nii.gz", tmp_path / "t2.nii.gz"
    nib.save(nib.Nifti1Image(t1_gt + 14.1, np.eye(4)), t1)
    nib.save(nib.Nifti1Image(t2_gt + 0.5, np.eye(4)), t2)
    output = tmp_path / "synthetic_metrics.json"
    completed = subprocess.run([sys.executable, "-m", "paper_experiments.joint_vs_2dfit.phase2_clean.evaluation", "--mode", "synthetic", "--synthetic-t1", str(t1), "--synthetic-t2", str(t2), "--gt-maps", str(gt), "--mesh-myo", str(mesh), "--output", str(output)], cwd=REPO, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    metrics = json.loads(output.read_text())["metrics"]
    assert metrics["t1_ms"]["relative_nrmse"] == pytest.approx(0.01)
    assert metrics["t2_ms"]["relative_nrmse"] == pytest.approx(0.01)
    assert metrics["t1_ms"]["correlation_status"] == "not_applicable_constant_reference"


def test_nesvor_failure_prevents_denormalization(tmp_path: Path) -> None:
    commands = build_independent_nesvor_commands(tmp_path / "inputs", tmp_path / "output", nesvor_executable=tmp_path / "bin/nesvor")
    fake = tmp_path / "bin"; fake.mkdir(); nesvor = fake / "nesvor"; nesvor.write_text("#!/usr/bin/env bash\nexit 17\n"); nesvor.chmod(0o755)
    env = dict(os.environ, PATH=f"{fake}:{os.environ['PATH']}")
    completed = subprocess.run(commands[0], shell=True, env=env, text=True, capture_output=True)
    assert completed.returncode == 17
    assert not (tmp_path / "output/t1/volume_ms.nii.gz").exists()


def test_historical_nesvor_command_is_explicit_and_uses_authoritative_flag() -> None:
    executable = "/authoritative/nesvor/bin/nesvor"
    command = build_independent_nesvor_commands("/inputs", "/output", nesvor_executable=executable)[0]
    assert command.startswith(executable + " reconstruct ")
    assert "--n-inference-samples 128" in command
    assert "--inference-samples" not in command
    assert "--pixel-variance-weight 1" in command
    assert "--weight-pixel-var-smooth 0" in command


def _fake_historical_runtime(tmp_path: Path, *, version: str = "0.6.0rc1", flags: str | None = None, package_parent: Path | None = None) -> tuple[Path, Path, Path]:
    source = tmp_path / "NeSVoR-master/nesvor"
    source.mkdir(parents=True)
    for relative in ("cli/parsers.py", "cli/commands.py", "inr/models.py", "inr/train.py"):
        path = source / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("# fixture\n")
    (source / "__init__.py").write_text("")
    (source / "version.py").write_text(f'__version__ = "{version}"\n')
    executable = tmp_path / "bin/nesvor"; executable.parent.mkdir()
    executable.write_text("#!/usr/bin/env bash\nprintf '%s\\n' '" + (flags or "--n-inference-samples --pixel-variance-weight --weight-pixel-var-smooth --scanner-space --no-slice-scale --single-precision --n-samples --batch-size --n-iter --output-psf-factor") + "'\n")
    executable.chmod(0o755)
    runtime_python = tmp_path / "bin/historical-python"
    runtime_python.write_text("#!/usr/bin/env bash\nPYTHONPATH='" + str(package_parent or source.parent) + "' exec '" + sys.executable + "' \"$@\"\n")
    runtime_python.chmod(0o755)
    return executable, runtime_python, source


def test_historical_runtime_gate_accepts_verified_fixture_and_rejects_runtime_drift(tmp_path: Path) -> None:
    executable, runtime_python, source = _fake_historical_runtime(tmp_path)
    report = verify_runtime(executable, runtime_python, source)
    assert report["historical_nesvor_runtime_gate"] is True
    assert report["historical_nesvor"]["version"] == "0.6.0rc1"
    _, wrong_version_python, wrong_version_source = _fake_historical_runtime(tmp_path / "wrong_version", version="0.5")
    with pytest.raises(ValueError, match="version"):
        verify_runtime(tmp_path / "wrong_version/bin/nesvor", wrong_version_python, wrong_version_source)
    _, missing_flag_python, missing_flag_source = _fake_historical_runtime(tmp_path / "missing_flag", flags="--n-inference-samples")
    with pytest.raises(ValueError, match="required flag"):
        verify_runtime(tmp_path / "missing_flag/bin/nesvor", missing_flag_python, missing_flag_source)
    other_parent = tmp_path / "other"; _fake_historical_runtime(other_parent)
    executable, wrong_source_python, source = _fake_historical_runtime(tmp_path / "wrong_source", package_parent=other_parent / "NeSVoR-master")
    with pytest.raises(ValueError, match="package source"):
        verify_runtime(executable, wrong_source_python, source)
    with pytest.raises(FileNotFoundError, match="executable"):
        verify_runtime(tmp_path / "missing/nesvor", runtime_python, source)


def test_formal_scripts_use_actual_preflight_and_all_real_evaluations() -> None:
    scripts = PHASE2 / "server_commands"
    for name in ("run_phase2_synthetic_proposed.sh", "run_phase2_synthetic_fit_first.sh", "run_phase2_real_proposed.sh", "run_phase2_real_fit_first.sh"):
        text = (scripts / name).read_text()
        assert "server_preflight_phase2.sh" in text
        assert "phase2_clean.runner" not in text
    for name in ("run_phase2_synthetic_fit_first.sh", "run_phase2_real_fit_first.sh"):
        assert '--nesvor-executable "$NESVOR_EXEC"' in (scripts / name).read_text()
    for name in ("run_phase2_synthetic_proposed.sh", "run_phase2_real_proposed.sh"):
        text = (scripts / name).read_text()
        assert '["approved_decoder"]["path"]' in text
        assert "approved_decoder_path" not in text
    evaluation = (scripts / "evaluate_phase2.sh").read_text()
    assert "--contract" not in evaluation
    assert all(subject in evaluation for subject in ("CYJ", "DYZ", "HHZ", "HJL"))
    preflight = (scripts / "server_preflight_phase2.sh").read_text()
    assert "PHASE2_NESVOR_EXECUTABLE" in preflight
    assert "PHASE2_NESVOR_PYTHON" in preflight
    assert "historical_nesvor_runtime.py" in preflight
    assert "import torch, tinycudann, paper_experiments" in preflight
    assert "import nesvor" not in preflight
    assert "PHASE2_APPROVED_DECODER" in preflight
    assert "CANONICAL_DECODER" in preflight
    assert "--approved-decoder-audit" in preflight
    assert "CANDIDATES" not in preflight
    assert "approved_decoder_duplicate_sha_matches" in preflight


def test_server_readiness_finalizer_requires_real_passing_preflight(tmp_path: Path) -> None:
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps({
        "actual_observation_hash_gate": True,
        "references_ready": True,
        "approved_decoder_gate": True,
        "b6_config_source_gate": True,
        "historical_nesvor_runtime_gate": True,
        "environment_gate": True,
        "phase1_gate": True,
    }))
    output = tmp_path / "readiness.json"
    written = finalize_server_readiness(preflight, output)
    assert json.loads(written.read_text())["formal_reconstruction_ready"] is True
    missing_historical = json.loads(preflight.read_text())
    del missing_historical["historical_nesvor_runtime_gate"]
    preflight.write_text(json.dumps(missing_historical))
    with pytest.raises(ValueError, match="server preflight did not pass"):
        finalize_server_readiness(preflight, tmp_path / "rejected.json")
