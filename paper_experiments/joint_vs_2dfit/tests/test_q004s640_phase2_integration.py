"""CPU contracts for Q004S640 in the CYJ fixed-L100 comparison plan."""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


def _manifest(path: Path, condition: str) -> Path:
    path.write_text(json.dumps({"subject": "CYJ", "condition": condition.split("_", 1)[0]}), encoding="utf-8")
    return path


def _volume(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.ones((1, 1, 1), np.float32), np.eye(4)), path)


def test_q004_config_differs_from_q002s640_only_in_k_related_training_fields() -> None:
    from quality_experiments.Q002_B6_roi_same_anchor_mse.contracts import Q002S640_EXPERIMENT_ID, Q004S640_K1_EXPERIMENT_ID, build_q002_route_config

    base = {"training": {"fixed": "B6"}, "decoder": {"approved": "FrozenMLP"}}
    q002 = build_q002_route_config(base, cropped_prepared_root="/cropped", experiment_id=Q002S640_EXPERIMENT_ID, anchor_batch_size=640)
    q004 = build_q002_route_config(base, cropped_prepared_root="/cropped", experiment_id=Q004S640_K1_EXPERIMENT_ID, anchor_batch_size=640)
    assert q004["resolved_config"] == q002["resolved_config"]
    assert {key: value for key, value in q004["route"]["sampling"].items() if key not in {"shared_training_psf_samples", "data_psf_inr_location_count"}} == {key: value for key, value in q002["route"]["sampling"].items() if key not in {"shared_training_psf_samples", "data_psf_inr_location_count"}}
    assert q002["route"]["sampling"]["shared_training_psf_samples"] == 8
    assert q004["route"]["sampling"]["shared_training_psf_samples"] == 1
    assert q004["route"]["scientific_change"] == "training_psf_samples_8_to_central_1_only"


def test_q004_phase2_adapter_selects_the_preexisting_k1_runner_and_r0_reuse(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.q004s640 import build_q004s640_command, select_cyj_prepared_root

    r0, sparse = tmp_path / "r0", tmp_path / "sparse"; r0.mkdir(); sparse.mkdir()
    l100_manifest = _manifest(tmp_path / "l100.json", "L100_fold0")
    l50_manifest = _manifest(tmp_path / "l50.json", "L50_fold0")
    assert select_cyj_prepared_root("L100_fold0", r0, r0, l100_manifest) == r0
    with pytest.raises(ValueError, match="L100"):
        select_cyj_prepared_root("L100_fold0", r0, sparse, l100_manifest)
    assert select_cyj_prepared_root("L50_fold0", r0, sparse, l50_manifest) == sparse
    command = build_q004s640_command("L50_fold0", sparse, decoder="decoder.pt", b6_model="b6.pt", output="new-output", device="cuda:0")
    assert "run_q004_s640_k1_reconstruction" in command
    assert "Q004S640_K1_B6_roi_same_anchor_mse" in command
    assert "--anchor-batch-size 640" in command


def test_q004_phase2_adapter_preserves_an_explicit_non_cyj_subject(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.q004s640 import build_q004s640_command, select_prepared_root

    r0 = tmp_path / "r0"; r0.mkdir()
    manifest = tmp_path / "l100.json"
    manifest.write_text(json.dumps({"subject": "DYL_2", "condition": "L100"}), encoding="utf-8")
    assert select_prepared_root("DYL_2", "L100_fold0", r0, r0, manifest) == r0
    command = build_q004s640_command("L100_fold0", r0, subject="DYL_2", decoder="decoder.pt", b6_model="b6.pt", output="new-output")
    assert "--subject-id DYL_2" in command


def test_q004_alias_records_k1_training_separately_from_central_and_psf128_exports(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj_matched_comparison import formalize_q004s640_alias

    root = tmp_path / "q004"; root.mkdir(); (root / "model.pt").write_bytes(b"checkpoint")
    (root / "experiment_manifest.json").write_text(json.dumps({
        "experiment_id": "Q004S640_K1_B6_roi_same_anchor_mse",
        "parent_experiment": "Q002S640_B6_roi_same_anchor_mse",
        "scientific_change": "training_psf_samples_8_to_central_1_only",
        "training_psf_samples": 1,
        "loss": "MSE_only",
        "fingerprint_cosine_enabled": False,
    }), encoding="utf-8")
    _volume(root / "T1_3D.nii.gz"); _volume(root / "T2_3D.nii.gz")
    psf = root / "formal_export_psf128"; _volume(psf / "T1_3D.nii.gz"); _volume(psf / "T2_3D.nii.gz")
    (psf / "formal_export_manifest.json").write_text(json.dumps({"n_inference_samples": 128}), encoding="utf-8")
    central = formalize_q004s640_alias(subject="CYJ", condition="L100_fold0", source_root=root, endpoint="central", output=tmp_path / "central")
    psf128 = formalize_q004s640_alias(subject="CYJ", condition="L100_fold0", source_root=root, endpoint="psf128", output=tmp_path / "psf128")
    central_manifest = json.loads((central / "manifest.json").read_text())
    psf128_manifest = json.loads((psf128 / "manifest.json").read_text())
    assert central_manifest["training_route"] == "Q004S640"
    assert central_manifest["training_psf"] == {"samples": 1, "label": "K1", "semantics": "central local coordinate training PSF"}
    assert central_manifest["export_psf"]["n_inference_samples"] == 1
    assert psf128_manifest["export_psf"]["n_inference_samples"] == 128


def test_q004_phase2_server_wrappers_are_immutable_and_restrict_cyj_cases() -> None:
    root = Path(__file__).resolve().parents[3]
    scripts = root / "paper_experiments" / "joint_vs_2dfit" / "phase2_clean" / "server_commands"
    run = (scripts / "run_q004s640_cyj.sh").read_text()
    aliases = (scripts / "formalize_q004s640_aliases_cyj.sh").read_text()
    evaluate = (scripts / "evaluate_q004s640_matched_case.sh").read_text()
    cases = "L100_fold0|L50_fold0|L50_fold1|L25_fold0|L25_fold1|L25_fold2|L25_fold3"
    for text in (run, aliases, evaluate):
        assert cases in text and "L33" not in text and "rm -rf" not in text
    assert "phase2_clean.q004s640 --subject" in run and "--condition" in run and "--execute" in run
    assert 'SUBJECT=${SUBJECT:-CYJ}' in run and '--subject "$SUBJECT"' in run
    assert "alias-q004s640" in aliases and "--endpoint central" in aliases and "--endpoint psf128" in aliases
    assert "FIT_FIRST_OR_Q002_OR_Q002S640_K256" in evaluate
    assert "--proposed-label Q004S640" in evaluate and "--comparator-label \"$LABEL\"" in evaluate
    assert "MASK_REFERENCE_EXPORT_ROOT" in evaluate
    assert "--comparator-export-root \"$COMPARATOR\"" in evaluate
    assert "--mask-reference-export-root \"$MASK_REFERENCE\"" in evaluate
