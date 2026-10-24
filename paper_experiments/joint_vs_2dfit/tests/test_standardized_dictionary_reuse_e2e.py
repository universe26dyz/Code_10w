"""End-to-end CPU contract: generated standardization to audited map reuse."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256
from paper_experiments.joint_vs_2dfit.phase2_clean.fit_first import NATIVE_FIT_METHOD, _dictionary_contract


def _observations(path: Path, count: int) -> None:
    rows = [(group, weight) for group in range(count) for weight in range(10)]
    groups = np.asarray([group for group, _weight in rows], np.int64)
    weights = np.asarray([weight for _group, weight in rows], np.int64)
    affine = np.repeat(np.eye(4)[None], len(rows), axis=0); affine[:, 2, 3] = groups * 5.0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, images=groups[:, None, None].astype(np.float32), masks=np.ones((len(rows), 1, 1), bool), group_idx=groups, weight_idx=weights, stack_idx=np.zeros(len(rows), np.int64), acquisition_time_ms=np.arange(len(rows)), timing9_ms=np.tile(np.arange(9.0), (len(rows), 1)), affine_lps_rc=affine, pixel_spacing_rc_mm=np.ones((len(rows), 2)), slice_thickness_mm=np.ones(len(rows)), tr_ms=np.full(len(rows), 2.61), vps=np.full(len(rows), 87))


def _native(root: Path, observations: Path) -> None:
    with np.load(observations, allow_pickle=False) as source:
        groups = np.unique(source["group_idx"]); first = [np.flatnonzero(source["group_idx"] == group)[0] for group in groups]
        values = groups.astype(np.float32)[:, None, None]
        root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(root / "native_maps.npz", t1_ms=values, t2_ms=values + 1, b1=values + 2, group_idx=groups)
        np.savez_compressed(root / "geometry.npz", affine_lps_rc=source["affine_lps_rc"][first], pixel_spacing_rc_mm=source["pixel_spacing_rc_mm"][first], slice_thickness_mm=source["slice_thickness_mm"][first])
    (root / "manifest.json").write_text(json.dumps({"method": NATIVE_FIT_METHOD, "dictionary_contract": _dictionary_contract(), "source_observations": {"path": str(observations), "sha256": sha256(observations)}, "native_maps_sha256": sha256(root / "native_maps.npz"), "geometry_sha256": sha256(root / "geometry.npz")}), encoding="utf-8")


def test_generated_standardization_manifest_validates_and_reuses_native_maps(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.reuse_standardized_native_dictionary_maps import audit_standardized_native_dictionary_maps, validate_standardized_native_dictionary_reuse
    from paper_experiments.joint_vs_2dfit.phase2_clean.standardize_subject_slices import standardize_subject_slices, validate_standardization_manifest

    prepared, native = tmp_path / "prepared", tmp_path / "native"
    for stack, count in {"sax": 12, "2ch": 16, "4ch": 10}.items():
        observation = prepared / "DYL_2" / stack / "observations.npz"; _observations(observation, count); _native(native / stack, observation)
    standardized = standardize_subject_slices(prepared, "DYL_2", tmp_path / "standardized", git_sha="test")
    manifest = validate_standardization_manifest(standardized / "slice_standardization_manifest.json", subject="DYL_2")
    assert isinstance(manifest["stacks"]["2ch"]["derived_to_original_group_idx"], list)
    reuse = audit_standardized_native_dictionary_maps(standardized / "slice_standardization_manifest.json", native, tmp_path / "reuse")
    payload = json.loads((reuse / "reuse_manifest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "REUSE_VERIFIED"
    assert validate_standardized_native_dictionary_reuse(reuse / "reuse_manifest.json", subject="DYL_2")["subject"] == "DYL_2"
    assert payload["stacks"]["2ch"]["standardized_to_original_group_idx"][0]["original_group_idx"] == 0
    with np.load(reuse / "2ch" / "native_maps.npz") as maps:
        assert maps["group_idx"].tolist() == list(range(15))


def test_standardized_reuse_baseline_exports_without_dictionary_refit_and_uses_sax_masks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import baseline
    from paper_experiments.joint_vs_2dfit.phase2_clean.reuse_standardized_native_dictionary_maps import audit_standardized_native_dictionary_maps
    from paper_experiments.joint_vs_2dfit.phase2_clean.standardize_subject_slices import standardize_subject_slices

    prepared, native = tmp_path / "prepared", tmp_path / "native"
    for stack, count in {"sax": 12, "2ch": 16, "4ch": 10}.items():
        observation = prepared / "DYL_2" / stack / "observations.npz"; _observations(observation, count); _native(native / stack, observation)
    standardized = standardize_subject_slices(prepared, "DYL_2", tmp_path / "standardized", git_sha="test")
    reuse = audit_standardized_native_dictionary_maps(standardized / "slice_standardization_manifest.json", native, tmp_path / "reuse")
    monkeypatch.setattr(baseline, "fit_native", lambda *_args, **_kwargs: pytest.fail("standardized reuse must not refit dictionaries"))

    output, commands = baseline.build_baseline_run(prepared, "DYL_2", tmp_path / "fit_first", standardized_native_maps_root=reuse, standardized_reuse_manifest=reuse / "reuse_manifest.json")

    assert all("--sample-mask" in command and "--volume-mask" not in command and "--stack-masks" not in command for command in commands)
    assert [len(np.load(output / f"selected_standardized_native_maps/native_maps/{stack}/native_maps.npz")["group_idx"]) for stack in ("sax", "2ch", "4ch")] == [12, 15, 10]
    assert all("--registration stack" in command and "--no-transformation-optimization" not in command for command in commands)
    manifest = json.loads((output / "input_manifest.json").read_text())
    assert manifest["standardized_native_dictionary_reuse"]["dictionary_refit"] is False


def test_standardized_sparse_baseline_selects_only_retained_groups_and_skips_empty_lax(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import baseline
    from paper_experiments.joint_vs_2dfit.phase2_clean.reuse_standardized_native_dictionary_maps import audit_standardized_native_dictionary_maps
    from paper_experiments.joint_vs_2dfit.phase2_clean.standardize_subject_slices import standardize_subject_slices

    prepared, native = tmp_path / "prepared", tmp_path / "native"
    for stack, count in {"sax": 12, "2ch": 16, "4ch": 10}.items():
        observation = prepared / "CYJ" / stack / "observations.npz"; _observations(observation, count); _native(native / stack, observation)
    standardized = standardize_subject_slices(prepared, "CYJ", tmp_path / "standardized", git_sha="test")
    reuse = audit_standardized_native_dictionary_maps(standardized / "slice_standardization_manifest.json", native, tmp_path / "reuse")
    standardization = json.loads((standardized / "slice_standardization_manifest.json").read_text())
    sparse_path = tmp_path / "sparse.json"; sparse_path.write_text("{}")
    sparse = {"subject": "CYJ", "profile_id": "L50", "case_id": "L50_fold0", "fold_id": 0, "fold_count": 2, "git_sha": "a" * 40, "_manifest_path": str(sparse_path), "_manifest_sha256": sha256(sparse_path), "stacks": {}}
    for stack, retained in {"sax": list(range(12)), "2ch": list(range(0, 15, 2)), "4ch": []}.items():
        observed = standardization["stacks"][stack]["output_observation"]
        sparse["stacks"][stack] = {"source_observation": observed, "retained_original_group_ids": retained, "training_stack_present": bool(retained)}
    monkeypatch.setattr(baseline, "validate_sparse_profile_manifest", lambda *_args, **_kwargs: sparse)
    output, commands = baseline.build_baseline_run(prepared, "CYJ", tmp_path / "fit_first", standardized_native_maps_root=reuse, standardized_reuse_manifest=reuse / "reuse_manifest.json", sparse_manifest=sparse_path, expected_generator_git_sha="a" * 40, expected_manifest_sha256=sha256(sparse_path))
    assert len(commands) == 2
    assert all("--registration stack" in command and "--no-transformation-optimization" not in command for command in commands)
    assert all("/4ch/" not in command for command in commands)
    assert np.load(output / "selected_standardized_native_maps/native_maps/2ch/native_maps.npz")["group_idx"].tolist() == list(range(0, 15, 2))
    for parameter in ("t1", "t2"):
        target = output / "nesvor" / parameter; target.mkdir(parents=True)
        (target / "model.pt").write_bytes(parameter.encode()); (target / "result.json").write_text(json.dumps({"registration": "stack", "no_slice_scale": True}))
        (target / "slices").mkdir(); (target / "slices/0.nii.gz").write_bytes(b"slice")
    baseline.finalize_scmr_one_stage_fit_first(output)
    route = json.loads((output / "manifest.json").read_text())
    assert route["schema"] == "phase4r_scmr_one_stage_fit_first/v1"
    assert route["active_stacks"] == ["sax", "2ch"]
    assert route["exact_nesvor_commands"] == commands
    assert route["independent_t1_t2"] is True and route["transformation_optimization_enabled"] is True


@pytest.mark.parametrize("mutation, message", [("dictionary", "dictionary contract"), ("geometry", "content or geometry mismatch"), ("mapping", "mapping")])
def test_reuse_audit_fails_closed_on_contract_geometry_or_mapping_mismatch(tmp_path: Path, mutation: str, message: str) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.reuse_standardized_native_dictionary_maps import audit_standardized_native_dictionary_maps
    from paper_experiments.joint_vs_2dfit.phase2_clean.standardize_subject_slices import standardize_subject_slices

    prepared, native = tmp_path / "prepared", tmp_path / "native"
    for stack, count in {"sax": 12, "2ch": 16, "4ch": 10}.items():
        observation = prepared / "CYJ" / stack / "observations.npz"; _observations(observation, count); _native(native / stack, observation)
    standardized = standardize_subject_slices(prepared, "CYJ", tmp_path / "standardized", git_sha="test")
    manifest = standardized / "slice_standardization_manifest.json"
    if mutation == "dictionary":
        path = native / "sax" / "manifest.json"; payload = json.loads(path.read_text()); payload["dictionary_contract"] = {"wrong": True}; path.write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "geometry":
        path = native / "sax" / "geometry.npz"
        with np.load(path, allow_pickle=False) as data: payload = {key: np.asarray(data[key]) for key in data.files}
        payload["pixel_spacing_rc_mm"][0, 0] = 2.0; np.savez_compressed(path, **payload)
        native_manifest = native / "sax" / "manifest.json"; value = json.loads(native_manifest.read_text()); value["geometry_sha256"] = sha256(path); native_manifest.write_text(json.dumps(value), encoding="utf-8")
    else:
        value = json.loads(manifest.read_text()); value["stacks"]["sax"]["derived_to_original_group_idx"] = {"0": 0}; manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        audit_standardized_native_dictionary_maps(manifest, native, tmp_path / "reuse")
    assert not (tmp_path / "reuse").exists()
