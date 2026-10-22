"""CPU-only contracts for CYJ Experiment 1.1 sparse execution planning."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256


def _obs(path: Path, groups: list[int], *, physical_groups: list[int] | None = None) -> None:
    rows = [(group, weight) for group in groups for weight in range(10)]; ids = np.asarray([a for a, _ in rows]); weights = np.asarray([b for _, b in rows])
    affine = np.repeat(np.eye(4)[None], len(rows), axis=0); affine[:, 2, 3] = np.repeat(physical_groups or groups, 10)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, images=np.ones((len(rows), 1, 1)), masks=np.ones((len(rows), 1, 1), bool), group_idx=ids, weight_idx=weights, affine_lps_rc=affine, pixel_spacing_rc_mm=np.ones((len(rows), 2)), slice_thickness_mm=np.ones(len(rows)), timing9_ms=np.tile(np.arange(9.0), (len(rows), 1)), tr_ms=np.full(len(rows), 2.61), vps=np.full(len(rows), 87))


def _strict_fixture(tmp_path: Path) -> tuple[Path, dict]:
    """A v2 profile with non-zero/non-contiguous IDs and physical ordering."""
    from paper_experiments.joint_vs_2dfit.phase2_clean.sparse import _json_sha256

    source, derived, profiles = tmp_path / "source", tmp_path / "derived", tmp_path / "profiles"
    profile = profiles / "S2CH50_4CH0.json"
    profile.parent.mkdir(parents=True); profile.write_text(json.dumps({"schema": "phase4r_sparse_profile/v2", "profile_id": "S2CH50_4CH0", "stacks": {"sax": {"mode": "all"}, "2ch": {"mode": "stride", "stride": 2}, "4ch": {"mode": "none"}}}))
    stacks = {}
    ordered = [10, 3, 40, 99]
    positions = [1.0, 2.0, 3.0, 4.0]
    for stack in ("sax", "2ch", "4ch"):
        source_path = source / "CYJ" / stack / "observations.npz"; _obs(source_path, [40, 10, 99, 3], physical_groups=[3, 1, 4, 2])
        if stack == "sax": retained = ordered; mode, stride = "all", None
        elif stack == "2ch": retained = [3, 99]; mode, stride = "stride", 2
        else: retained = []; mode, stride = "none", None
        heldout = [group for group in ordered if group not in retained]
        item = {"source_observation": {"path": str(source_path), "sha256": sha256(source_path)}, "mode": mode, "stride": stride, "training_stack_present": bool(retained), "ordered_source_group_ids": ordered, "ordered_source_positions_mm": positions, "stack_normal_lps": [0.0, 0.0, 1.0], "retained_original_group_ids": retained, "heldout_original_group_ids": heldout, "source_count": 4, "retained_count": len(retained), "heldout_count": len(heldout), "weight_integrity": "every source, retained, and held-out group has exactly weights 0..9"}
        if retained:
            derived_path = derived / stack / "observations.npz"; _obs(derived_path, list(range(len(retained))), physical_groups=[positions[ordered.index(group)] for group in retained])
            gap = abs(positions[ordered.index(retained[1])] - positions[ordered.index(retained[0])]) if len(retained) > 1 else 0.0
            item.update({"derived_observation": {"path": str(derived_path), "sha256": sha256(derived_path)}, "derived_to_original_group_idx": [{"derived_group_idx": index, "original_group_idx": group} for index, group in enumerate(retained)], "stack_gap_qc": {"regular_gap": True, "gap_mm": gap, "group_count": len(retained)}})
        stacks[stack] = item
    resolved = {"profile_id": "S2CH50_4CH0", "case_id": "S2CH50_4CH0_fold1", "fold_id": 1, "fold_count": 2, "stacks": {"sax": {"mode": "all"}, "2ch": {"mode": "stride", "stride": 2}, "4ch": {"mode": "none"}}, "profile_path": str(profile.resolve()), "subject": "CYJ", "prepared_root": str((source / "CYJ").resolve()), "stack_stride_overrides": {}, "stack_mode_overrides": {}}
    payload = {"schema_version": "phase4r_sparse_profile_v2", "status": "ELIGIBLE", "subject": "CYJ", "profile_id": "S2CH50_4CH0", "case_id": "S2CH50_4CH0_fold1", "fold_id": 1, "phase": 1, "fold_count": 2, "source_prepared_root": str((source / "CYJ").resolve()), "prepared_root": str(derived.resolve()), "git_sha": "b" * 40, "resolved_config": resolved, "resolved_config_sha256": _json_sha256(resolved), "stacks": stacks}
    manifest = derived / "sparse_manifest.json"; manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, payload


def test_strict_validator_recomputes_profile_physical_selection_and_provenance(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.sparse_profile_manifest import validate_sparse_profile_manifest

    manifest, payload = _strict_fixture(tmp_path)
    validated = validate_sparse_profile_manifest(manifest, subject="CYJ", strict=True, expected_generator_git_sha="b" * 40, expected_manifest_sha256=sha256(manifest))
    assert validated["stacks"]["2ch"]["retained_original_group_ids"] == [3, 99]

    def reject(mutator, match: str) -> None:
        broken = json.loads(json.dumps(payload)); mutator(broken); manifest.write_text(json.dumps(broken), encoding="utf-8")
        with pytest.raises(ValueError, match=match): validate_sparse_profile_manifest(manifest, subject="CYJ", strict=True, expected_generator_git_sha="b" * 40)

    reject(lambda value: value.__setitem__("profile_id", "wrong"), "profile")
    reject(lambda value: value.__setitem__("fold_id", 0), "fold")
    def wrong_selection(value: dict) -> None:
        value["stacks"]["2ch"]["retained_original_group_ids"] = [40, 99]
        value["stacks"]["2ch"]["heldout_original_group_ids"] = [10, 3]
    reject(wrong_selection, "selection")
    reject(lambda value: value.__setitem__("resolved_config_sha256", "0" * 64), "resolved_config")
    reject(lambda value: value.__setitem__("git_sha", "a" * 40), "generator")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest SHA256"):
        validate_sparse_profile_manifest(manifest, subject="CYJ", strict=True, expected_generator_git_sha="b" * 40, expected_manifest_sha256="0" * 64)


@pytest.mark.parametrize("target,field,value,message", [
    ("source", "tr_ms", 2.7, "tr_ms"),
    ("derived", "vps", 86, "vps"),
    ("derived", "images", 2.0, "content/geometry"),
])
def test_strict_validator_rejects_acquisition_and_source_derived_identity_tampering(tmp_path: Path, target: str, field: str, value: float, message: str) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.sparse_profile_manifest import validate_sparse_profile_manifest

    manifest, payload = _strict_fixture(tmp_path / target / field)
    observation = Path(payload["stacks"]["2ch"][f"{target}_observation"]["path"])
    with np.load(observation, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]).copy() for name in source.files}
    arrays[field][0] = value
    np.savez_compressed(observation, **arrays)
    payload["stacks"]["2ch"][f"{target}_observation"]["sha256"] = sha256(observation)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        validate_sparse_profile_manifest(manifest, subject="CYJ", strict=True, expected_generator_git_sha="b" * 40)


@pytest.mark.parametrize("none_stacks", [("2ch",), ("4ch",), ("2ch", "4ch")])
def test_profile_validator_and_k8_launcher_preserve_empty_stack_contract(tmp_path: Path, none_stacks: tuple[str, ...]) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.experiment_1_1 import build_experiment_1_1_command
    from paper_experiments.joint_vs_2dfit.phase2_clean.sparse_profile_manifest import validate_sparse_profile_manifest

    source, derived = tmp_path / "source", tmp_path / "derived"; stacks = {}
    for index, stack in enumerate(("sax", "2ch", "4ch")):
        original = source / "CYJ" / stack / "observations.npz"; _obs(original, [index * 10, index * 10 + 1])
        present = stack not in none_stacks; item = {"source_observation": {"path": str(original), "sha256": sha256(original)}, "training_stack_present": present, "ordered_source_group_ids": [index * 10, index * 10 + 1], "ordered_source_positions_mm": [index * 10.0, index * 10.0 + 1.0], "stack_normal_lps": [0.0, 0.0, 1.0], "retained_original_group_ids": [index * 10, index * 10 + 1] if present else [], "heldout_original_group_ids": [] if present else [index * 10, index * 10 + 1], "source_count": 2, "retained_count": 2 if present else 0, "heldout_count": 0 if present else 2, "stack_gap_qc": {"regular_gap": True, "gap_mm": 1.0, "group_count": 2}}
        if present:
            output = derived / stack / "observations.npz"; _obs(output, [0, 1], physical_groups=[index * 10, index * 10 + 1]); item["derived_observation"] = {"path": str(output), "sha256": sha256(output)}; item["derived_to_original_group_idx"] = [{"derived_group_idx": 0, "original_group_idx": index * 10}, {"derived_group_idx": 1, "original_group_idx": index * 10 + 1}]
        stacks[stack] = item
    manifest = derived / "sparse_manifest.json"; manifest.write_text(json.dumps({"schema_version": "phase4r_sparse_profile_v2", "status": "ELIGIBLE", "subject": "CYJ", "profile_id": "CYJ_test", "case_id": "CYJ_test", "source_prepared_root": str((source / "CYJ").resolve()), "prepared_root": str(derived.resolve()), "git_sha": "a" * 40, "stacks": stacks}), encoding="utf-8")
    validated = validate_sparse_profile_manifest(manifest, subject="CYJ")
    command = build_experiment_1_1_command(validated, expected_generator_git_sha="a" * 40, b6_model="/b6.pt", decoder="/decoder.pt", output=tmp_path / "fresh", device="cuda:0")
    assert "Q002S640_B6_roi_same_anchor_mse" in command and "--anchor-batch-size 640" in command and "--sparse-manifest" in command and "--expected-sparse-generator-git-sha" in command
    for stack in none_stacks: assert not (derived / stack / "observations.npz").exists()

    from paper_experiments.joint_vs_2dfit.phase2_clean.baseline import build_baseline_run
    from paper_experiments.joint_vs_2dfit.phase2_clean.fit_first import fit_native
    from quality_experiments.Q002_B6_roi_same_anchor_mse.run_q002_reconstruction import sparse_profile_provenance

    native_cache = tmp_path / "native_cache"
    for stack in ("sax", "2ch", "4ch"):
        fit_native(source / "CYJ" / stack / "observations.npz", native_cache / stack)
    fit_first, commands = build_baseline_run(derived, "CYJ", tmp_path / "fit_first", reuse_native_maps_root=native_cache, sparse_manifest=manifest)
    reuse = json.loads((fit_first / "native_maps" / "reuse_manifest.json").read_text())
    for stack in none_stacks:
        assert not (fit_first / "native_maps" / stack).exists()
        assert reuse["stacks"][stack]["heldout_native_geometry_evaluation"].startswith("preserved")
        assert str(fit_first / "nesvor_inputs" / stack) not in "\n".join(commands)
    assert json.loads((fit_first / "input_manifest.json").read_text())["case_id"] == "CYJ_test"
    provenance = sparse_profile_provenance(validated, [derived / stack / "observations.npz" for stack in ("sax", "2ch", "4ch") if stack not in none_stacks])
    assert provenance["case_id"] == "CYJ_test" and provenance["sparse_manifest"]["sha256"] == sha256(manifest)

    corrupted = json.loads(manifest.read_text())
    corrupted["stacks"]["sax"]["derived_to_original_group_idx"][0]["original_group_idx"] = 999
    manifest.write_text(json.dumps(corrupted), encoding="utf-8")
    with pytest.raises(ValueError, match="derived_to_original"):
        validate_sparse_profile_manifest(manifest, subject="CYJ")
    manifest.write_text(json.dumps({key: value for key, value in validated.items() if not key.startswith("_")}), encoding="utf-8")
    fake_empty = derived / none_stacks[0] / "observations.npz"; _obs(fake_empty, [0])
    with pytest.raises(ValueError, match="empty training stack"):
        validate_sparse_profile_manifest(manifest, subject="CYJ")
