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
    command = build_experiment_1_1_command(validated, b6_model="/b6.pt", decoder="/decoder.pt", output=tmp_path / "fresh", device="cuda:0")
    assert "Q002S640_B6_roi_same_anchor_mse" in command and "--anchor-batch-size 640" in command and "--sparse-manifest" in command
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
