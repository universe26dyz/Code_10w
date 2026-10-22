"""CPU contracts for explicit CYJ sparse-profile v2 inputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _observations(path: Path, ids: list[int]) -> None:
    rows = [(group, weight) for group in ids for weight in range(10)]
    groups = np.asarray([group for group, _weight in rows], dtype=np.int64)
    weights = np.asarray([weight for _group, weight in rows], dtype=np.int64)
    affine = np.repeat(np.eye(4)[None], len(rows), axis=0)
    positions = {group: float(index * 5) for index, group in enumerate(ids)}
    affine[:, 2, 3] = [positions[int(group)] for group in groups]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, images=np.ones((len(rows), 2, 3), np.float32), masks=np.ones((len(rows), 2, 3), bool), group_idx=groups, weight_idx=weights, stack_idx=np.zeros(len(rows), np.int64), acquisition_time_ms=np.arange(len(rows), dtype=float), timing9_ms=np.tile(np.arange(1.0, 10.0), (len(rows), 1)), affine_lps_rc=affine, pixel_spacing_rc_mm=np.ones((len(rows), 2)), slice_thickness_mm=np.full(len(rows), 8.0), tr_ms=np.full(len(rows), 2.61), vps=np.full(len(rows), 87, np.int64))


def _prepared(root: Path) -> Path:
    _observations(root / "CYJ" / "sax" / "observations.npz", list(range(100, 114)))
    _observations(root / "CYJ" / "2ch" / "observations.npz", list(range(200, 215)))
    _observations(root / "CYJ" / "4ch" / "observations.npz", list(range(300, 312)))
    return root


def _profile(tmp_path: Path, profile_id: str, stacks: dict[str, dict[str, object]]) -> Path:
    path = tmp_path / f"{profile_id}.json"
    path.write_text(json.dumps({"schema": "phase4r_sparse_profile/v2", "profile_id": profile_id, "stacks": stacks}), encoding="utf-8")
    return path


def test_profile_uses_physical_ordinals_for_complementary_asymmetric_folds(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.sparse import build_sparse_profile_condition

    profile = _profile(tmp_path, "S2CH50_4CH100", {"sax": {"mode": "all"}, "2ch": {"mode": "stride", "stride": 2}, "4ch": {"mode": "all"}})
    first = build_sparse_profile_condition(_prepared(tmp_path / "prepared"), "CYJ", profile, 0, tmp_path / "out", git_sha="test", audit_stack=lambda _path: {"regular_gap": True, "gap_mm": 5.0})
    second = build_sparse_profile_condition(tmp_path / "prepared", "CYJ", profile, 1, tmp_path / "out", git_sha="test", audit_stack=lambda _path: {"regular_gap": True, "gap_mm": 5.0})
    left, right = (json.loads(Path(item["manifest_path"]).read_text(encoding="utf-8")) for item in (first, second))
    assert left["stacks"]["2ch"]["retained_original_group_ids"] == [200, 202, 204, 206, 208, 210, 212, 214]
    assert right["stacks"]["2ch"]["retained_original_group_ids"] == [201, 203, 205, 207, 209, 211, 213]
    assert left["stacks"]["2ch"]["source_count"] == 15
    assert left["stacks"]["4ch"]["retained_count"] == 12
    assert set(left["stacks"]["2ch"]["retained_original_group_ids"]).isdisjoint(right["stacks"]["2ch"]["retained_original_group_ids"])
    assert set(left["stacks"]["2ch"]["retained_original_group_ids"]) | set(right["stacks"]["2ch"]["retained_original_group_ids"]) == set(range(200, 215))
    assert left["no_fake_observations"] is True and left["schema_version"] == "phase4r_sparse_profile_v2"


def test_none_stack_has_no_derived_observation_but_exhaustive_heldout_source_ids(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.sparse import build_sparse_profile_condition

    profile = _profile(tmp_path, "SAX100_2CH0_4CH0", {"sax": {"mode": "all"}, "2ch": {"mode": "none"}, "4ch": {"mode": "none"}})
    result = build_sparse_profile_condition(_prepared(tmp_path / "prepared"), "CYJ", profile, None, tmp_path / "out", git_sha="test", audit_stack=lambda _path: {"regular_gap": True, "gap_mm": 5.0})
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["status"] == "ELIGIBLE"
    assert manifest["stacks"]["2ch"]["training_stack_present"] is False
    assert manifest["stacks"]["2ch"]["retained_original_group_ids"] == []
    assert manifest["stacks"]["2ch"]["heldout_original_group_ids"] == list(range(200, 215))
    assert not (Path(result["prepared_root"]) / "2ch" / "observations.npz").exists()


def test_profile_dry_run_resolves_overrides_without_writing(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.sparse import build_sparse_profile_condition

    profile = _profile(tmp_path, "S2CH100_4CH50", {"sax": {"mode": "all"}, "2ch": {"mode": "all"}, "4ch": {"mode": "stride", "stride": 2}})
    result = build_sparse_profile_condition(_prepared(tmp_path / "prepared"), "CYJ", profile, 1, tmp_path / "out", stack_strides={"4ch": 2}, stack_modes={"2ch": "none"}, dry_run=True, git_sha="test")
    assert result["dry_run"] is True
    assert result["resolved_config"]["stacks"]["2ch"]["mode"] == "none"
    assert not (tmp_path / "out").exists()


def test_profile_rejects_unknown_configuration_keys(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.sparse import load_sparse_profile

    path = tmp_path / "bad.json"; path.write_text(json.dumps({"schema": "phase4r_sparse_profile/v2", "profile_id": "bad", "stacks": {}, "unknown": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_sparse_profile(path)


def test_profile_cli_prints_resolved_dry_run_without_output_write(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import sparse

    profile = _profile(tmp_path, "S2CH50_4CH100", {"sax": {"mode": "all"}, "2ch": {"mode": "stride", "stride": 2}, "4ch": {"mode": "all"}})
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["sparse", "--prepared-root", str(_prepared(tmp_path / "prepared")), "--subject", "CYJ", "--profile", str(profile), "--fold", "0", "--output-root", str(output), "--stack-stride", "2ch=2", "--dry-run", "--print-resolved-config"])
    sparse.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True and payload["resolved_config"]["case_id"] == "S2CH50_4CH100_fold0"
    assert not output.exists()
