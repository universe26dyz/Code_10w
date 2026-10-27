"""CPU-only planning contracts for read-only legacy r5 figures."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _r5_inputs(root: Path, subject: str = "DYZ") -> None:
    for stack in ("sax", "2ch", "4ch"):
        _write(root / "prepared" / subject / stack / "observations.npz")
        _write(root / "native_cache" / subject / stack / "native_maps.npz")
        _write(root / "native_cache" / subject / stack / "geometry.npz")
        _write(root / "native_cache" / subject / stack / "manifest.json", "{}")
    _write(root / "metrics" / "r4_00d88b45c75f" / "native_support" / subject / "legacy-density" / "L100_fold0" / "current_native_masks" / "manifest.json", "{}")
    _write(root / "reconstruction" / "proposed" / subject / "L100_fold0" / "model.pt")
    for case in ("L100_fold0", "L50_fold0", "L25_fold0"):
        _write(root / "sparse_inputs" / subject / case / "sparse_manifest.json", json.dumps({"subject": subject, "condition": case.split("_fold", 1)[0], "status": "ELIGIBLE"}))
        _write(root / "endpoints" / "central" / "proposed" / subject / case / "T1_3D.nii.gz")
        _write(root / "endpoints" / "central" / "proposed" / subject / case / "T2_3D.nii.gz")
        _write(root / "endpoints" / "central" / "proposed" / subject / case / "manifest.json", "{}")
        _write(root / "endpoints" / "central" / "fit_first" / subject / case / "t1" / "volume_ms.nii.gz")
        _write(root / "endpoints" / "central" / "fit_first" / subject / case / "t2" / "volume_ms.nii.gz")
        _write(root / "endpoints" / "central" / "fit_first" / subject / case / "manifest.json", "{}")
    _write(root / "endpoints" / "central" / "fit_first" / subject / "L100_fold0" / "t1" / "l100_checkpoint_mask.nii.gz")
    _write(root / "endpoints" / "central" / "fit_first" / subject / "L100_fold0" / "t2" / "l100_checkpoint_mask.nii.gz")
    _write(root / "reconstruction" / "fit_first" / subject / "L100_fold0" / "nesvor" / "t1" / "model.pt")


def _heads(path: Path) -> str:
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_r5_legacy_figures import (
        DIRECT_NATIVE_COMPAT_COMMIT,
        FIXED_FIGURE_COMMIT,
    )
    return DIRECT_NATIVE_COMPAT_COMMIT if path.name == "direct-native-d596" else FIXED_FIGURE_COMMIT


def _step(plan: dict, identifier: str) -> dict:
    return next(item for item in plan["steps"] if item["id"] == identifier)


def test_r5_plan_reuses_historical_l100_bundle_chain_and_fixed_renderer(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_r5_legacy_figures import R5Inputs, build_r5_plan

    root = tmp_path / "analysis"
    _r5_inputs(root)
    fixed, direct = tmp_path / "fit-00d88", tmp_path / "direct-native-d596"
    fixed.mkdir(); direct.mkdir()

    plan = build_r5_plan(R5Inputs(subject="DYZ", analysis_root=root, historical_worktree=fixed, direct_native_worktree=direct), git_head=_heads)

    assert plan["schema"] == "cyj11_r5_legacy_figures_plan/v1"
    proposed = _step(plan, "l100_proposed_direct_native_bundle")
    assert proposed["cwd"] == str(direct.resolve())
    assert proposed["argv"][2].endswith("direct_native_bundles")
    pose = _step(plan, "l100_fit_first_pose_audit")
    assert pose["cwd"] == str(fixed.resolve())
    assert pose["argv"][2].endswith("fit_first_native_pose_audit")
    bundle = _step(plan, "l100_fit_first_native_bundle")
    assert bundle["cwd"] == str(fixed.resolve())
    assert bundle["argv"][2].endswith("fit_first_native_bundles")
    l100_config = _step(plan, "figure_config_L100_fold0")["payload"]
    assert "native_bundle" in l100_config["methods"]["Q002S640"]
    assert "fit_first_native_bundle" in l100_config["methods"]["Fit-first"]
    l50_config = _step(plan, "figure_config_L50_fold0")["payload"]
    assert set(l50_config["methods"]["Q002S640"]) == {"t1", "t2"}
    assert set(l50_config["methods"]["Fit-first"]) == {"t1", "t2"}
    assert l50_config["native_mask_root"].endswith("native_support/DYZ/legacy-density/L100_fold0/current_native_masks")
    assert _step(plan, "figures_L25_fold0")["cwd"] == str(fixed.resolve())


def test_r5_plan_refuses_missing_r4_mask_or_existing_figure_destination(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_r5_legacy_figures import R5Inputs, build_r5_plan

    root = tmp_path / "analysis"
    _r5_inputs(root)
    fixed, direct = tmp_path / "fit-00d88", tmp_path / "direct-native-d596"
    fixed.mkdir(); direct.mkdir()
    mask = root / "metrics" / "r4_00d88b45c75f" / "native_support" / "DYZ" / "legacy-density" / "L100_fold0" / "current_native_masks" / "manifest.json"
    mask.unlink()
    with pytest.raises(FileNotFoundError, match="r4 legacy native-mask manifest"):
        build_r5_plan(R5Inputs(subject="DYZ", analysis_root=root, historical_worktree=fixed, direct_native_worktree=direct), git_head=_heads)

    _write(mask, "{}")
    destination = root / "visualizations" / "r5_00d88b45c75f" / "DYZ" / "legacy-density" / "L50_fold0"
    destination.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="destination"):
        build_r5_plan(R5Inputs(subject="DYZ", analysis_root=root, historical_worktree=fixed, direct_native_worktree=direct), git_head=_heads)
