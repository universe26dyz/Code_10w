"""CPU-only contracts for the fixed-00d88 CYJ1.1 r4 reporting plan."""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _case_inputs(root: Path, subject: str, case: str, *, profile: bool) -> None:
    sparse = {
        "subject": subject,
        "condition": "L100" if case == "L100_fold0" else "L50",
        "status": "ELIGIBLE",
        "stacks": {stack: {} for stack in ("sax", "2ch", "4ch")},
    }
    if profile:
        sparse.update({"schema_version": "phase4r_sparse_profile_v2", "profile_id": "SAX100_2CH0_4CH0", "case_id": case, "fold_id": None, "fold_count": 1})
    _write(root / "sparse_inputs" / subject / case / "sparse_manifest.json", json.dumps(sparse))
    _write(root / "endpoints" / "central" / "proposed" / subject / case / "T1_3D.nii.gz")
    _write(root / "endpoints" / "central" / "proposed" / subject / case / "T2_3D.nii.gz")
    _write(root / "endpoints" / "central" / "fit_first" / subject / case / "t1" / "volume_ms.nii.gz")
    _write(root / "endpoints" / "central" / "fit_first" / subject / case / "t2" / "volume_ms.nii.gz")


def _inputs(tmp_path: Path, *, protocol: str = "profile-v2"):
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_r4_reporting import R4Inputs

    root = tmp_path / "analysis"
    subject = "CYJ"
    _write(root / "prepared" / subject / "sax" / "observations.npz")
    _write(root / "prepared" / subject / "2ch" / "observations.npz")
    _write(root / "prepared" / subject / "4ch" / "observations.npz")
    for stack in ("sax", "2ch", "4ch"):
        for name in ("native_maps.npz", "geometry.npz", "manifest.json"):
            _write(root / "native_cache" / subject / stack / name, "{}")
    _write(root / "reconstruction" / "proposed" / subject / "L100_fold0" / "model.pt")
    _case_inputs(root, subject, "L100_fold0", profile=False)
    _write(root / "endpoints" / "central" / "fit_first" / subject / "L100_fold0" / "t1" / "l100_checkpoint_mask.nii.gz")
    _write(root / "endpoints" / "central" / "fit_first" / subject / "L100_fold0" / "t2" / "l100_checkpoint_mask.nii.gz")
    if protocol == "profile-v2":
        _case_inputs(root, subject, "SAX100_2CH0_4CH0", profile=True)
    historical = tmp_path / "fit-00d88"; historical.mkdir()
    direct_native_audit = tmp_path / "direct-native-audit-d596"; direct_native_audit.mkdir()
    cases = ("SAX100_2CH0_4CH0",) if protocol == "profile-v2" else ("L100_fold0",)
    return R4Inputs(subject=subject, protocol=protocol, analysis_root=root, historical_worktree=historical, direct_native_audit_worktree=direct_native_audit, python_executable="python", device="cuda:0", cases=cases)


def _head(_path: Path) -> str:
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_r4_reporting import DIRECT_NATIVE_AUDIT_COMPAT_COMMIT, FIXED_00D88_COMMIT
    return DIRECT_NATIVE_AUDIT_COMPAT_COMMIT if _path.name == "direct-native-audit-d596" else FIXED_00D88_COMMIT


def _step(plan: dict[str, object], identifier: str) -> dict[str, object]:
    return next(item for item in plan["steps"] if item["id"] == identifier)


def test_profile_plan_uses_fixed_worktree_and_separate_r4_namespace(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_r4_reporting import build_r4_plan

    inputs = _inputs(tmp_path)
    plan = build_r4_plan(inputs, git_head=_head)

    assert plan["schema"] == "cyj11_r4_reporting_plan/v1"
    assert plan["fixed_metrics_visualization_commit"] == "00d88b45c75f272f05538997e38fbda85988e7aa"
    assert plan["identity"] == {"subject": "CYJ", "protocol": "profile-v2", "cases": ["SAX100_2CH0_4CH0"]}
    assert not (inputs.analysis_root / "metrics").exists()
    audit = _step(plan, "direct_native_l100_audit")
    assert audit["cwd"] == str(inputs.direct_native_audit_worktree.resolve())
    assert audit["argv"][1:3] == ["-m", "paper_experiments.joint_vs_2dfit.phase2_clean.direct_native_endpoint_audit"]
    metric = _step(plan, "metrics_SAX100_2CH0_4CH0")
    assert metric["cwd"] == str(inputs.historical_worktree.resolve())
    assert metric["argv"][1:3] == ["-m", "paper_experiments.joint_vs_2dfit.phase2_clean.final_stratified_native_metrics"]
    assert "r4_00d88b45c75f" in " ".join(metric["argv"])
    assert "historical" not in " ".join(metric["argv"]).lower()


def test_direct_native_audit_uses_its_checkpoint_compatible_worktree_only(tmp_path: Path) -> None:
    """The support audit must restore r3's post-stack-init checkpoint space.

    Metrics, masks, figures and aggregation remain fixed to 00d88; only the
    compatibility audit is allowed to run from the post-00d repair commit.
    """
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_r4_reporting import (
        DIRECT_NATIVE_AUDIT_COMPAT_COMMIT,
        R4Inputs,
        build_r4_plan,
    )

    inputs = _inputs(tmp_path)
    audit_worktree = inputs.direct_native_audit_worktree
    assert audit_worktree is not None

    def heads(path: Path) -> str:
        return DIRECT_NATIVE_AUDIT_COMPAT_COMMIT if path == audit_worktree else _head(path)

    plan = build_r4_plan(inputs, git_head=heads)

    assert _step(plan, "direct_native_l100_audit")["cwd"] == str(audit_worktree.resolve())
    assert _step(plan, "current_native_masks")["cwd"] == str(inputs.historical_worktree.resolve())
    assert _step(plan, "metrics_SAX100_2CH0_4CH0")["cwd"] == str(inputs.historical_worktree.resolve())


def test_direct_native_audit_is_pinned_to_r3_l100_provenance_compatibility_fix() -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_r4_reporting import DIRECT_NATIVE_AUDIT_COMPAT_COMMIT

    assert DIRECT_NATIVE_AUDIT_COMPAT_COMMIT == "d596308c3816864bb6e5f55a3ce91544f422fc43"


def test_profile_figure_configuration_preserves_explicit_custom_case_identity(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_r4_reporting import build_r4_plan

    plan = build_r4_plan(_inputs(tmp_path, protocol="profile-v2",), git_head=_head, render_figures=True)
    config = _step(plan, "figure_config_SAX100_2CH0_4CH0")["payload"]

    assert config["endpoint"] == "central"
    assert config["profile_id"] == "SAX100_2CH0_4CH0"
    assert config["case_id"] == "SAX100_2CH0_4CH0"
    assert set(config["methods"]) == {"Q002S640", "Fit-first"}
    assert config["native"]["stacks"] == ["sax", "2ch", "4ch"]


def test_plan_uses_explicit_versioned_native_cache_root_everywhere(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_r4_reporting import build_r4_plan

    inputs = _inputs(tmp_path)
    versioned_cache = tmp_path / "versioned_cache" / "CYJ"
    for stack in ("sax", "2ch", "4ch"):
        for name in ("native_maps.npz", "geometry.npz", "manifest.json"):
            _write(versioned_cache / stack / name, "{}")
    plan = build_r4_plan(replace(inputs, r0_native_maps_root=versioned_cache), git_head=_head, render_figures=True)

    assert plan["provenance"]["r0_native_maps_root"] == str(versioned_cache.resolve())
    audit = _step(plan, "direct_native_l100_audit")
    assert audit["argv"][audit["argv"].index("--r0-native-maps-root") + 1] == str(versioned_cache.resolve())
    figure_config = _step(plan, "figure_config_SAX100_2CH0_4CH0")["payload"]
    assert figure_config["native"]["r0_native_maps_root"] == str(versioned_cache.resolve())


def test_legacy_plan_keeps_l100_native_support_separate_from_profile_support(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_r4_reporting import build_r4_plan

    inputs = _inputs(tmp_path, protocol="legacy-density")
    shared_profile_path = (
        inputs.analysis_root / "metrics" / "r4_00d88b45c75f" / "native_support" / "CYJ"
        / "L100_fold0" / "direct_native_endpoint_audit"
    )
    shared_profile_path.mkdir(parents=True)

    plan = build_r4_plan(inputs, git_head=_head)
    audit = _step(plan, "direct_native_l100_audit")
    output = Path(audit["argv"][audit["argv"].index("--output") + 1])
    assert "legacy-density" in output.parts
    assert output != shared_profile_path


def test_plan_rejects_wrong_fixed_commit_and_existing_r4_destination(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_r4_reporting import build_r4_plan

    inputs = _inputs(tmp_path)
    try:
        build_r4_plan(inputs, git_head=lambda _path: "0" * 40)
    except ValueError as error:
        assert "00d88b45" in str(error)
    else:
        raise AssertionError("wrong fixed worktree commit must be rejected")
    (inputs.analysis_root / "metrics" / "central" / "CYJ" / "SAX100_2CH0_4CH0" / "r4_00d88b45c75f").mkdir(parents=True)
    try:
        build_r4_plan(inputs, git_head=_head)
    except FileExistsError as error:
        assert "refusing existing r4 destination" in str(error)
    else:
        raise AssertionError("existing r4 metric destination must be rejected")


def test_metric_comparison_is_read_only_and_only_diffs_identical_keys(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_r4_reporting import compare_metric_trees

    historical = tmp_path / "historical" / "case" / "central" / "aggregate_metrics.json"
    refreshed = tmp_path / "r4" / "case" / "r4_00d88b45c75f" / "aggregate_metrics.json"
    _write(historical, json.dumps({"primary_aggregate": {"t1": {"proposed": {"rmse": 1.0, "mae": 2.0, "bias": 3.0}}}}))
    _write(refreshed, json.dumps({"regions": {"t1": {"primary_aggregate": {"proposed": {"rmse": 1.5, "mae": 2.5, "bias": 2.0}}, "all_lax": {"proposed": {"rmse": 5.0}}}}}))
    output = compare_metric_trees(historical_root=tmp_path / "historical", r4_root=tmp_path / "r4", cases=["case"], output=tmp_path / "comparison.json")

    record = json.loads(output.read_text(encoding="utf-8"))["cases"][0]
    assert record["comparison_status"] == "DIAGNOSTIC_ONLY_SUPPORT_CONTRACT_MUST_BE_REVIEWED"
    assert record["numeric_deltas_for_exactly_shared_keys"][0]["r4_minus_old"] == {"rmse": 0.5, "mae": 0.5, "bias": -1.0}
    assert record["only_r4_keys"] == [{"parameter": "t1", "region": "all_lax", "method": "proposed"}]


def test_r4_runner_dry_run_writes_no_analysis_artifact(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text("#!/bin/sh\nif [ \"$3\" = rev-parse ]; then case \"$2\" in *direct-native-audit-d596) echo d596308c3816864bb6e5f55a3ce91544f422fc43 ;; *) echo 00d88b45c75f272f05538997e38fbda85988e7aa ;; esac; exit 0; fi\nexit 2\n", encoding="utf-8")
    fake_git.chmod(0o755)
    repo = Path(__file__).resolve().parents[3]
    script = repo / "paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_cyj11_r4_reporting.sh"
    env = {
        **os.environ, "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"], "REPO": str(repo),
        "CYJ11_R4_PYTHON": os.sys.executable, "CYJ11_R4_SUBJECT": "CYJ", "CYJ11_R4_PROTOCOL": "profile-v2",
        "CYJ11_R4_ANALYSIS_ROOT": str(inputs.analysis_root), "CYJ11_R4_HISTORICAL_WORKTREE": str(inputs.historical_worktree),
        "CYJ11_R4_DIRECT_NATIVE_AUDIT_WORKTREE": str(inputs.direct_native_audit_worktree),
        "CYJ11_R4_CASES": "SAX100_2CH0_4CH0",
    }
    result = subprocess.run(["bash", str(script), "--dry-run"], cwd=repo, env=env, text=True, capture_output=True, check=True)
    assert '"protocol": "profile-v2"' in result.stdout
    assert not (inputs.analysis_root / "metrics" / "r4_00d88b45c75f").exists()
