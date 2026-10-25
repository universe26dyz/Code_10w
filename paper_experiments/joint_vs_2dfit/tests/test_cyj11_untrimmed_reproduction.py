"""CPU-only contract tests for the strict CYJ 1.1 command-plan builder."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _observation(path: Path, groups: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    group_idx = np.repeat(np.arange(groups, dtype=np.int64), 10)
    np.savez_compressed(path, group_idx=group_idx, weight_idx=np.tile(np.arange(10), groups))


def _inputs(tmp_path: Path, *, group_counts: tuple[int, int, int] = (14, 15, 12)):
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_untrimmed_reproduction import ReproductionInputs

    prepared = tmp_path / "Code_10w_prepared"
    for stack, groups in zip(("sax", "2ch", "4ch"), group_counts):
        _observation(prepared / "CYJ" / stack / "observations.npz", groups)
    assets = tmp_path / "assets"
    assets.mkdir()
    b6 = assets / "b6.pt"; b6.write_bytes(b"b6")
    simulator = assets / "signal_simulator.pth"; simulator.write_bytes(b"simulator")
    protocol = assets / "protocol.yaml"; protocol.write_text("protocol: historical\n", encoding="utf-8")
    native_cache = assets / "native_cache"; native_cache.mkdir()
    native_mask = assets / "native_mask"; native_mask.mkdir(); (native_mask / "manifest.json").write_text("{}", encoding="utf-8")
    l100_proposed = assets / "l100_proposed"; l100_proposed.mkdir()
    l100_fit_first = assets / "l100_fit_first"; l100_fit_first.mkdir()
    worktrees = {name: tmp_path / f"{name}_worktree" for name in ("sparse", "q002", "fit_first")}
    for path in worktrees.values():
        path.mkdir()
    return ReproductionInputs(
        subject="CYJ",
        case_id="SAX100_2CH100_4CH0",
        prepared_root=prepared,
        output_root=tmp_path / "new_output",
        sparse_worktree=worktrees["sparse"],
        q002_worktree=worktrees["q002"],
        fit_first_worktree=worktrees["fit_first"],
        b6_model=b6,
        signal_simulator=simulator,
        protocol=protocol,
        native_cache_root=native_cache,
        native_mask_root=native_mask,
        l100_proposed_root=l100_proposed,
        l100_fit_first_root=l100_fit_first,
        expected_b6_model_sha256=_sha256(b6),
        expected_signal_simulator_sha256=_sha256(simulator),
        expected_protocol_sha256=_sha256(protocol),
    )


def _historical_heads(_path: Path) -> str:
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_untrimmed_reproduction import (
        FIT_FIRST_COMMIT,
        Q002_COMMIT,
        SPARSE_COMMIT,
    )

    return {"sparse_worktree": SPARSE_COMMIT, "q002_worktree": Q002_COMMIT, "fit_first_worktree": FIT_FIRST_COMMIT}[_path.name]


def test_plan_preserves_historical_q002_and_fit_first_sampling_contract(tmp_path: Path) -> None:
    """Removing historical sampling or adding Final5 central-only changes must fail this test."""
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_untrimmed_reproduction import build_reproduction_plan

    inputs = _inputs(tmp_path)
    plan = build_reproduction_plan(inputs, git_head=_historical_heads)
    assert not inputs.output_root.exists()
    assert plan["schema"] == "cyj11_untrimmed_reproduction_plan/v1"
    assert plan["identity"] == {"subject": "CYJ", "case_id": "SAX100_2CH100_4CH0"}
    q002 = next(step["argv"] for step in plan["steps"] if step["id"] == "proposed_sparse")
    assert "Q002S640_B6_roi_same_anchor_mse" in q002
    assert q002[q002.index("--anchor-batch-size") + 1] == "640"
    assert "--endpoint-policy" not in q002
    fit_first = next(step for step in plan["steps"] if step["id"] == "fit_first_sparse_prepare")
    assert "baseline" in " ".join(fit_first["argv"])
    for parameter, sample_mask in fit_first["expected_output_sampling_masks"].items():
        assert sample_mask.endswith(f"/nesvor_inputs/sax/{parameter}/stack.nii.gz")
        assert "stack_mask.nii.gz" not in sample_mask
    publications = [step for step in plan["steps"] if step["id"].endswith("_central")]
    assert {step["id"] for step in publications} == {"proposed_central", "fit_first_central", "metric_central"}
    assert all("central" in " ".join(step["argv"]) for step in publications)


@pytest.mark.parametrize("counts", [(12, 15, 10), (14, 15, 11)])
def test_plan_rejects_standardized_or_nonhistorical_cyj_group_layout(tmp_path: Path, counts: tuple[int, int, int]) -> None:
    """Accepting Final5's 12/15/10 route would invalidate strict historical reproduction."""
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_untrimmed_reproduction import build_reproduction_plan

    with pytest.raises(ValueError, match="14 SAX, 15 2CH, and 12 4CH"):
        build_reproduction_plan(_inputs(tmp_path, group_counts=counts), git_head=_historical_heads)


def test_plan_rejects_wrong_stage_commit_and_existing_output(tmp_path: Path) -> None:
    """A changed historical code stage or occupied target must fail before command execution."""
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_untrimmed_reproduction import build_reproduction_plan

    inputs = _inputs(tmp_path)
    def wrong_q002_head(path: Path) -> str:
        return "0" * 40 if path.name == "q002_worktree" else _historical_heads(path)

    with pytest.raises(ValueError, match="q002 worktree"):
        build_reproduction_plan(inputs, git_head=wrong_q002_head)
    inputs.output_root.mkdir()
    (inputs.output_root / "existing.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing non-empty output root"):
        build_reproduction_plan(inputs, git_head=_historical_heads)


def test_plan_rejects_changed_explicit_input_or_dependency_hash(tmp_path: Path) -> None:
    """Hash pins remain content checks even when paths are local mirrors."""
    from paper_experiments.joint_vs_2dfit.phase2_clean.cyj11_untrimmed_reproduction import ReproductionInputs, build_reproduction_plan

    inputs = _inputs(tmp_path)
    mismatched = ReproductionInputs(**{**inputs.__dict__, "expected_b6_model_sha256": "0" * 64})
    with pytest.raises(ValueError, match="B6 model SHA256"):
        build_reproduction_plan(mismatched, git_head=_historical_heads)
    mismatched_decoder = ReproductionInputs(**{**inputs.__dict__, "expected_signal_simulator_sha256": "0" * 64})
    with pytest.raises(ValueError, match="signal simulator SHA256"):
        build_reproduction_plan(mismatched_decoder, git_head=_historical_heads)


def test_server_wrapper_dry_run_prints_plan_without_creating_output(tmp_path: Path) -> None:
    """Dry-run must validate and print deterministic commands without GPU/output side effects."""
    inputs = _inputs(tmp_path)
    fake_bin = tmp_path / "bin"; fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text(
        "#!/bin/sh\ncase \"$2\" in\n"
        "*sparse_worktree) echo 6f4142cccc17782ba6e985e74531822560b369f3 ;;\n"
        "*q002_worktree) echo de41aef1c52b12037efed20f71cdf0db4696d49c ;;\n"
        "*fit_first_worktree) echo 00d88b45c75f272f05538997e38fbda85988e7aa ;;\n"
        "*) exit 2 ;;\nesac\n",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    nesvor = tmp_path / "assets" / "nesvor"; nesvor.write_text("#!/bin/sh\n", encoding="utf-8"); nesvor.chmod(0o755)
    repo = Path(__file__).resolve().parents[3]
    env = {
        **os.environ,
        "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
        "REPO": str(repo),
        "CYJ11_PYTHON": os.environ.get("PYTHON", os.sys.executable),
        "CYJ11_PREPARED_ROOT": str(inputs.prepared_root),
        "CYJ11_B6_MODEL": str(inputs.b6_model),
        "CYJ11_SIGNAL_SIMULATOR": str(inputs.signal_simulator),
        "CYJ11_PROTOCOL": str(inputs.protocol),
        "CYJ11_NATIVE_CACHE_ROOT": str(inputs.native_cache_root),
        "CYJ11_NATIVE_MASK_ROOT": str(inputs.native_mask_root),
        "CYJ11_L100_PROPOSED_ROOT": str(inputs.l100_proposed_root),
        "CYJ11_L100_FIT_FIRST_ROOT": str(inputs.l100_fit_first_root),
        "CYJ11_OUTPUT_ROOT": str(inputs.output_root),
        "CYJ11_SPARSE_WORKTREE": str(inputs.sparse_worktree),
        "CYJ11_Q002_WORKTREE": str(inputs.q002_worktree),
        "CYJ11_FIT_FIRST_WORKTREE": str(inputs.fit_first_worktree),
        "CYJ11_EXPECTED_B6_SHA256": inputs.expected_b6_model_sha256,
        "CYJ11_EXPECTED_SIGNAL_SIMULATOR_SHA256": inputs.expected_signal_simulator_sha256,
        "CYJ11_EXPECTED_PROTOCOL_SHA256": inputs.expected_protocol_sha256,
        "CYJ11_NESVOR_EXECUTABLE": str(nesvor),
    }
    script = repo / "paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_cyj11_untrimmed_reproduction.sh"
    result = subprocess.run(["bash", str(script), "--dry-run"], cwd=repo, env=env, text=True, capture_output=True, check=True)
    assert '"case_id": "SAX100_2CH100_4CH0"' in result.stdout
    assert not inputs.output_root.exists()
