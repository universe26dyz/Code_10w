"""Build the immutable, independent CYJ1.1 r3 reconstruction plan.

This is deliberately a planner: it validates provenance and emits commands,
but neither creates result directories nor runs reconstruction.  The paired
server runner owns the filesystem copy/archive and command execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


PLAN_SCHEMA = "cyj11_r3_reproduction_plan/v1"
DAD9698_COMMIT = "dad9698b5b2b5cd98267a6d23f1eb866b54791d3"
CASES = ("L100_fold0", "L50_fold0", "L25_fold0")
CASE_PARAMETERS = {
    "L100_fold0": ("L100", 0),
    "L50_fold0": ("L50", 0),
    "L25_fold0": ("L25", 0),
}
SUBJECTS = ("DYZ", "HJL", "HHZ", "DYL_2")
STACKS = ("sax", "2ch", "4ch")
EXPECTED_GROUP_COUNTS = {"sax": 12, "2ch": 16, "4ch": 10}


@dataclass(frozen=True)
class R3Inputs:
    subject: str
    source_prepared_root: Path
    analysis_root: Path
    historical_worktree: Path
    b6_model: Path
    signal_simulator: Path
    nesvor_executable: Path
    expected_b6_model_sha256: str
    expected_signal_simulator_sha256: str
    python_executable: str = sys.executable
    device: str = "cuda:0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError as error:
        raise ValueError(f"cannot determine git HEAD for r3 historical worktree: {path}") from error


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _audit_observations(path: Path) -> dict[str, object]:
    _require_file(path, "prepared observation")
    with np.load(path, allow_pickle=False) as payload:
        if "group_idx" not in payload or "weight_idx" not in payload:
            raise ValueError(f"{path}: expected group_idx and weight_idx arrays")
        groups = np.asarray(payload["group_idx"], dtype=int)
        weights = np.asarray(payload["weight_idx"], dtype=int)
    if groups.shape != weights.shape or groups.ndim != 1:
        raise ValueError(f"{path}: group_idx and weight_idx must be paired one-dimensional arrays")
    identities = np.unique(groups)
    for group in identities:
        if not np.array_equal(np.sort(weights[groups == group]), np.arange(10)):
            raise ValueError(f"{path}: group {int(group)} does not contain exactly weights 0..9")
    return {"path": str(path.resolve()), "sha256": _sha256(path), "groups": [int(value) for value in identities]}


def _sampling_device(device: str) -> str:
    value = str(device).strip()
    if value == "-1" or value.isdecimal():
        return value
    if value.startswith("cuda:") and value.removeprefix("cuda:").isdecimal():
        return value.removeprefix("cuda:")
    raise ValueError("device must be cuda:N, a CUDA index N, or -1")


def _command(*items: str | Path) -> list[str]:
    return [str(item) for item in items]


def _validate(inputs: R3Inputs, *, git_head: Callable[[Path], str]) -> dict[str, object]:
    if inputs.subject not in SUBJECTS:
        raise ValueError(f"r3 subject must be one of {', '.join(SUBJECTS)}, got {inputs.subject}")
    _require_directory(inputs.source_prepared_root, "source full prepared root")
    _require_directory(inputs.historical_worktree, "r3 historical worktree")
    if git_head(inputs.historical_worktree) != DAD9698_COMMIT:
        raise ValueError(f"r3 historical worktree must be pinned to dad9698b ({DAD9698_COMMIT})")
    for label, path, expected in (
        ("B6 model", inputs.b6_model, inputs.expected_b6_model_sha256),
        ("signal simulator", inputs.signal_simulator, inputs.expected_signal_simulator_sha256),
    ):
        _require_file(path, label)
        if _sha256(path) != expected:
            raise ValueError(f"{label} SHA256 disagrees with the explicit admission pin")
    _require_file(inputs.nesvor_executable, "NeSVoR executable")

    observations: dict[str, dict[str, object]] = {}
    manifests: dict[str, str | None] = {}
    for stack, expected_groups in EXPECTED_GROUP_COUNTS.items():
        stack_root = inputs.source_prepared_root / stack
        observation = _audit_observations(stack_root / "observations.npz")
        if len(observation["groups"]) != expected_groups:
            raise ValueError(f"{stack}: expected {expected_groups} groups, got {len(observation['groups'])}")
        observations[stack] = observation
        manifest = stack_root / "manifest.json"
        manifests[stack] = str(manifest.resolve()) if manifest.is_file() else None

    root = inputs.analysis_root.resolve()
    planned_destinations = [root / "prepared" / inputs.subject, root / "native_cache" / inputs.subject]
    for case in CASES:
        planned_destinations.extend((
            root / "sparse_inputs" / inputs.subject / case,
            root / "reconstruction" / "proposed" / inputs.subject / case,
            root / "reconstruction" / "fit_first" / inputs.subject / case,
            root / "endpoints" / "central" / "proposed" / inputs.subject / case,
            root / "endpoints" / "central" / "fit_first" / inputs.subject / case,
        ))
    existing = next((path for path in planned_destinations if path.exists()), None)
    if existing is not None:
        raise FileExistsError(f"refusing non-empty r3 destination: {existing}")
    return {
        "observations": observations,
        "manifests": manifests,
        "source_manifest_absent": any(path is None for path in manifests.values()),
    }


def build_r3_plan(inputs: R3Inputs, *, git_head: Callable[[Path], str] = _git_head) -> dict[str, object]:
    """Validate r3 inputs and return the complete per-subject execution DAG."""
    source_audit = _validate(inputs, git_head=git_head)
    root = inputs.analysis_root.resolve()
    prepared_root = root / "prepared"
    prepared_subject = prepared_root / inputs.subject
    native_cache = root / "native_cache" / inputs.subject
    migration_record = root / "migrations" / "r3_generation" / f"{inputs.subject}.json"
    steps: list[dict[str, object]] = [
        {
            "id": "copy_prepared",
            "kind": "copy_prepared",
            "source": str(inputs.source_prepared_root.resolve()),
            "destination": str(prepared_subject),
            "provenance_output": str(migration_record),
        }
    ]
    l100_fit_root = root / "reconstruction" / "fit_first" / inputs.subject / "L100_fold0"
    for case in CASES:
        condition, phase = CASE_PARAMETERS[case]
        sparse_root = root / "sparse_inputs"
        sparse_case = sparse_root / inputs.subject / case
        sparse_manifest = sparse_case / "sparse_manifest.json"
        proposed_root = root / "reconstruction" / "proposed" / inputs.subject / case
        fit_root = root / "reconstruction" / "fit_first" / inputs.subject / case
        proposed_endpoint = root / "endpoints" / "central" / "proposed" / inputs.subject / case
        fit_endpoint = root / "endpoints" / "central" / "fit_first" / inputs.subject / case
        steps.append({"id": f"build_sparse_{case}", "cwd": str(inputs.historical_worktree.resolve()), "argv": _command(
            inputs.python_executable, "-m", "paper_experiments.joint_vs_2dfit.phase2_clean.sparse",
            "--prepared-root", prepared_root, "--subject", inputs.subject,
            "--condition", condition, "--phase", str(phase), "--output-root", sparse_root,
        )})
        steps.append({"id": f"proposed_{case}", "cwd": str(inputs.historical_worktree.resolve()), "argv": _command(
            inputs.python_executable, "-m", "paper_experiments.joint_vs_2dfit.phase2_clean.proposed",
            "--prepared-root", sparse_case, "--decoder", inputs.signal_simulator,
            "--b6-model", inputs.b6_model, "--output", proposed_root,
            "--subject-id", inputs.subject, "--device", inputs.device, "--execute",
        )})
        baseline = _command(
            inputs.python_executable, "-m", "paper_experiments.joint_vs_2dfit.phase2_clean.baseline",
            "--prepared-root", sparse_case, "--subject-id", inputs.subject, "--output", fit_root,
            "--nesvor-executable", inputs.nesvor_executable,
        )
        if case != "L100_fold0":
            baseline.extend(("--reuse-native-maps-root", str(native_cache), "--sparse-manifest", str(sparse_manifest)))
        baseline.append("--execute")
        steps.append({"id": f"fit_first_{case}", "cwd": str(inputs.historical_worktree.resolve()), "argv": baseline})
        if case == "L100_fold0":
            steps.append({
                "id": "publish_native_cache",
                "kind": "copy_native_cache",
                "source": str((fit_root / "native_maps").resolve()),
                "destination": str(native_cache),
            })
        steps.append({"id": f"proposed_central_{case}", "cwd": str(inputs.historical_worktree.resolve()), "argv": _command(
            inputs.python_executable, "-m", "paper_experiments.joint_vs_2dfit.phase2_clean.matched_output", "proposed",
            "--subject", inputs.subject, "--condition", case, "--source-root", proposed_root,
            "--endpoint", "central", "--output", proposed_endpoint,
        )})
        steps.append({"id": f"fit_first_central_{case}", "cwd": str(inputs.historical_worktree.resolve()), "argv": _command(
            inputs.python_executable, "-m", "paper_experiments.joint_vs_2dfit.phase2_clean.matched_output", "fit-first",
            "--subject", inputs.subject, "--condition", case,
            "--l100-fit-first-root", l100_fit_root, "--sparse-fit-first-root", fit_root,
            "--sparse-manifest", sparse_manifest, "--r0-native-maps-root", native_cache,
            "--nesvor-executable", inputs.nesvor_executable, "--endpoint", "central",
            "--output", fit_endpoint, "--device", _sampling_device(inputs.device),
        )})
    return {
        "schema": PLAN_SCHEMA,
        "identity": {"subject": inputs.subject, "cases": list(CASES)},
        "historical_commit": DAD9698_COMMIT,
        "analysis_root": str(root),
        "source": {
            "prepared_root": str(inputs.source_prepared_root.resolve()),
            **source_audit,
        },
        "provenance": {
            "b6_model": {"path": str(inputs.b6_model.resolve()), "sha256": _sha256(inputs.b6_model)},
            "signal_simulator": {"path": str(inputs.signal_simulator.resolve()), "sha256": _sha256(inputs.signal_simulator)},
            "nesvor_executable": str(inputs.nesvor_executable.resolve()),
            "native_cache": str(native_cache),
            "l100_fit_first_root": str(l100_fit_root),
        },
        "publication": {"endpoint": "central", "metrics": "not_generated_by_r3"},
        "steps": steps,
    }


def _parse_args(argv: Iterable[str] | None = None) -> R3Inputs:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-prepared-root", "analysis-root", "historical-worktree", "b6-model", "signal-simulator", "nesvor-executable"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--subject", required=True, choices=SUBJECTS)
    parser.add_argument("--expected-b6-model-sha256", required=True)
    parser.add_argument("--expected-signal-simulator-sha256", required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    args = vars(parser.parse_args(argv))
    return R3Inputs(**{key: Path(value) if key in {
        "source_prepared_root", "analysis_root", "historical_worktree", "b6_model", "signal_simulator", "nesvor_executable"
    } else value for key, value in args.items()})


def main(argv: Iterable[str] | None = None) -> None:
    print(json.dumps(build_r3_plan(_parse_args(argv)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
