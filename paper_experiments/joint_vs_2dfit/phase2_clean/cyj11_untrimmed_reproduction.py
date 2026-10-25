"""Build, but never run, CYJ historical or explicit full-input validation commands.

This module intentionally keeps historical stages in caller-provided pinned
worktrees.  It does not import current Final5 route builders, so a local
planning run cannot silently acquire newer reconstruction behaviour.
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


PLAN_SCHEMA = "cyj11_untrimmed_reproduction_plan/v1"
SPARSE_COMMIT = "6f4142cccc17782ba6e985e74531822560b369f3"
Q002_COMMIT = "de41aef1c52b12037efed20f71cdf0db4696d49c"
FIT_FIRST_COMMIT = "00d88b45c75f272f05538997e38fbda85988e7aa"
HISTORICAL_GROUP_COUNTS = {"sax": 14, "2ch": 15, "4ch": 12}
STACKS = ("sax", "2ch", "4ch")


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    profile_name: str | None = None
    fold: int | None = None
    legacy_condition: str | None = None
    legacy_phase: int | None = None


CASE_SPECS = {
    "L100_fold0": CaseSpec("L100_fold0", legacy_condition="L100", legacy_phase=0),
    "L50_fold0": CaseSpec("L50_fold0", legacy_condition="L50", legacy_phase=0),
    "L50_fold1": CaseSpec("L50_fold1", legacy_condition="L50", legacy_phase=1),
    "L25_fold0": CaseSpec("L25_fold0", legacy_condition="L25", legacy_phase=0),
    "L25_fold1": CaseSpec("L25_fold1", legacy_condition="L25", legacy_phase=1),
    "L25_fold2": CaseSpec("L25_fold2", legacy_condition="L25", legacy_phase=2),
    "L25_fold3": CaseSpec("L25_fold3", legacy_condition="L25", legacy_phase=3),
    "SAX100_2CH0_4CH0": CaseSpec("SAX100_2CH0_4CH0", profile_name="CYJ_SAX100_2CH0_4CH0"),
    "SAX100_2CH0_4CH100": CaseSpec("SAX100_2CH0_4CH100", profile_name="CYJ_SAX100_2CH0_4CH100"),
    "SAX100_2CH100_4CH0": CaseSpec("SAX100_2CH100_4CH0", profile_name="CYJ_SAX100_2CH100_4CH0"),
    "S2CH50_4CH100_fold0": CaseSpec("S2CH50_4CH100_fold0", profile_name="CYJ_S2CH50_4CH100", fold=0),
    "S2CH50_4CH100_fold1": CaseSpec("S2CH50_4CH100_fold1", profile_name="CYJ_S2CH50_4CH100", fold=1),
    "S2CH100_4CH50_fold0": CaseSpec("S2CH100_4CH50_fold0", profile_name="CYJ_S2CH100_4CH50", fold=0),
    "S2CH100_4CH50_fold1": CaseSpec("S2CH100_4CH50_fold1", profile_name="CYJ_S2CH100_4CH50", fold=1),
}


@dataclass(frozen=True)
class ReproductionInputs:
    subject: str
    case_id: str
    prepared_root: Path
    output_root: Path
    sparse_worktree: Path
    q002_worktree: Path
    fit_first_worktree: Path
    b6_model: Path
    signal_simulator: Path
    protocol: Path
    native_cache_root: Path
    native_mask_root: Path | None
    l100_fit_first_root: Path
    expected_b6_model_sha256: str
    expected_signal_simulator_sha256: str
    expected_protocol_sha256: str
    nesvor_executable: Path | None = None
    python_executable: str = sys.executable
    device: str = "cuda:0"
    include_metric: bool = True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"cannot determine git HEAD for historical worktree: {path}") from exc


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _observation_groups(path: Path) -> int:
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
    return int(len(identities))


def _validate_inputs(inputs: ReproductionInputs, *, git_head: Callable[[Path], str]) -> tuple[CaseSpec, dict[str, dict[str, str]], dict[str, int]]:
    try:
        case = CASE_SPECS[inputs.case_id]
    except KeyError as exc:
        raise ValueError(f"unknown historical CYJ case: {inputs.case_id}") from exc
    _require_directory(inputs.prepared_root, "untrimmed prepared root")
    observation_records: dict[str, dict[str, str]] = {}
    counts: dict[str, int] = {}
    for stack in STACKS:
        path = inputs.prepared_root / inputs.subject / stack / "observations.npz"
        counts[stack] = _observation_groups(path)
        observation_records[stack] = {"path": str(path.resolve()), "sha256": _sha256(path)}
    if inputs.subject == "CYJ" and counts != HISTORICAL_GROUP_COUNTS:
        raise ValueError(
            "strict CYJ 1.1 reproduction requires 14 SAX, 15 2CH, and 12 4CH "
            f"groups from untrimmed input; got {counts}"
        )
    for label, path, expected in (
        ("sparse", inputs.sparse_worktree, SPARSE_COMMIT),
        ("q002", inputs.q002_worktree, Q002_COMMIT),
        ("fit_first", inputs.fit_first_worktree, FIT_FIRST_COMMIT),
    ):
        _require_directory(path, f"{label} worktree")
        actual = git_head(path)
        if actual != expected:
            raise ValueError(f"{label} worktree is at {actual}, expected historical commit {expected}")
    if inputs.output_root.exists() and (not inputs.output_root.is_dir() or any(inputs.output_root.iterdir())):
        raise FileExistsError(f"refusing non-empty output root: {inputs.output_root}")
    for label, path in (
        ("B6 model", inputs.b6_model),
        ("signal simulator", inputs.signal_simulator),
        ("protocol", inputs.protocol),
        ("R0 native cache", inputs.native_cache_root),
        ("L100 Fit-first root", inputs.l100_fit_first_root),
    ):
        (_require_file if label in {"B6 model", "signal simulator", "protocol"} else _require_directory)(path, label)
    if inputs.include_metric:
        if inputs.native_mask_root is None:
            raise ValueError("native mask root is required unless --skip-metric is set")
        _require_directory(inputs.native_mask_root, "native mask")
        _require_file(inputs.native_mask_root / "manifest.json", "native mask manifest")
    if _sha256(inputs.b6_model) != inputs.expected_b6_model_sha256:
        raise ValueError("B6 model SHA256 disagrees with the explicit admission pin")
    if _sha256(inputs.signal_simulator) != inputs.expected_signal_simulator_sha256:
        raise ValueError("signal simulator SHA256 disagrees with the explicit admission pin")
    if _sha256(inputs.protocol) != inputs.expected_protocol_sha256:
        raise ValueError("protocol SHA256 disagrees with the explicit admission pin")
    if inputs.nesvor_executable is not None:
        _require_file(inputs.nesvor_executable, "NeSVoR executable")
    return case, observation_records, counts


def _command(*items: str | Path) -> list[str]:
    return [str(item) for item in items]


def _fit_first_sampling_device(device: str) -> str:
    """Adapt the planner's Q002-style CUDA spelling to historical sampling CLI."""
    value = str(device).strip()
    if value == "-1" or value.isdecimal():
        return value
    if value.startswith("cuda:") and value.removeprefix("cuda:").isdecimal():
        return value.removeprefix("cuda:")
    raise ValueError("device must be cuda:N, a CUDA index N, or -1")


def _sparse_command(inputs: ReproductionInputs, case: CaseSpec, sparse_root: Path) -> list[str]:
    values: list[str | Path] = [
        inputs.python_executable, "-m", "paper_experiments.joint_vs_2dfit.phase2_clean.sparse",
        "--prepared-root", inputs.prepared_root, "--subject", inputs.subject, "--output-root", sparse_root,
    ]
    if case.profile_name is not None:
        values.extend(("--profile", inputs.sparse_worktree / "paper_experiments/joint_vs_2dfit/configs/sparse_profiles" / f"{case.profile_name}.json"))
        if case.fold is not None:
            values.extend(("--fold", str(case.fold)))
    else:
        values.extend(("--condition", str(case.legacy_condition), "--phase", str(case.legacy_phase)))
    return _command(*values)


def build_reproduction_plan(inputs: ReproductionInputs, *, git_head: Callable[[Path], str] = _git_head) -> dict[str, object]:
    """Validate static provenance and return the historical execution plan.

    The sparse manifest digest is a runtime binding because the immutable
    manifest is intentionally created by the first execution step.  The shell
    wrapper resolves that digest immediately after creation before it starts
    either reconstruction route.
    """
    case, observations, group_counts = _validate_inputs(inputs, git_head=git_head)
    root = inputs.output_root.resolve()
    sparse_root = root / "sparse_inputs"
    sparse_case_root = sparse_root / inputs.subject / inputs.case_id
    sparse_manifest = sparse_case_root / "sparse_manifest.json"
    proposed_root = root / "proposed" / inputs.subject / inputs.case_id
    fit_root = root / "fit_first" / inputs.subject / inputs.case_id
    proposed_endpoint = root / "formal_endpoints" / "Q002S640" / inputs.subject / inputs.case_id / "central"
    fit_endpoint = root / "formal_endpoints" / "Fit-first" / inputs.subject / inputs.case_id / "central"
    metric_root = root / "metrics" / "Q002S640" / inputs.case_id / "central"
    manifest_sha = "{sparse_manifest_sha256}"
    nesvor = inputs.nesvor_executable or Path("nesvor")
    sample_masks = {parameter: str((fit_root / "nesvor_inputs" / "sax" / parameter / "stack.nii.gz").resolve()) for parameter in ("t1", "t2")}
    steps: list[dict[str, object]] = [
        {"id": "build_sparse_input", "cwd": str(inputs.sparse_worktree.resolve()), "argv": _sparse_command(inputs, case, sparse_root)},
        {"id": "l100_fit_first_dependency", "kind": "verified_existing_dependency", "path": str(inputs.l100_fit_first_root.resolve())},
        {"id": "proposed_sparse", "cwd": str(inputs.q002_worktree.resolve()), "argv": _command(
            inputs.python_executable, "-m", "quality_experiments.Q002_B6_roi_same_anchor_mse.run_q002_reconstruction",
            "--cropped-prepared-root", sparse_case_root, "--subject-id", inputs.subject,
            "--sparse-manifest", sparse_manifest, "--expected-sparse-generator-git-sha", SPARSE_COMMIT,
            "--expected-sparse-manifest-sha256", manifest_sha, "--b6-model", inputs.b6_model,
            "--expected-b6-model-sha256", inputs.expected_b6_model_sha256, "--signal-simulator", inputs.signal_simulator,
            "--protocol", inputs.protocol, "--expected-protocol-sha256", inputs.expected_protocol_sha256,
            "--output", proposed_root, "--device", inputs.device,
            "--experiment-id", "Q002S640_B6_roi_same_anchor_mse", "--anchor-batch-size", "640",
        )},
        {"id": "fit_first_sparse_prepare", "cwd": str(inputs.fit_first_worktree.resolve()), "argv": _command(
            inputs.python_executable, "-m", "paper_experiments.joint_vs_2dfit.phase2_clean.baseline",
            "--prepared-root", sparse_case_root, "--subject-id", inputs.subject, "--output", fit_root,
            "--nesvor-executable", nesvor, "--reuse-native-maps-root", inputs.native_cache_root,
            "--sparse-manifest", sparse_manifest, "--expected-generator-git-sha", SPARSE_COMMIT,
            "--expected-manifest-sha256", manifest_sha, "--execute",
        ), "expected_output_sampling_masks": sample_masks,
           "output_sampling_semantics": "post_training_raw_volume_sampling_only"},
        {"id": "proposed_central", "cwd": str(inputs.fit_first_worktree.resolve()), "argv": _command(
            inputs.python_executable, "-m", "paper_experiments.joint_vs_2dfit.phase2_clean.matched_output", "proposed",
            "--subject", inputs.subject, "--condition", inputs.case_id, "--source-root", proposed_root,
            "--endpoint", "central", "--output", proposed_endpoint,
        )},
        {"id": "fit_first_central", "cwd": str(inputs.fit_first_worktree.resolve()), "argv": _command(
            inputs.python_executable, "-m", "paper_experiments.joint_vs_2dfit.phase2_clean.matched_output", "fit-first",
            "--subject", inputs.subject, "--condition", inputs.case_id, "--l100-fit-first-root", inputs.l100_fit_first_root,
            "--sparse-fit-first-root", fit_root, "--sparse-manifest", sparse_manifest,
            "--r0-native-maps-root", inputs.native_cache_root, "--nesvor-executable", nesvor,
            "--endpoint", "central", "--output", fit_endpoint, "--device", _fit_first_sampling_device(inputs.device),
        )},
    ]
    if inputs.include_metric:
        assert inputs.native_mask_root is not None
        steps.append({"id": "metric_central", "cwd": str(inputs.fit_first_worktree.resolve()), "argv": _command(
            inputs.python_executable, "-m", "paper_experiments.joint_vs_2dfit.phase2_clean.final_stratified_native_metrics",
            "--proposed-endpoint-root", proposed_endpoint, "--fit-first-endpoint-root", fit_endpoint,
            "--native-mask-root", inputs.native_mask_root, "--r0-native-maps-root", inputs.native_cache_root,
            "--sparse-manifest", sparse_manifest, "--output", metric_root,
        )})
    source = {
        "prepared_root": str(inputs.prepared_root.resolve()),
        "observations": observations,
        "observed_group_counts": group_counts,
        "layout_policy": "cyj_historical_untrimmed_14_15_12" if inputs.subject == "CYJ" else "subject_full_input_audited",
        "required_group_counts": HISTORICAL_GROUP_COUNTS if inputs.subject == "CYJ" else None,
    }
    provenance: dict[str, object] = {
        "b6_model": {"path": str(inputs.b6_model.resolve()), "sha256": _sha256(inputs.b6_model)},
        "signal_simulator": {"path": str(inputs.signal_simulator.resolve()), "sha256": _sha256(inputs.signal_simulator)},
        "protocol": {"path": str(inputs.protocol.resolve()), "sha256": _sha256(inputs.protocol)},
        "l100_fit_first_root": str(inputs.l100_fit_first_root.resolve()),
    }
    if inputs.include_metric:
        assert inputs.native_mask_root is not None
        provenance["native_mask_manifest"] = {"path": str((inputs.native_mask_root / "manifest.json").resolve()), "sha256": _sha256(inputs.native_mask_root / "manifest.json")}
    return {
        "schema": PLAN_SCHEMA,
        "identity": {"subject": inputs.subject, "case_id": inputs.case_id},
        "historical_stage_commits": {"sparse": SPARSE_COMMIT, "q002": Q002_COMMIT, "fit_first_endpoint_metric": FIT_FIRST_COMMIT},
        "source": source,
        "output_root": str(root),
        "runtime_bindings": {"sparse_manifest": str(sparse_manifest), "sparse_manifest_sha256": manifest_sha},
        "provenance": provenance,
        "publication": {"endpoint": "central", "q002_raw_route": "historical_default_including_psf128", "published_endpoints": ["central"], "metric": "included" if inputs.include_metric else "skipped_pending_mask_schema_audit"},
        "steps": steps,
    }


def _parse_args(argv: Iterable[str] | None = None) -> ReproductionInputs:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-root", "output-root", "sparse-worktree", "q002-worktree", "fit-first-worktree", "b6-model", "signal-simulator", "protocol", "native-cache-root", "l100-fit-first-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--native-mask-root")
    parser.add_argument("--subject", required=True); parser.add_argument("--case", dest="case_id", required=True, choices=tuple(CASE_SPECS))
    parser.add_argument("--expected-b6-model-sha256", required=True); parser.add_argument("--expected-signal-simulator-sha256", required=True); parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--nesvor-executable"); parser.add_argument("--python-executable", default=sys.executable); parser.add_argument("--device", default="cuda:0"); parser.add_argument("--skip-metric", action="store_true")
    args = parser.parse_args(argv)
    values = vars(args)
    values["include_metric"] = not values.pop("skip_metric")
    return ReproductionInputs(**{key: Path(value) if key in {
        "prepared_root", "output_root", "sparse_worktree", "q002_worktree", "fit_first_worktree", "b6_model", "signal_simulator", "protocol", "native_cache_root", "native_mask_root", "l100_fit_first_root", "nesvor_executable"
    } and value is not None else value for key, value in values.items()})


def main(argv: Iterable[str] | None = None) -> None:
    inputs = _parse_args(argv)
    print(json.dumps(build_reproduction_plan(inputs), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
