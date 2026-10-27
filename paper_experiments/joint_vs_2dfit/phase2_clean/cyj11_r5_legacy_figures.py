"""Plan and publish read-only r5 legacy-density figures from r3/r4 artifacts.

R5 never reconstructs or recalculates r4 metrics.  It restores the two
historical L100 native prediction bundles needed by the frozen CYJ1.1 figure
renderer, then renders L100/L50/L25 from explicit unified-root inputs.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .cyj11_r4_reporting import (
    DIRECT_NATIVE_AUDIT_COMPAT_COMMIT,
    FIXED_00D88_COMMIT,
    R4_SOURCE,
    SUBJECTS,
)


PLAN_SCHEMA = "cyj11_r5_legacy_figures_plan/v1"
R5_SOURCE = "r5_00d88b45c75f"
FIXED_FIGURE_COMMIT = FIXED_00D88_COMMIT
DIRECT_NATIVE_COMPAT_COMMIT = DIRECT_NATIVE_AUDIT_COMPAT_COMMIT
CASES = ("L100_fold0", "L50_fold0", "L25_fold0")


@dataclass(frozen=True)
class R5Inputs:
    subject: str
    analysis_root: Path
    historical_worktree: Path
    direct_native_worktree: Path
    python_executable: str = sys.executable
    device: str = "cuda:0"
    prepared_root: Path | None = None
    r0_native_maps_root: Path | None = None
    r4_native_mask_root: Path | None = None
    l100_proposed_checkpoint: Path | None = None
    cases: tuple[str, ...] = CASES


def _git_head(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError as error:
        raise ValueError(f"cannot determine git HEAD: {path}") from error


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _command(*items: str | Path) -> list[str]:
    return [str(item) for item in items]


def _selected_cases(inputs: R5Inputs) -> tuple[str, ...]:
    cases = tuple(inputs.cases)
    if not cases or len(cases) != len(set(cases)) or any(case not in CASES for case in cases):
        raise ValueError("r5 cases must be a unique non-empty subset of L100_fold0/L50_fold0/L25_fold0")
    if "L100_fold0" not in cases:
        raise ValueError("r5 always requires L100_fold0 to build the historical shared native support")
    return cases


def _case_paths(root: Path, subject: str, case: str) -> dict[str, Path]:
    endpoint = root / "endpoints" / "central"
    return {
        "sparse": root / "sparse_inputs" / subject / case / "sparse_manifest.json",
        "proposed": endpoint / "proposed" / subject / case,
        "fit_first": endpoint / "fit_first" / subject / case,
    }


def _figure_config(*, root: Path, subject: str, case: str, r0_native_maps_root: Path, native_mask_root: Path,
                   proposed_bundle: Path, fit_first_bundle: Path) -> dict[str, Any]:
    paths = _case_paths(root, subject, case)
    methods: dict[str, dict[str, str]] = {
        "Q002S640": {
            "t1": str((paths["proposed"] / "T1_3D.nii.gz").resolve()),
            "t2": str((paths["proposed"] / "T2_3D.nii.gz").resolve()),
        },
        "Fit-first": {
            "t1": str((paths["fit_first"] / "t1" / "volume_ms.nii.gz").resolve()),
            "t2": str((paths["fit_first"] / "t2" / "volume_ms.nii.gz").resolve()),
        },
    }
    if case == "L100_fold0":
        methods["Q002S640"]["native_bundle"] = str(proposed_bundle.resolve())
        methods["Fit-first"]["fit_first_native_bundle"] = str(fit_first_bundle.resolve())
    return {
        "schema": "cyj11_r5_00d88_legacy_figure_config/v1",
        "subject": subject,
        "condition": case,
        "endpoint": "central",
        "methods": methods,
        "l100_masks": {
            "t1": str((root / "endpoints" / "central" / "fit_first" / subject / "L100_fold0" / "t1" / "l100_checkpoint_mask.nii.gz").resolve()),
            "t2": str((root / "endpoints" / "central" / "fit_first" / subject / "L100_fold0" / "t2" / "l100_checkpoint_mask.nii.gz").resolve()),
        },
        "longitudinal": {"axis": 1, "index": "support_median", "common_support_contract": "frozen_L100_display_mask AND in_volume(all configured comparison methods) AND finite(all configured comparison methods)"},
        "scmr_longitudinal": {
            "sparse_manifest": str(paths["sparse"].resolve()),
            "r0_native_maps_root": str(r0_native_maps_root.resolve()),
            "reconstruction_spacing_mm": 1.0,
            "common_support_contract": "source comparison domain AND in_volume(all configured comparison methods) AND finite(all configured comparison methods)",
        },
        "native_mask_root": str(native_mask_root.resolve()),
        "native": {"mode": "all_source_groups", "stacks": ["sax", "2ch", "4ch"], "sparse_manifest": str(paths["sparse"].resolve()), "r0_native_maps_root": str(r0_native_maps_root.resolve())},
    }


def build_r5_plan(inputs: R5Inputs, *, git_head: Callable[[Path], str] = _git_head) -> dict[str, Any]:
    """Return a no-overwrite r5 plan after validating all source artifacts."""
    if inputs.subject not in SUBJECTS:
        raise ValueError(f"r5 subject must be one of {', '.join(SUBJECTS)}")
    cases = _selected_cases(inputs)
    root = inputs.analysis_root.resolve()
    fixed, direct = inputs.historical_worktree.resolve(), inputs.direct_native_worktree.resolve()
    _require_directory(fixed, "r5 fixed figure worktree")
    _require_directory(direct, "r5 direct-native compatibility worktree")
    if git_head(fixed) != FIXED_FIGURE_COMMIT:
        raise ValueError(f"r5 fixed figure worktree must be pinned to {FIXED_FIGURE_COMMIT}")
    if git_head(direct) != DIRECT_NATIVE_COMPAT_COMMIT:
        raise ValueError(f"r5 direct-native worktree must be pinned to {DIRECT_NATIVE_COMPAT_COMMIT}")
    prepared = (inputs.prepared_root or root / "prepared" / inputs.subject).resolve()
    r0_native = (inputs.r0_native_maps_root or root / "native_cache" / inputs.subject).resolve()
    native_masks = (inputs.r4_native_mask_root or root / "metrics" / R4_SOURCE / "native_support" / inputs.subject / "legacy-density" / "L100_fold0" / "current_native_masks").resolve()
    checkpoint = (inputs.l100_proposed_checkpoint or root / "reconstruction" / "proposed" / inputs.subject / "L100_fold0" / "model.pt").resolve()
    _require_directory(prepared, "r5 prepared root")
    _require_file(checkpoint, "r5 L100 Proposed checkpoint")
    _require_file(native_masks / "manifest.json", "r4 legacy native-mask manifest")
    for stack in ("sax", "2ch", "4ch"):
        _require_file(prepared / stack / "observations.npz", f"r5 prepared {stack} observation")
        for name in ("native_maps.npz", "geometry.npz", "manifest.json"):
            _require_file(r0_native / stack / name, f"r5 R0 native cache {stack}/{name}")
    baseline = root / "reconstruction" / "fit_first" / inputs.subject / "L100_fold0"
    _require_directory(baseline, "r5 L100 Fit-first baseline root")
    destinations: list[Path] = []
    support = root / "metrics" / R5_SOURCE / "native_bundles" / inputs.subject / "legacy-density" / "L100_fold0"
    proposed_bundle = support / "proposed_direct_native"
    pose_audit = support / "fit_first_native_pose_audit"
    fit_first_bundle = support / "fit_first_native"
    destinations.extend((proposed_bundle, pose_audit, fit_first_bundle))
    figure_root = root / "visualizations" / R5_SOURCE / inputs.subject / "legacy-density"
    config_root = root / "configs" / "r5" / inputs.subject / "legacy-density" / R5_SOURCE
    for case in cases:
        paths = _case_paths(root, inputs.subject, case)
        _require_file(paths["sparse"], f"r5 {case} sparse manifest")
        for path, label in (
            (paths["proposed"] / "T1_3D.nii.gz", f"r5 {case} Proposed T1 endpoint"),
            (paths["proposed"] / "T2_3D.nii.gz", f"r5 {case} Proposed T2 endpoint"),
            (paths["proposed"] / "manifest.json", f"r5 {case} Proposed endpoint manifest"),
            (paths["fit_first"] / "t1" / "volume_ms.nii.gz", f"r5 {case} Fit-first T1 endpoint"),
            (paths["fit_first"] / "t2" / "volume_ms.nii.gz", f"r5 {case} Fit-first T2 endpoint"),
            (paths["fit_first"] / "manifest.json", f"r5 {case} Fit-first endpoint manifest"),
        ):
            _require_file(path, label)
        destinations.extend((config_root / f"{case}.central.figure.json", figure_root / case))
    for parameter in ("t1", "t2"):
        _require_file(root / "endpoints" / "central" / "fit_first" / inputs.subject / "L100_fold0" / parameter / "l100_checkpoint_mask.nii.gz", f"r5 L100 {parameter} checkpoint mask")
    migration = root / "migrations" / "r5_generation" / f"{inputs.subject}.legacy-density.{R5_SOURCE}.json"
    destinations.append(migration)
    existing = next((path for path in destinations if path.exists()), None)
    if existing is not None:
        raise FileExistsError(f"refusing existing r5 destination: {existing}")
    l100 = _case_paths(root, inputs.subject, "L100_fold0")
    steps: list[dict[str, Any]] = [
        {"id": "l100_proposed_direct_native_bundle", "kind": "command", "cwd": str(direct), "argv": _command(
            inputs.python_executable, "-m", "paper_experiments.joint_vs_2dfit.phase2_clean.direct_native_bundles",
            "--checkpoint", checkpoint, "--prepared-root", prepared, "--endpoint", "central", "--output", proposed_bundle, "--device", inputs.device,
        )},
        {"id": "l100_fit_first_pose_audit", "kind": "command", "cwd": str(fixed), "argv": _command(
            inputs.python_executable, "-m", "paper_experiments.joint_vs_2dfit.phase2_clean.fit_first_native_pose_audit",
            "--baseline-root", baseline, "--r0-native-maps-root", r0_native, "--output", pose_audit,
        )},
        {"id": "l100_fit_first_native_bundle", "kind": "command", "cwd": str(fixed), "argv": _command(
            inputs.python_executable, "-m", "paper_experiments.joint_vs_2dfit.phase2_clean.fit_first_native_bundles",
            "--audit-root", pose_audit, "--baseline-root", baseline, "--r0-native-maps-root", r0_native,
            "--endpoint", "central", "--endpoint-source-root", l100["fit_first"], "--output", fit_first_bundle,
        )},
    ]
    for case in cases:
        config = config_root / f"{case}.central.figure.json"
        steps.extend((
            {"id": f"figure_config_{case}", "kind": "write_json", "path": str(config), "payload": _figure_config(root=root, subject=inputs.subject, case=case, r0_native_maps_root=r0_native, native_mask_root=native_masks, proposed_bundle=proposed_bundle, fit_first_bundle=fit_first_bundle)},
            {"id": f"figures_{case}", "kind": "command", "cwd": str(fixed), "argv": _command(inputs.python_executable, "-m", "paper_experiments.joint_vs_2dfit.phase2_clean.final_cyj_reporting", "figures", "--config", config, "--output", figure_root / case)},
        ))
    steps.append({"id": "publish_migration_record", "kind": "write_json", "path": str(migration), "payload": {
        "schema": "cyj11_r5_legacy_figure_generation/v1", "status": "COMPLETE", "subject": inputs.subject,
        "protocol": "legacy-density", "cases": list(cases), "r5_source": R5_SOURCE,
        "fixed_figure_commit": FIXED_FIGURE_COMMIT, "direct_native_compat_commit": DIRECT_NATIVE_COMPAT_COMMIT,
        "r4_native_mask_root": str(native_masks), "native_bundle_root": str(support), "figure_root": str(figure_root),
        "no_reconstruction": True, "no_r4_metric_modified": True,
    }})
    return {"schema": PLAN_SCHEMA, "identity": {"subject": inputs.subject, "protocol": "legacy-density", "cases": list(cases)}, "analysis_root": str(root), "r5_source": R5_SOURCE, "provenance": {"prepared_root": str(prepared), "r0_native_maps_root": str(r0_native), "r4_native_mask_root": str(native_masks), "l100_proposed_checkpoint": str(checkpoint), "l100_fit_first_baseline_root": str(baseline), "fixed_worktree": str(fixed), "direct_native_worktree": str(direct)}, "steps": steps}


def execute_r5_plan(plan_path: str | Path, commands_log: str | Path) -> None:
    plan_path, commands_log = Path(plan_path), Path(commands_log)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"unsupported r5 plan schema: {plan.get('schema')}")
    if commands_log.exists():
        raise FileExistsError(f"refusing existing r5 commands log: {commands_log}")
    commands_log.parent.mkdir(parents=True, exist_ok=True)
    with commands_log.open("x", encoding="utf-8") as log:
        for step in plan["steps"]:
            log.write(json.dumps({"id": step["id"], "kind": step["kind"]}, sort_keys=True) + "\n"); log.flush()
            if step["kind"] == "command":
                result = subprocess.run(step["argv"], cwd=step["cwd"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                log.write(result.stdout); log.flush()
                if result.returncode:
                    raise subprocess.CalledProcessError(result.returncode, step["argv"])
            elif step["kind"] == "write_json":
                path = Path(step["path"])
                if path.exists():
                    raise FileExistsError(f"refusing existing r5 config: {path}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(step["payload"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
            else:
                raise ValueError(f"unsupported r5 step kind: {step['kind']}")


def _parse_inputs(argv: Iterable[str]) -> R5Inputs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True, choices=SUBJECTS)
    parser.add_argument("--analysis-root", required=True, type=Path)
    parser.add_argument("--historical-worktree", required=True, type=Path)
    parser.add_argument("--direct-native-worktree", required=True, type=Path)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--r0-native-maps-root", type=Path)
    parser.add_argument("--r4-native-mask-root", type=Path)
    parser.add_argument("--l100-proposed-checkpoint", type=Path)
    parser.add_argument("--case", action="append")
    args = parser.parse_args(list(argv))
    return R5Inputs(subject=args.subject, analysis_root=args.analysis_root, historical_worktree=args.historical_worktree, direct_native_worktree=args.direct_native_worktree, python_executable=args.python_executable, device=args.device, prepared_root=args.prepared_root, r0_native_maps_root=args.r0_native_maps_root, r4_native_mask_root=args.r4_native_mask_root, l100_proposed_checkpoint=args.l100_proposed_checkpoint, cases=tuple(args.case or CASES))


def main(argv: Iterable[str] | None = None) -> None:
    args = list(argv if argv is not None else sys.argv[1:])
    if args and args[0] == "execute-r5-plan":
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--plan", required=True, type=Path); parser.add_argument("--commands-log", required=True, type=Path)
        parsed = parser.parse_args(args[1:]); execute_r5_plan(parsed.plan, parsed.commands_log); return
    print(json.dumps(build_r5_plan(_parse_inputs(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
