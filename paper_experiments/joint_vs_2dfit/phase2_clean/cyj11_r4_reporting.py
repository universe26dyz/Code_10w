"""Plan and execute r4 metrics/reporting with the immutable CYJ1.1 00d88 code.

This module deliberately owns only admission, paths, provenance and physical
publication.  Every scientific calculation and every figure is run as a
subprocess in the detached ``00d88b45...`` worktree.
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


PLAN_SCHEMA = "cyj11_r4_reporting_plan/v1"
FIXED_00D88_COMMIT = "00d88b45c75f272f05538997e38fbda85988e7aa"
R4_SOURCE = "r4_00d88b45c75f"
SUBJECTS = ("CYJ", "DYZ", "HJL", "HHZ", "DYL_2")
PROFILE_CASES = (
    "SAX100_2CH0_4CH0", "SAX100_2CH0_4CH100", "SAX100_2CH100_4CH0",
    "S2CH50_4CH100_fold0", "S2CH50_4CH100_fold1",
    "S2CH100_4CH50_fold0", "S2CH100_4CH50_fold1",
)
LEGACY_CASES = ("L100_fold0", "L50_fold0", "L25_fold0")
PROTOCOL_CASES = {"profile-v2": PROFILE_CASES, "legacy-density": LEGACY_CASES}


@dataclass(frozen=True)
class R4Inputs:
    subject: str
    protocol: str
    analysis_root: Path
    historical_worktree: Path
    python_executable: str = sys.executable
    device: str = "cuda:0"
    prepared_root: Path | None = None
    r0_native_maps_root: Path | None = None
    l100_proposed_checkpoint: Path | None = None
    cases: tuple[str, ...] = ()


def _git_head(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError as error:
        raise ValueError(f"cannot determine git HEAD for r4 fixed worktree: {path}") from error


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _command(*items: str | Path) -> list[str]:
    return [str(item) for item in items]


def _selected_cases(inputs: R4Inputs) -> tuple[str, ...]:
    if inputs.protocol not in PROTOCOL_CASES:
        raise ValueError("r4 protocol must be profile-v2 or legacy-density")
    cases = inputs.cases or PROTOCOL_CASES[inputs.protocol]
    if not cases or len(cases) != len(set(cases)) or any(case not in PROTOCOL_CASES[inputs.protocol] for case in cases):
        raise ValueError(f"r4 cases must be a unique non-empty subset of {inputs.protocol}")
    return tuple(cases)


def _paths(root: Path, subject: str, case: str) -> dict[str, Path]:
    endpoint = root / "endpoints" / "central"
    return {
        "sparse": root / "sparse_inputs" / subject / case / "sparse_manifest.json",
        "proposed": endpoint / "proposed" / subject / case,
        "fit_first": endpoint / "fit_first" / subject / case,
        "metric": root / "metrics" / "central" / subject / case / R4_SOURCE,
    }


def _load_sparse(path: Path, *, subject: str, protocol: str, case: str) -> dict[str, Any]:
    _require_file(path, f"{case} sparse manifest")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("subject") != subject or payload.get("status") != "ELIGIBLE":
        raise ValueError(f"{case}: sparse manifest must be ELIGIBLE for {subject}")
    if protocol == "profile-v2":
        if payload.get("schema_version") != "phase4r_sparse_profile_v2" or payload.get("case_id") != case:
            raise ValueError(f"{case}: expected explicit phase4r_sparse_profile_v2 identity")
        if not isinstance(payload.get("profile_id"), str) or not payload["profile_id"]:
            raise ValueError(f"{case}: profile-v2 manifest lacks profile_id")
    elif payload.get("condition") != case.split("_fold", 1)[0]:
        raise ValueError(f"{case}: legacy sparse condition disagrees with requested case")
    return payload


def _figure_config(*, root: Path, subject: str, case: str, sparse: Path, profile: dict[str, Any] | None, native_mask_root: Path, r0_native_maps_root: Path) -> dict[str, Any]:
    paths = _paths(root, subject, case)
    fit = paths["fit_first"]
    proposed = paths["proposed"]
    config: dict[str, Any] = {
        "schema": "cyj11_r4_00d88_figure_config/v1",
        "subject": subject,
        "condition": case,
        "endpoint": "central",
        "methods": {
            "Q002S640": {"t1": str((proposed / "T1_3D.nii.gz").resolve()), "t2": str((proposed / "T2_3D.nii.gz").resolve())},
            "Fit-first": {"t1": str((fit / "t1" / "volume_ms.nii.gz").resolve()), "t2": str((fit / "t2" / "volume_ms.nii.gz").resolve())},
        },
        "l100_masks": {
            "t1": str((root / "endpoints" / "central" / "fit_first" / subject / "L100_fold0" / "t1" / "l100_checkpoint_mask.nii.gz").resolve()),
            "t2": str((root / "endpoints" / "central" / "fit_first" / subject / "L100_fold0" / "t2" / "l100_checkpoint_mask.nii.gz").resolve()),
        },
        "longitudinal": {"axis": 1, "index": "support_median", "common_support_contract": "frozen_L100_display_mask AND in_volume(all configured comparison methods) AND finite(all configured comparison methods)"},
        "native_mask_root": str(native_mask_root.resolve()),
        "native": {"mode": "all_source_groups", "stacks": ["sax", "2ch", "4ch"], "sparse_manifest": str(sparse.resolve()), "r0_native_maps_root": str(r0_native_maps_root.resolve())},
    }
    if profile is not None:
        config.update({"profile_id": profile["profile_id"], "case_id": case})
    return config


def build_r4_plan(inputs: R4Inputs, *, git_head: Callable[[Path], str] = _git_head, render_figures: bool = False) -> dict[str, Any]:
    """Validate immutable r4 inputs and return a no-overwrite execution DAG."""
    if inputs.subject not in SUBJECTS:
        raise ValueError(f"r4 subject must be one of {', '.join(SUBJECTS)}")
    cases = _selected_cases(inputs)
    if render_figures and inputs.protocol == "legacy-density" and "L100_fold0" in cases:
        raise ValueError(
            "legacy L100 figures require a fixed final-pose Fit-first native bundle, which r3 does not publish; "
            "run legacy metrics without --render-figures, or render only L50/L25 explicitly"
        )
    root = inputs.analysis_root.resolve()
    _require_directory(inputs.historical_worktree, "r4 fixed historical worktree")
    if git_head(inputs.historical_worktree) != FIXED_00D88_COMMIT:
        raise ValueError(f"r4 fixed historical worktree must be pinned to 00d88b45 ({FIXED_00D88_COMMIT})")
    prepared = (inputs.prepared_root or root / "prepared" / inputs.subject).resolve()
    _require_directory(prepared, "r4 cropped prepared root")
    for stack in ("sax", "2ch", "4ch"):
        _require_file(prepared / stack / "observations.npz", f"r4 prepared {stack} observation")
    native_cache = (inputs.r0_native_maps_root or root / "native_cache" / inputs.subject).resolve()
    _require_directory(native_cache, "r4 R0 native cache")
    for stack in ("sax", "2ch", "4ch"):
        for name in ("native_maps.npz", "geometry.npz", "manifest.json"):
            _require_file(native_cache / stack / name, f"r4 R0 native cache {stack}/{name}")
    l100 = _paths(root, inputs.subject, "L100_fold0")
    l100_sparse = _load_sparse(l100["sparse"], subject=inputs.subject, protocol="legacy-density", case="L100_fold0")
    checkpoint = (inputs.l100_proposed_checkpoint or root / "reconstruction" / "proposed" / inputs.subject / "L100_fold0" / "model.pt").resolve()
    _require_file(checkpoint, "r4 L100 Proposed checkpoint")
    for path, label in (
        (l100["proposed"] / "T1_3D.nii.gz", "r4 L100 Proposed T1 endpoint"),
        (l100["proposed"] / "T2_3D.nii.gz", "r4 L100 Proposed T2 endpoint"),
        (l100["fit_first"] / "t1" / "l100_checkpoint_mask.nii.gz", "r4 frozen L100 T1 mask"),
        (l100["fit_first"] / "t2" / "l100_checkpoint_mask.nii.gz", "r4 frozen L100 T2 mask"),
    ):
        _require_file(path, label)
    sparse_records: dict[str, dict[str, Any]] = {}
    destinations: list[Path] = []
    native_support = root / "metrics" / R4_SOURCE / "native_support" / inputs.subject / "L100_fold0" / "direct_native_endpoint_audit"
    native_masks = root / "metrics" / R4_SOURCE / "native_support" / inputs.subject / "L100_fold0" / "current_native_masks"
    destinations.extend((native_support, native_masks))
    for case in cases:
        paths = _paths(root, inputs.subject, case)
        sparse_records[case] = _load_sparse(paths["sparse"], subject=inputs.subject, protocol=inputs.protocol, case=case)
        for path, label in (
            (paths["proposed"] / "T1_3D.nii.gz", f"{case} Proposed T1 endpoint"),
            (paths["proposed"] / "T2_3D.nii.gz", f"{case} Proposed T2 endpoint"),
            (paths["fit_first"] / "t1" / "volume_ms.nii.gz", f"{case} Fit-first T1 endpoint"),
            (paths["fit_first"] / "t2" / "volume_ms.nii.gz", f"{case} Fit-first T2 endpoint"),
        ):
            _require_file(path, label)
        destinations.append(paths["metric"])
    collection = root / "metrics" / R4_SOURCE / "report_collections" / inputs.subject / inputs.protocol
    summary = root / "metrics" / R4_SOURCE / "summaries" / inputs.subject / inputs.protocol
    destinations.extend((collection, summary))
    figure_root = root / "visualizations" / R4_SOURCE / inputs.subject / inputs.protocol
    if render_figures:
        destinations.extend(figure_root / case for case in cases)
        destinations.extend(root / "configs" / "r4" / inputs.subject / inputs.protocol / R4_SOURCE / f"{case}.central.figure.json" for case in cases)
    migration_record = root / "migrations" / "r4_generation" / f"{inputs.subject}.{inputs.protocol}.{R4_SOURCE}.json"
    destinations.append(migration_record)
    existing = next((path for path in destinations if path.exists()), None)
    if existing is not None:
        raise FileExistsError(f"refusing existing r4 destination: {existing}")

    fixed = inputs.historical_worktree.resolve()
    steps: list[dict[str, Any]] = [
        {"id": "direct_native_l100_audit", "kind": "command", "cwd": str(fixed), "argv": _command(
            inputs.python_executable, "-m", "paper_experiments.joint_vs_2dfit.phase2_clean.direct_native_endpoint_audit",
            "--checkpoint", checkpoint, "--prepared-root", prepared, "--r0-native-maps-root", native_cache,
            "--sparse-manifest", l100["sparse"], "--central-t1", l100["proposed"] / "T1_3D.nii.gz",
            "--central-t2", l100["proposed"] / "T2_3D.nii.gz", "--l100-t1", l100["fit_first"] / "t1" / "l100_checkpoint_mask.nii.gz",
            "--l100-t2", l100["fit_first"] / "t2" / "l100_checkpoint_mask.nii.gz", "--output", native_support, "--device", inputs.device,
        )},
        {"id": "current_native_masks", "kind": "command", "cwd": str(fixed), "argv": _command(
            inputs.python_executable, "-m", "paper_experiments.joint_vs_2dfit.phase2_clean.current_native_central_component_mask",
            "--r0-native-maps-root", native_cache, "--sparse-manifest", l100["sparse"],
            "--l100-native-support-root", native_support, "--output", native_masks,
        )},
    ]
    for case in cases:
        paths = _paths(root, inputs.subject, case)
        steps.append({"id": f"metrics_{case}", "kind": "command", "cwd": str(fixed), "argv": _command(
            inputs.python_executable, "-m", "paper_experiments.joint_vs_2dfit.phase2_clean.final_stratified_native_metrics",
            "--proposed-endpoint-root", paths["proposed"], "--fit-first-endpoint-root", paths["fit_first"],
            "--native-mask-root", native_masks, "--r0-native-maps-root", native_cache,
            "--sparse-manifest", paths["sparse"], "--output", paths["metric"],
        )})
        report_case = collection / "Q002S640" / inputs.subject / case / "central"
        steps.append({"id": f"copy_metric_{case}", "kind": "copytree", "source": str(paths["metric"]), "destination": str(report_case)})
        if render_figures:
            config_path = root / "configs" / "r4" / inputs.subject / inputs.protocol / R4_SOURCE / f"{case}.central.figure.json"
            steps.append({"id": f"figure_config_{case}", "kind": "write_json", "path": str(config_path), "payload": _figure_config(root=root, subject=inputs.subject, case=case, sparse=paths["sparse"], profile=sparse_records[case] if inputs.protocol == "profile-v2" else None, native_mask_root=native_masks, r0_native_maps_root=native_cache)})
            steps.append({"id": f"figures_{case}", "kind": "command", "cwd": str(fixed), "argv": _command(
                inputs.python_executable, "-m", "paper_experiments.joint_vs_2dfit.phase2_clean.final_cyj_reporting", "figures",
                "--config", config_path, "--output", figure_root / case,
            )})
    steps.append({"id": "aggregate", "kind": "command", "cwd": str(fixed), "argv": _command(
        inputs.python_executable, "-m", "paper_experiments.joint_vs_2dfit.phase2_clean.final_cyj_reporting", "aggregate",
        "--subject", inputs.subject, "--fit-first-source-method", "Q002S640", "--method-root", f"Q002S640={collection}", "--output", summary,
    )})
    steps.append({"id": "publish_migration_record", "kind": "write_json", "path": str(migration_record), "payload": {
        "schema": "cyj11_r4_generation/v1", "status": "COMPLETE", "subject": inputs.subject,
        "protocol": inputs.protocol, "cases": list(cases), "r4_source": R4_SOURCE,
        "fixed_metrics_visualization_commit": FIXED_00D88_COMMIT,
        "metrics_root": str((root / "metrics" / "central" / inputs.subject).resolve()),
        "report_collection": str(collection), "summary": str(summary), "figures_enabled": render_figures,
    }})
    return {
        "schema": PLAN_SCHEMA,
        "fixed_metrics_visualization_commit": FIXED_00D88_COMMIT,
        "identity": {"subject": inputs.subject, "protocol": inputs.protocol, "cases": list(cases)},
        "analysis_root": str(root),
        "r4_source": R4_SOURCE,
        "provenance": {"prepared_root": str(prepared), "r0_native_maps_root": str(native_cache.resolve()), "l100_proposed_checkpoint": str(checkpoint), "l100_sparse_manifest": str(l100["sparse"].resolve()), "fixed_worktree": str(fixed)},
        "publication": {"metrics": str(root / "metrics" / "central" / inputs.subject), "report_collection": str(collection), "summary": str(summary), "figures_enabled": render_figures},
        "steps": steps,
    }


def execute_r4_plan(plan_path: Path, commands_log: Path) -> None:
    """Execute a saved plan in order, with no shell interpolation or overwrite."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"unsupported r4 plan schema: {plan.get('schema')}")
    if commands_log.exists():
        raise FileExistsError(f"refusing existing r4 commands log: {commands_log}")
    commands_log.parent.mkdir(parents=True, exist_ok=True)
    with commands_log.open("x", encoding="utf-8") as log:
        for step in plan["steps"]:
            kind = step["kind"]
            log.write(json.dumps({"id": step["id"], "kind": kind}, sort_keys=True) + "\n")
            log.flush()
            if kind == "command":
                result = subprocess.run(step["argv"], cwd=step["cwd"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                log.write(result.stdout)
                log.flush()
                if result.returncode:
                    raise subprocess.CalledProcessError(result.returncode, step["argv"])
            elif kind == "copytree":
                source, destination = Path(step["source"]), Path(step["destination"])
                if destination.exists():
                    raise FileExistsError(f"refusing existing r4 copy destination: {destination}")
                shutil.copytree(source, destination)
            elif kind == "write_json":
                path = Path(step["path"])
                if path.exists():
                    raise FileExistsError(f"refusing existing r4 config: {path}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(step["payload"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
            else:
                raise ValueError(f"unsupported r4 step kind: {kind}")


def _metric_file(root: Path, case: str) -> Path:
    candidates = sorted(path for path in root.rglob("aggregate_metrics.json") if case in path.parts)
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one aggregate_metrics.json for {case} under {root}, found {len(candidates)}")
    return candidates[0]


def _metric_rows(payload: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Return explicit parameter/region/method rows without inventing a mapping."""
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    if isinstance(payload.get("regions"), dict):
        for parameter, regions in payload["regions"].items():
            if not isinstance(regions, dict):
                continue
            for region, values in regions.items():
                if not isinstance(values, dict):
                    continue
                for side in ("proposed", "fit_first"):
                    if isinstance(values.get(side), dict):
                        rows[(str(parameter), str(region), side)] = values[side]
    if isinstance(payload.get("primary_aggregate"), dict):
        for parameter, sides in payload["primary_aggregate"].items():
            if isinstance(sides, dict):
                for side in ("proposed", "fit_first"):
                    if isinstance(sides.get(side), dict):
                        rows[(str(parameter), "primary_aggregate", side)] = sides[side]
    return rows


def compare_metric_trees(*, historical_root: str | Path, r4_root: str | Path, cases: Iterable[str], output: str | Path) -> Path:
    """Write a diagnostic-only old-vs-r4 comparison; neither source is changed."""
    historical_root, r4_root, output = Path(historical_root), Path(r4_root), Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    result_cases: list[dict[str, Any]] = []
    for case in cases:
        historical_path, r4_path = _metric_file(historical_root, case), _metric_file(r4_root, case)
        historical = json.loads(historical_path.read_text(encoding="utf-8"))
        current = json.loads(r4_path.read_text(encoding="utf-8"))
        old_rows, new_rows = _metric_rows(historical), _metric_rows(current)
        shared = sorted(set(old_rows).intersection(new_rows))
        deltas = []
        for key in shared:
            previous, refreshed = old_rows[key], new_rows[key]
            values = {name: (float(refreshed[name]) - float(previous[name])) for name in ("rmse", "mae", "bias") if name in previous and name in refreshed}
            deltas.append({"parameter": key[0], "region": key[1], "method": key[2], "old": previous, "r4": refreshed, "r4_minus_old": values})
        result_cases.append({
            "case": case,
            "historical_metrics": str(historical_path.resolve()), "r4_metrics": str(r4_path.resolve()),
            "historical_schema": historical.get("schema", "legacy_primary"), "r4_schema": current.get("schema", "legacy_primary"),
            "shared_metric_keys": [{"parameter": key[0], "region": key[1], "method": key[2]} for key in shared],
            "only_historical_keys": [{"parameter": key[0], "region": key[1], "method": key[2]} for key in sorted(set(old_rows).difference(new_rows))],
            "only_r4_keys": [{"parameter": key[0], "region": key[1], "method": key[2]} for key in sorted(set(new_rows).difference(old_rows))],
            "numeric_deltas_for_exactly_shared_keys": deltas,
            "comparison_status": "DIAGNOSTIC_ONLY_SUPPORT_CONTRACT_MUST_BE_REVIEWED",
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "cyj11_r4_historical_metric_comparison/v1",
        "comparison_contract": "Read-only diagnostic comparison. Numeric deltas are emitted only for identical parameter/region/method keys. They are not claims of statistical equivalence unless the recorded support provenance is separately audited.",
        "historical_root": str(historical_root.resolve()), "r4_root": str(r4_root.resolve()), "cases": result_cases,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _parse_inputs(argv: Iterable[str] | None = None) -> tuple[R4Inputs, bool]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True, choices=SUBJECTS)
    parser.add_argument("--protocol", required=True, choices=tuple(PROTOCOL_CASES))
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--historical-worktree", required=True)
    parser.add_argument("--prepared-root")
    parser.add_argument("--r0-native-maps-root")
    parser.add_argument("--l100-proposed-checkpoint")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--render-figures", action="store_true")
    args = parser.parse_args(argv)
    return R4Inputs(subject=args.subject, protocol=args.protocol, analysis_root=Path(args.analysis_root), historical_worktree=Path(args.historical_worktree), prepared_root=Path(args.prepared_root) if args.prepared_root else None, r0_native_maps_root=Path(args.r0_native_maps_root) if args.r0_native_maps_root else None, l100_proposed_checkpoint=Path(args.l100_proposed_checkpoint) if args.l100_proposed_checkpoint else None, cases=tuple(args.case), python_executable=args.python_executable, device=args.device), args.render_figures


def main(argv: Iterable[str] | None = None) -> None:
    args = list(argv if argv is not None else sys.argv[1:])
    if args and args[0] == "execute-r4-plan":
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--plan", required=True, type=Path); parser.add_argument("--commands-log", required=True, type=Path)
        parsed = parser.parse_args(args[1:]); execute_r4_plan(parsed.plan, parsed.commands_log); return
    if args and args[0] == "compare-metric-trees":
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--historical-root", required=True, type=Path); parser.add_argument("--r4-root", required=True, type=Path)
        parser.add_argument("--case", action="append", required=True); parser.add_argument("--output", required=True, type=Path)
        parsed = parser.parse_args(args[1:]); print(compare_metric_trees(historical_root=parsed.historical_root, r4_root=parsed.r4_root, cases=parsed.case, output=parsed.output)); return
    inputs, render_figures = _parse_inputs(args)
    print(json.dumps(build_r4_plan(inputs, render_figures=render_figures), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
