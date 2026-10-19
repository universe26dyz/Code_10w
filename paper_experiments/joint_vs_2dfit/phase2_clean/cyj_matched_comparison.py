"""CYJ-only Q002S640/Fit-first matched-output aliases and result aggregation."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .contracts import sha256, write_immutable_json
from .matched_output import endpoint_contract, export_fixed_l100_fit_first, formalize_proposed_export


CYJ_CASES = (
    "L100_fold0",
    "L50_fold0", "L50_fold1",
    "L25_fold0", "L25_fold1", "L25_fold2", "L25_fold3",
)
EXPECTED_FOLDS = {"L100": (0,), "L50": (0, 1), "L25": (0, 1, 2, 3)}
STACK_LABELS = {"2ch": "2CH", "4ch": "4CH", "all_lax": "ALL_LAX"}
PARAMETERS = ("t1", "t2")
ENDPOINTS = ("central", "psf128")


def parse_cyj_case(condition: str) -> tuple[str, int]:
    if condition not in CYJ_CASES:
        raise ValueError(f"CYJ matched comparison only permits {CYJ_CASES}, got {condition}")
    level, fold = condition.split("_fold", 1)
    return level, int(fold)


def require_l100_r0_reuse(condition: str, l100_fit_first_root: str | Path, sparse_fit_first_root: str | Path) -> None:
    """L100 is an R0 identity case, never an R1 reconstruction input."""
    level, _ = parse_cyj_case(condition)
    if level == "L100" and Path(l100_fit_first_root).resolve() != Path(sparse_fit_first_root).resolve():
        raise ValueError("L100 must reuse the R0/L100 Fit-first root as the sparse checkpoint root")


def formal_case_layout(output_root: str | Path, condition: str, endpoint: str) -> Path:
    parse_cyj_case(condition)
    endpoint_contract(endpoint)
    return Path(output_root) / "CYJ" / condition / endpoint


def q002s640_provenance(source_root: str | Path, endpoint: str) -> dict[str, Any]:
    """Validate the existing Q002S640 route and distinguish K8 training from export PSF."""
    source_root = Path(source_root)
    manifest_path = source_root / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Q002S640 source manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != "Q002S640_B6_roi_same_anchor_mse":
        raise ValueError("matched CYJ aliases require Q002S640_B6_roi_same_anchor_mse")
    route = manifest.get("route", {})
    sampling = route.get("sampling", {}) if isinstance(route, dict) else {}
    training_samples = manifest.get("training_psf_samples", sampling.get("shared_training_psf_samples"))
    if int(training_samples if training_samples is not None else -1) != 8:
        raise ValueError("Q002S640 provenance must declare training K8 in training_psf_samples or route.sampling")
    export = endpoint_contract(endpoint)["output_query_psf"]
    return {
        "training_route": "Q002S640_B6_roi_same_anchor_mse",
        "training_psf": {"samples": 8, "label": "K8", "semantics": "Q002S640 training PSF"},
        "export_psf": {**export, "post_hoc_map_domain": endpoint == "psf128"},
        "source_manifest": {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)},
    }


def formalize_q002s640_alias(*, subject: str, condition: str, source_root: str | Path, endpoint: str, output: str | Path) -> Path:
    """Create an immutable endpoint alias from an existing Q002S640 reconstruction only."""
    if subject != "CYJ":
        raise ValueError("Prompt 02 formal aliases are CYJ-only")
    parse_cyj_case(condition)
    q002s640_provenance(source_root, endpoint)
    return formalize_proposed_export(subject=subject, condition=condition, source_root=source_root, endpoint=endpoint, output=output)


def export_cyj_fit_first_alias(*, subject: str, condition: str, l100_fit_first_root: str | Path,
                               sparse_fit_first_root: str | Path, sparse_manifest: str | Path,
                               r0_native_maps_root: str | Path, nesvor_executable: str | Path,
                               endpoint: str, output: str | Path, device: str) -> Path:
    """Apply the Prompt-02 CYJ case constraints before the generic immutable exporter."""
    if subject != "CYJ":
        raise ValueError("Prompt 02 Fit-first aliases are CYJ-only")
    parse_cyj_case(condition)
    require_l100_r0_reuse(condition, l100_fit_first_root, sparse_fit_first_root)
    return export_fixed_l100_fit_first(
        subject=subject, condition=condition, l100_fit_first_root=l100_fit_first_root,
        sparse_fit_first_root=sparse_fit_first_root, sparse_manifest=sparse_manifest,
        r0_native_maps_root=r0_native_maps_root, nesvor_executable=nesvor_executable,
        endpoint=endpoint, output=output, device=device,
    )


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _result_files(root: Path) -> Iterable[tuple[Path, Path]]:
    for manifest_path in sorted(root.rglob("manifest.json")):
        metrics_path = manifest_path.with_name("aggregate_metrics.json")
        if metrics_path.is_file():
            yield manifest_path, metrics_path


def _fold_rows(input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for manifest_path, metrics_path in _result_files(input_root):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("subject") != "CYJ":
            raise ValueError(f"CYJ aggregation refuses subject {manifest.get('subject')!r}: {manifest_path}")
        endpoint = manifest.get("endpoint")
        if endpoint not in ENDPOINTS:
            raise ValueError(f"invalid endpoint provenance in {manifest_path}")
        condition = manifest.get("condition")
        level, fold = parse_cyj_case(condition)
        identity = (endpoint, condition)
        if identity in seen:
            raise ValueError(f"duplicate formal result for {endpoint}/{condition}")
        seen.add(identity)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for parameter in PARAMETERS:
            for source_stack, stack in STACK_LABELS.items():
                try:
                    section = metrics["regions"][parameter][source_stack]["l100_full"]
                    proposed, fit_first, deltas = section["proposed"], section["fit_first"], section["paired_deltas"]
                except KeyError as error:
                    raise ValueError(f"formal result lacks l100_full {parameter}/{source_stack}: {metrics_path}") from error
                rows.append({
                    "subject": "CYJ", "endpoint": endpoint, "condition": condition, "level": level, "fold": fold,
                    "parameter": parameter.upper(), "stack": stack, "primary_region": "l100_full",
                    "proposed_rmse": float(proposed["rmse"]), "fit_first_rmse": float(fit_first["rmse"]),
                    "proposed_mae": float(proposed["mae"]), "fit_first_mae": float(fit_first["mae"]),
                    "proposed_bias": float(proposed["bias"]), "fit_first_bias": float(fit_first["bias"]),
                    "reference_mean_ms": float(proposed["reference_mean_ms"]), "proposed_prediction_mean_ms": float(proposed["prediction_mean_ms"]),
                    "fit_first_prediction_mean_ms": float(fit_first["prediction_mean_ms"]),
                    "voxel_count": int(proposed["voxel_count"]), "coverage": float(proposed["coverage_fraction"]),
                    "delta_rmse_proposed_minus_fit_first": float(deltas["delta_rmse_proposed_minus_fit_first"]),
                    "delta_mae_proposed_minus_fit_first": float(deltas["delta_mae_proposed_minus_fit_first"]),
                    "delta_bias_proposed_minus_fit_first": float(deltas["delta_bias_proposed_minus_fit_first"]),
                    "source_manifest": str(manifest_path.resolve()), "source_metrics": str(metrics_path.resolve()),
                })
    if not rows:
        raise ValueError(f"no formal matched aggregate_metrics.json under {input_root}")
    return rows


def _sample_sd(values: list[float]) -> float | None:
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def _summary_rows(folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in folds:
        grouped[(row["endpoint"], row["level"], row["parameter"], row["stack"])].append(row)
    result: list[dict[str, Any]] = []
    for (endpoint, level, parameter, stack), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["fold"])
        observed = [int(row["fold"]) for row in rows]
        expected = list(EXPECTED_FOLDS[level])
        missing = sorted(set(expected).difference(observed))
        metric = lambda key: [float(row[key]) for row in rows]
        proposed_rmse, fit_rmse = metric("proposed_rmse"), metric("fit_first_rmse")
        proposed_mae, fit_mae = metric("proposed_mae"), metric("fit_first_mae")
        result.append({
            "subject": "CYJ", "endpoint": endpoint, "level": level, "parameter": parameter, "stack": stack,
            "primary_region": "l100_full", "expected_fold_count": len(expected), "observed_fold_count": len(rows),
            "observed_folds": observed, "missing_folds": missing,
            "aggregation_status": "COMPLETE" if not missing else "INCOMPLETE_MISSING_FOLDS",
            "proposed_rmse_mean": float(np.mean(proposed_rmse)), "proposed_rmse_sample_sd": _sample_sd(proposed_rmse),
            "fit_first_rmse_mean": float(np.mean(fit_rmse)), "fit_first_rmse_sample_sd": _sample_sd(fit_rmse),
            "proposed_mae_mean": float(np.mean(proposed_mae)), "proposed_mae_sample_sd": _sample_sd(proposed_mae),
            "fit_first_mae_mean": float(np.mean(fit_mae)), "fit_first_mae_sample_sd": _sample_sd(fit_mae),
            "mean_delta_rmse_proposed_minus_fit_first": float(np.mean(metric("delta_rmse_proposed_minus_fit_first"))),
            "mean_delta_mae_proposed_minus_fit_first": float(np.mean(metric("delta_mae_proposed_minus_fit_first"))),
            "proposed_rmse_win_count": int(sum(proposed < fit for proposed, fit in zip(proposed_rmse, fit_rmse))),
            "proposed_mae_win_count": int(sum(proposed < fit for proposed, fit in zip(proposed_mae, fit_mae))),
            "win_total_folds": len(rows),
        })
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _mean_sd(mean: float, sd: float | None) -> str:
    return f"{mean:.4g}" if sd is None else f"{mean:.4g} ± {sd:.3g}"


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# CYJ Q002S640 vs Fit-first matched output summary", "",
        "All rows use the frozen `l100_full` primary support. Incomplete rows contain only observed folds; no missing fold is imputed.", "",
        "| Endpoint | Condition | Parameter | Stack | Proposed RMSE | Fit-first RMSE | Proposed MAE | Fit-first MAE | ΔRMSE | RMSE wins | MAE wins | Status |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['endpoint']} | {row['level']} | {row['parameter']} | {row['stack']} | "
            f"{_mean_sd(row['proposed_rmse_mean'], row['proposed_rmse_sample_sd'])} | "
            f"{_mean_sd(row['fit_first_rmse_mean'], row['fit_first_rmse_sample_sd'])} | "
            f"{_mean_sd(row['proposed_mae_mean'], row['proposed_mae_sample_sd'])} | "
            f"{_mean_sd(row['fit_first_mae_mean'], row['fit_first_mae_sample_sd'])} | "
            f"{row['mean_delta_rmse_proposed_minus_fit_first']:.4g} | "
            f"{row['proposed_rmse_win_count']}/{row['win_total_folds']} | {row['proposed_mae_win_count']}/{row['win_total_folds']} | {row['aggregation_status']} |"
        )
    return "\n".join(lines) + "\n"


def aggregate_cyj_matched_results(input_root: str | Path, output: str | Path) -> Path:
    """Aggregate available immutable formal endpoint results without imputing folds."""
    input_root, output = Path(input_root), Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    fold_rows = _fold_rows(input_root)
    summaries = _summary_rows(fold_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        _write_csv(staging / "fold_metrics.csv", fold_rows)
        _write_csv(staging / "summary.csv", summaries)
        write_immutable_json(staging / "summary.json", {
            "schema": "phase4r_cyj_q002s640_fitfirst_matched_summary/v1", "subject": "CYJ",
            "primary_region": "l100_full", "input_root": str(input_root.resolve()),
            "input_result_files": [_record(path) for pair in _result_files(input_root) for path in pair],
            "fold_rows": fold_rows, "summaries": summaries,
        })
        (staging / "summary.md").write_text(_markdown(summaries), encoding="utf-8")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    alias = subparsers.add_parser("alias-q002s640")
    alias.add_argument("--condition", required=True); alias.add_argument("--source-root", required=True)
    alias.add_argument("--endpoint", choices=ENDPOINTS, required=True); alias.add_argument("--output", required=True)
    fit = subparsers.add_parser("export-fit-first")
    for name in ("condition", "l100-fit-first-root", "sparse-fit-first-root", "sparse-manifest", "r0-native-maps-root", "nesvor-executable", "endpoint", "output", "device"):
        fit.add_argument("--" + name, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--input-root", required=True); aggregate.add_argument("--output", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "alias-q002s640":
        print(formalize_q002s640_alias(subject="CYJ", **args))
    elif command == "export-fit-first":
        print(export_cyj_fit_first_alias(subject="CYJ", **args))
    else:
        print(aggregate_cyj_matched_results(**args))


if __name__ == "__main__":
    main()
