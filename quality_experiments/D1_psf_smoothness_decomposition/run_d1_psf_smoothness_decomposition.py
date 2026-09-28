"""Read-only D1 central-plane versus map-domain-PSF smoothness diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:  # support the required ``python path/to/script.py --help`` invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quality_experiments.D1_psf_smoothness_decomposition.figures import render_comparison
from quality_experiments.D1_psf_smoothness_decomposition.io import D1Inputs, git_commit, load_d1_inputs, prepare_output_layout, write_json
from quality_experiments.D1_psf_smoothness_decomposition.metrics import DETAIL_METRIC_COLUMNS, d1_metrics, strict_paired_support


METHODS = (("Bloch", "central_bloch", "psf_bloch"), ("FrozenMLP", "central_mlp", "psf_mlp"))
METRIC_COLUMNS = ("N", "bias_ms", "MAE_ms", "RMSE_ms", "NRMSE", "Pearson_r", "NCC", "SSIM", *DETAIL_METRIC_COLUMNS)
INCREMENT_COLUMNS = ("RMSE_ms", "NCC", "SSIM", *DETAIL_METRIC_COLUMNS)


def _mode_row(
    inputs: D1Inputs, parameter: str, stack: str, group: int, method: str, mode: str, support_scope: str,
    native: np.ndarray, prediction: np.ndarray, support: np.ndarray,
) -> dict[str, Any]:
    return {
        "subject_id": inputs.subject_id,
        "parameter": parameter,
        "stack": stack,
        "group_idx": group,
        "method": method,
        "mode": mode,
        "support_scope": support_scope,
        "support_provenance": {
            "method_specific": "native_valid AND finite(native) AND mode_export_support AND finite(mode_prediction)",
            "strict_paired": "native_valid AND finite(native) AND central_bloch_support AND finite(central_bloch) AND psf_bloch_support AND finite(psf_bloch) AND central_mlp_support AND finite(central_mlp) AND psf_mlp_support AND finite(psf_mlp)",
        }[support_scope],
        **d1_metrics(native, prediction, support),
    }


def _weighted_summary(rows: list[dict[str, Any]], metric_columns: tuple[str, ...] = METRIC_COLUMNS) -> dict[str, float | int]:
    total_n = int(sum(int(row["N"]) for row in rows if np.isfinite(row["N"])))
    result: dict[str, float | int] = {"N": total_n}
    weights = np.asarray([float(row["N"]) for row in rows], dtype=float)
    for key in metric_columns:
        if key == "N":
            continue
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        result[key] = float(np.average(values[valid], weights=weights[valid])) if valid.any() else float("nan")
    return result


def _grouped_summaries(rows: list[dict[str, Any]], keys: tuple[str, ...], metric_columns: tuple[str, ...] = METRIC_COLUMNS) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for identifier, members in sorted(groups.items()):
        result.append(dict(zip(keys, identifier)) | {"aggregation": "N-weighted mean of full-grid, per-slice metrics; no resize/crop/pad", **_weighted_summary(members, metric_columns)})
    return result


def _increment_row(
    inputs: D1Inputs, parameter: str, stack: str, group: int, method: str, scope: str,
    native: np.ndarray, central: np.ndarray, psf: np.ndarray, support: np.ndarray,
) -> dict[str, Any]:
    central_metric, psf_metric = d1_metrics(native, central, support), d1_metrics(native, psf, support)
    return {
        "subject_id": inputs.subject_id, "parameter": parameter, "stack": stack, "group_idx": group,
        "method": method, "support_scope": scope, "N": central_metric["N"],
        "support_provenance": {
            "method_psf_paired": "native_valid AND finite(native) AND central_support AND finite(central) AND psf_support AND finite(psf)",
            "strict_paired": "native_valid AND finite(native) AND all four prediction supports AND finite(all four predictions)",
        }[scope],
        **{f"central_{key}": central_metric[key] for key in INCREMENT_COLUMNS},
        **{f"psf_{key}": psf_metric[key] for key in INCREMENT_COLUMNS},
        **{f"delta_psf_minus_central_{key}": float(psf_metric[key]) - float(central_metric[key]) for key in INCREMENT_COLUMNS},
    }


def evaluate(inputs: D1Inputs) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    increments: list[dict[str, Any]] = []
    masks: dict[str, np.ndarray] = {}
    for parameter, stacks in inputs.arrays.items():
        for stack, planes in stacks.items():
            native = planes["native"]
            all_values = [planes[key].values for key in ("central_bloch", "psf_bloch", "central_mlp", "psf_mlp")]
            all_supports = [planes[key].support for key in ("central_bloch", "psf_bloch", "central_mlp", "psf_mlp")]
            strict = strict_paired_support(native.support, all_supports, all_values)
            masks[f"{parameter}_{stack}_native_valid"] = native.support
            masks[f"{parameter}_{stack}_strict_paired_support"] = strict
            for method, central_key, psf_key in METHODS:
                central, psf = planes[central_key], planes[psf_key]
                central_specific = native.support & central.support & np.isfinite(central.values)
                psf_specific = native.support & psf.support & np.isfinite(psf.values)
                method_pair = native.support & central.support & psf.support & np.isfinite(central.values) & np.isfinite(psf.values)
                masks[f"{parameter}_{stack}_{method}_central_method_specific_support"] = central_specific
                masks[f"{parameter}_{stack}_{method}_psf_method_specific_support"] = psf_specific
                masks[f"{parameter}_{stack}_{method}_psf_paired_support"] = method_pair
                for group in range(native.values.shape[0]):
                    rows.append(_mode_row(inputs, parameter, stack, group, method, "CENTRAL_PLANE", "method_specific", native.values[group], central.values[group], central_specific[group]))
                    rows.append(_mode_row(inputs, parameter, stack, group, method, "MAP_DOMAIN_PSF", "method_specific", native.values[group], psf.values[group], psf_specific[group]))
                    rows.append(_mode_row(inputs, parameter, stack, group, method, "CENTRAL_PLANE", "strict_paired", native.values[group], central.values[group], strict[group]))
                    rows.append(_mode_row(inputs, parameter, stack, group, method, "MAP_DOMAIN_PSF", "strict_paired", native.values[group], psf.values[group], strict[group]))
                    increments.append(_increment_row(inputs, parameter, stack, group, method, "method_psf_paired", native.values[group], central.values[group], psf.values[group], method_pair[group]))
                    increments.append(_increment_row(inputs, parameter, stack, group, method, "strict_paired", native.values[group], central.values[group], psf.values[group], strict[group]))
    return rows, increments, masks


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def _render_figures(inputs: D1Inputs, masks: dict[str, np.ndarray], root: Path) -> dict[str, int]:
    selected: dict[str, int] = {}
    for stack in ("sax", "2ch", "4ch"):
        support = masks[f"T1_{stack}_strict_paired_support"] & masks[f"T2_{stack}_strict_paired_support"]
        selected[stack] = int(np.argmax(support.sum(axis=(1, 2))))
    for parameter, stacks in inputs.arrays.items():
        for stack, planes in stacks.items():
            for group in range(planes["native"].values.shape[0]):
                target_root = root / ("representative" if group == selected[stack] else "all_slices")
                render_comparison(
                    parameter=parameter, stack=stack, group=group, native=planes["native"].values[group],
                    central_bloch=planes["central_bloch"].values[group], psf_bloch=planes["psf_bloch"].values[group],
                    central_mlp=planes["central_mlp"].values[group], psf_mlp=planes["psf_mlp"].values[group],
                    support=masks[f"{parameter}_{stack}_strict_paired_support"][group],
                    target=target_root / f"D1_{parameter}_{stack}_group_{group:03d}.png",
                )
    return selected


def _summary_text(global_rows: list[dict[str, Any]], increment_rows: list[dict[str, Any]]) -> str:
    lookup = {(row["parameter"], row["method"], row["mode"], row["support_scope"]): row for row in global_rows}
    increment_lookup = {(row["parameter"], row["method"], row["support_scope"]): row for row in increment_rows}
    lines = ["# D1 result summary", "", "Formal result generated: YES", "", "All values below are computed from actual server inputs; no conclusion is prefilled.", ""]
    for parameter in ("T1", "T2"):
        lines.extend([f"## {parameter}", ""])
        for method in ("Bloch", "FrozenMLP"):
            central, psf = lookup[(parameter, method, "CENTRAL_PLANE", "strict_paired")], lookup[(parameter, method, "MAP_DOMAIN_PSF", "strict_paired")]
            delta = increment_lookup[(parameter, method, "strict_paired")]
            lines.append(f"- {method}: central RMSE={central['RMSE_ms']:.4g}, NCC={central['NCC']:.4g}, gradient ratio={central['gradient_magnitude_ratio']:.4g}, HF ratio={central['high_frequency_energy_ratio']:.4g}; PSF RMSE={psf['RMSE_ms']:.4g}, NCC={psf['NCC']:.4g}, gradient ratio={psf['gradient_magnitude_ratio']:.4g}, HF ratio={psf['high_frequency_energy_ratio']:.4g}; PSF−central RMSE={delta['delta_psf_minus_central_RMSE_ms']:.4g}, NCC={delta['delta_psf_minus_central_NCC']:.4g}, gradient ratio={delta['delta_psf_minus_central_gradient_magnitude_ratio']:.4g}, HF ratio={delta['delta_psf_minus_central_high_frequency_energy_ratio']:.4g}.")
        lines.append("")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> Path:
    output = prepare_output_layout(args.output, sys.argv)
    inputs = load_d1_inputs(subject_id=args.subject_id, trad_run=args.trad_run, mlp_run=args.mlp_run, trad_eval=args.trad_eval, mlp_eval=args.mlp_eval, native_reference_root=args.native_reference_root, preprocessed_root=args.preprocessed_root)
    rows, increments, masks = evaluate(inputs)
    per_stack = _grouped_summaries(rows, ("subject_id", "parameter", "stack", "method", "mode", "support_scope", "support_provenance"))
    global_rows = _grouped_summaries(rows, ("subject_id", "parameter", "method", "mode", "support_scope", "support_provenance"))
    increment_metrics = tuple(f"delta_psf_minus_central_{key}" for key in INCREMENT_COLUMNS)
    increment_stack = _grouped_summaries(increments, ("subject_id", "parameter", "stack", "method", "support_scope", "support_provenance"), increment_metrics)
    increment_global = _grouped_summaries(increments, ("subject_id", "parameter", "method", "support_scope", "support_provenance"), increment_metrics)
    _write_csv(output / "metrics/D1_per_slice.csv", rows)
    _write_csv(output / "metrics/D1_per_stack.csv", per_stack)
    write_json(output / "metrics/D1_global.json", {"schema": "code10w_d1_global_metrics/v1", "aggregation": "N-weighted mean of unresized per-slice metrics", "rows": global_rows})
    write_json(output / "metrics/D1_sharpness_metrics.json", {"schema": "code10w_d1_sharpness_metrics/v1", "detail_metric_definitions": list(DETAIL_METRIC_COLUMNS), "per_stack": per_stack, "global": global_rows})
    write_json(output / "metrics/D1_psf_increment.json", {"schema": "code10w_d1_psf_increment/v1", "definition": "PSF-matched metric minus central-plane metric, evaluated on a common mode-pair support", "per_slice": increments, "per_stack": increment_stack, "global": increment_global})
    np.savez_compressed(output / "artifacts/paired_support_masks.npz", **masks)
    selected = _render_figures(inputs, masks, output / "figures")
    write_json(output / "artifacts/selected_groups.json", {"schema": "code10w_d1_selected_groups/v1", "selection": "largest joint T1/T2 strict paired support per stack", "selected_groups": selected})
    manifest = {"schema": "code10w_d1_psf_smoothness_decomposition/v1", "implementation_revision": "D1_psf_smoothness_decomposition_baseline_v2", "experiment_id": "D1_BASELINE_PSF_SMOOTHNESS_V1", "category": "diagnostic", "subject_id": args.subject_id, "status": "COMPLETE", "baseline_read_only": True, "git_commit": git_commit(Path(__file__).resolve().parents[2]), "inputs": inputs.provenance | {"myocardium_mask_root": str(Path(args.myocardium_mask_root).resolve())}, "support_provenance": {"method_specific": "native_valid AND each mode's own exported support AND finite native/prediction", "strict_paired": "native_valid AND all four mode supports AND finite native/all four predictions", "psf_increment_method_pair": "native_valid AND central/PSF supports for that method AND finite native/central/PSF", "psf_increment_strict": "D1 strict paired support"}, "output_files": [str(path.relative_to(output)) for path in sorted(output.rglob("*")) if path.is_file()]}
    write_json(output / "manifest.json", manifest)
    (output / "README.md").write_text("# D1 output\n\nRead-only baseline diagnostic. Metrics use full native grids; any figure crop is display-only. See `manifest.json` for provenance and `RESULT_SUMMARY.md` for actual values.\n", encoding="utf-8")
    (output / "RESULT_SUMMARY.md").write_text(_summary_text(global_rows, increment_global), encoding="utf-8")
    manifest["output_files"] = [str(path.relative_to(output)) for path in sorted(output.rglob("*")) if path.is_file()]
    write_json(output / "manifest.json", manifest)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--trad-run", required=True)
    parser.add_argument("--mlp-run", required=True)
    parser.add_argument("--trad-eval", required=True)
    parser.add_argument("--mlp-eval", required=True)
    parser.add_argument("--preprocessed-root", required=True)
    parser.add_argument("--native-reference-root", required=True)
    parser.add_argument("--myocardium-mask-root", required=True)
    parser.add_argument("--output", required=True)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
