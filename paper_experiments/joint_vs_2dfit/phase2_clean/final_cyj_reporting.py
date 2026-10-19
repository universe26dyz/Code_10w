"""Read-only final CYJ matched summaries and fixed-domain paper figures."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import sha256, write_immutable_json
from .evaluation import _world_sample, reproject_volume_to_native_plane
from .l100_mask_paired_eval import reproject_binary_mask_to_native_plane
from .fit_first import select_verified_native_cache_groups
from .heldout import _heldout_ids
from .visualization import DISPLAY_RANGES_MS, colorbars


METHODS = ("Q002S640", "Q002S640_K256", "Q004S640")
ENDPOINTS = ("central", "psf128")
EXPECTED_FOLDS = {"L100": (0,), "L50": (0, 1), "L25": (0, 1, 2, 3)}
PARAMETERS = {"t1": "T1", "t2": "T2"}


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _case(condition: str) -> tuple[str, int]:
    try:
        level, fold = condition.split("_fold", 1)
        fold_int = int(fold)
    except ValueError as error:
        raise ValueError(f"invalid CYJ condition: {condition}") from error
    if level not in EXPECTED_FOLDS or fold_int not in EXPECTED_FOLDS[level]:
        raise ValueError(f"unsupported CYJ condition: {condition}")
    return level, fold_int


def _metric_records(root: str | Path, method: str, side: str) -> list[dict[str, Any]]:
    root = Path(root)
    if method not in METHODS:
        raise ValueError(f"unsupported Proposed method: {method}")
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for manifest_path in sorted(root.rglob("manifest.json")):
        metrics_path = manifest_path.with_name("aggregate_metrics.json")
        if not metrics_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("subject") != "CYJ":
            raise ValueError(f"final CYJ summary rejects non-CYJ result: {manifest_path}")
        endpoint, condition = manifest.get("endpoint"), manifest.get("condition")
        if endpoint not in ENDPOINTS or not isinstance(condition, str):
            raise ValueError(f"missing endpoint/condition in {manifest_path}")
        identity = (endpoint, condition)
        if identity in seen:
            raise ValueError(f"duplicate {method} result for {endpoint}/{condition}")
        seen.add(identity)
        level, fold = _case(condition)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for parameter, display_parameter in PARAMETERS.items():
            try:
                values = metrics["regions"][parameter]["all_lax"]["l100_full"][side]
            except KeyError as error:
                raise ValueError(f"{metrics_path} lacks all_lax/l100_full/{parameter}/{side}") from error
            rows.append({
                "method": method if side == "proposed" else "Fit-first", "endpoint": endpoint, "condition": condition,
                "level": level, "fold": fold, "parameter": display_parameter, "rmse": values["rmse"],
                "mae": values["mae"], "bias": values["bias"], "voxel_count": values["voxel_count"],
                "coverage_fraction": values["coverage_fraction"], "source_manifest": str(manifest_path.resolve()),
                "source_metrics": str(metrics_path.resolve()), "source_method": method,
            })
    if not rows:
        raise ValueError(f"no formal aggregate results under {root}")
    return rows


def _sample_sd(values: list[float]) -> float | None:
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def _format_mean_sd(mean: float, sample_sd: float | None) -> str:
    return f"{mean:.4g}" if sample_sd is None else f"{mean:.4g} ± {sample_sd:.3g}"


def _summaries(rows: list[dict[str, Any]], fit_rows: dict[tuple[str, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["endpoint"], row["level"], row["parameter"])].append(row)
    output: list[dict[str, Any]] = []
    for (method, endpoint, level, parameter), values in sorted(grouped.items()):
        values.sort(key=lambda row: row["fold"])
        observed = [int(row["fold"]) for row in values]
        missing = sorted(set(EXPECTED_FOLDS[level]).difference(observed))
        metric = lambda name: [float(row[name]) for row in values]
        row: dict[str, Any] = {
            "method": method, "endpoint": endpoint, "level": level, "parameter": parameter,
            "expected_folds": list(EXPECTED_FOLDS[level]), "observed_folds": observed, "missing_folds": missing,
            "status": "COMPLETE" if not missing else "INCOMPLETE_MISSING_FOLDS",
        }
        for name in ("rmse", "mae", "bias"):
            numbers = metric(name)
            row[f"{name}_mean"] = float(np.mean(numbers))
            row[f"{name}_sample_sd"] = _sample_sd(numbers)
        if method != "Fit-first":
            paired = [fit_rows.get((endpoint, value["condition"], parameter)) for value in values]
            if all(item is not None for item in paired):
                for name in ("rmse", "mae", "bias"):
                    deltas = [float(value[name]) - float(reference[name]) for value, reference in zip(values, paired)]
                    row[f"delta_{name}_vs_fit_first_mean"] = float(np.mean(deltas))
                row["proposed_rmse_win_count"] = int(sum(float(value["rmse"]) < float(reference["rmse"]) for value, reference in zip(values, paired)))
                row["proposed_mae_win_count"] = int(sum(float(value["mae"]) < float(reference["mae"]) for value, reference in zip(values, paired)))
                row["win_total_folds"] = len(values)
            else:
                row["fit_first_relative_status"] = "UNAVAILABLE_FOR_ALL_OBSERVED_FOLDS"
        output.append(row)
    return output


def _robustness(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["method"], row["endpoint"], row["parameter"], row["level"]): row for row in summaries}
    rows: list[dict[str, Any]] = []
    identities = sorted({key[:3] for key in by_key})
    for method, endpoint, parameter in identities:
        levels = {level: by_key.get((method, endpoint, parameter, level)) for level in EXPECTED_FOLDS}
        result = {"method": method, "endpoint": endpoint, "parameter": parameter}
        for metric in ("rmse", "mae"):
            for first, second in (("L100", "L50"), ("L50", "L25"), ("L100", "L25")):
                a, b = levels[first], levels[second]
                result[f"{metric}_{first}_to_{second}_change"] = None if a is None or b is None else float(b[f"{metric}_mean"] - a[f"{metric}_mean"])
        rows.append(result)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _paper_table(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    output: list[dict[str, Any]] = []
    for row in rows:
        for metric in ("rmse", "mae"):
            value, sd = row[f"{metric}_mean"], row[f"{metric}_sample_sd"]
            output.append({"method": row["method"], "endpoint": row["endpoint"], "sparsity": row["level"], "parameter": row["parameter"], "metric": metric.upper(), "value": f"{value:.4g}" if sd is None else f"{value:.4g} ± {sd:.3g}"})
    header = "# CYJ final matched table\n\nAll values are `l100_full`; L100 is a single value and L50/L25 use observed-fold mean ± sample SD.\n\n"
    lines = [header, "| Method | Endpoint | Sparsity | Parameter | Metric | Value |", "|---|---|---|---|---|---:|"]
    lines.extend(f"| {row['method']} | {row['endpoint']} | {row['sparsity']} | {row['parameter']} | {row['metric']} | {row['value']} |" for row in output)
    return output, "\n".join(lines) + "\n"


def _summary_markdown(rows: list[dict[str, Any]], robustness: list[dict[str, Any]]) -> str:
    lines = [
        "# CYJ final matched summary", "",
        "All values use the frozen `l100_full` support. L100 is a single value; L50/L25 are observed-fold mean ± sample SD. Missing folds are not imputed.", "",
        "| Method | Endpoint | Level | Parameter | RMSE | MAE | Bias | ΔRMSE vs Fit-first | ΔMAE vs Fit-first | RMSE wins | MAE wins | Status |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        delta_rmse = row.get("delta_rmse_vs_fit_first_mean")
        delta_mae = row.get("delta_mae_vs_fit_first_mean")
        wins = row.get("proposed_rmse_win_count")
        total = row.get("win_total_folds")
        rmse_wins = "—" if wins is None else f"{wins}/{total}"
        mae_wins = "—" if wins is None else f"{row['proposed_mae_win_count']}/{total}"
        lines.append(
            f"| {row['method']} | {row['endpoint']} | {row['level']} | {row['parameter']} | "
            f"{_format_mean_sd(row['rmse_mean'], row['rmse_sample_sd'])} | {_format_mean_sd(row['mae_mean'], row['mae_sample_sd'])} | "
            f"{_format_mean_sd(row['bias_mean'], row['bias_sample_sd'])} | "
            f"{'—' if delta_rmse is None else f'{delta_rmse:.4g}'} | {'—' if delta_mae is None else f'{delta_mae:.4g}'} | "
            f"{rmse_wins} | {mae_wins} | {row['status']} |"
        )
    lines.extend(("", "## Sparsity robustness", "", "| Method | Endpoint | Parameter | RMSE L100→L50 | RMSE L50→L25 | RMSE L100→L25 | MAE L100→L50 | MAE L50→L25 | MAE L100→L25 |", "|---|---|---|---:|---:|---:|---:|---:|---:|"))
    for row in robustness:
        values = [row[f"{metric}_{first}_to_{second}_change"] for metric in ("rmse", "mae") for first, second in (("L100", "L50"), ("L50", "L25"), ("L100", "L25"))]
        lines.append(f"| {row['method']} | {row['endpoint']} | {row['parameter']} | " + " | ".join("—" if value is None else f"{value:.4g}" for value in values) + " |")
    return "\n".join(lines) + "\n"


def aggregate_final_cyj(method_roots: Mapping[str, str | Path], output: str | Path, *, fit_first_source_method: str) -> Path:
    """Aggregate explicitly supplied formal comparison roots without fabricating methods or folds."""
    if fit_first_source_method not in method_roots:
        raise ValueError("fit_first_source_method must be one explicitly supplied Proposed method root")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    proposed_rows = [row for method, root in sorted(method_roots.items()) for row in _metric_records(root, method, "proposed")]
    fit_rows_list = _metric_records(method_roots[fit_first_source_method], fit_first_source_method, "fit_first")
    fit_index = {(row["endpoint"], row["condition"], row["parameter"]): row for row in fit_rows_list}
    all_rows = proposed_rows + fit_rows_list
    summaries = _summaries(all_rows, fit_index)
    robustness = _robustness(summaries)
    paper_rows, paper_markdown = _paper_table(summaries)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        _write_csv(staging / "summary.csv", all_rows + [{"row_type": "level_summary", **row} for row in summaries] + [{"row_type": "robustness", **row} for row in robustness])
        _write_csv(staging / "paper_table.csv", paper_rows)
        (staging / "paper_table.md").write_text(paper_markdown, encoding="utf-8")
        (staging / "summary.md").write_text(_summary_markdown(summaries, robustness), encoding="utf-8")
        write_immutable_json(staging / "summary.json", {"schema": "phase4r_final_cyj_summary/v1", "subject": "CYJ", "primary_region": "l100_full", "fit_first_source_method": fit_first_source_method, "method_roots": {key: str(Path(value).resolve()) for key, value in method_roots.items()}, "fold_rows": all_rows, "level_summaries": summaries, "sparsity_robustness": robustness})
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True); raise
    return output


def longitudinal_plane(mask: np.ndarray, affine: np.ndarray, *, axis: int, index: int) -> dict[str, Any]:
    """Return one full physical display plane from the frozen L100 grid, without cropping."""
    shape = np.asarray(mask).shape
    if len(shape) != 3 or axis not in (0, 1, 2) or not 0 <= index < shape[axis]:
        raise ValueError("invalid frozen-L100 plane definition")
    remaining = [item for item in range(3) if item != axis]
    coordinates = np.indices((shape[remaining[0]], shape[remaining[1]]), dtype=float)
    indices = np.zeros((4, *coordinates.shape[1:]), dtype=float)
    indices[remaining[0]], indices[remaining[1]], indices[axis], indices[3] = coordinates[0], coordinates[1], float(index), 1.0
    world = np.asarray(affine, float) @ indices.reshape(4, -1)
    world = world.reshape(indices.shape)
    spacing = [float(np.linalg.norm(np.asarray(affine, float)[:3, item])) for item in remaining]
    extent = [0.0, (shape[remaining[0]] - 1) * spacing[0], 0.0, (shape[remaining[1]] - 1) * spacing[1]]
    return {"world": world, "shape": list(coordinates.shape[1:]), "extent_mm": extent, "axis": axis, "index": index, "in_plane_axes": remaining}


def final_panel_specifications(parameter: str, methods: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    parameter = parameter.upper()
    cmap, limits = colorbars().mapping[parameter], DISPLAY_RANGES_MS[parameter]
    specs = [{"kind": "reference_ms", "label": "Native reference", "cmap": cmap, "range_ms": limits, "displayed_quantity": "quantitative_map_ms"}]
    for method in methods:
        specs.extend(({"kind": "map_ms", "method": method, "label": method, "cmap": cmap, "range_ms": limits, "displayed_quantity": "quantitative_map_ms"}, {"kind": "absolute_error_ms", "method": method, "label": f"|{method} − Native|", "cmap": cmap, "range_ms": limits, "displayed_quantity": "absolute_error_ms"}))
    return tuple(specs)


def _load_mask_spec(config: Mapping[str, Any], parameter: str) -> Path:
    value = config["l100_masks"][parameter.lower()]
    return Path(value["path"] if isinstance(value, Mapping) else value)


def _image(path: str | Path) -> Any:
    import nibabel as nib
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return nib.load(str(path))


def _plot_panels(path: Path, panels: list[dict[str, Any]], *, extent: list[float], title: str, contour: np.ndarray | None = None) -> None:
    """Render full supplied plane bounds; only masks, never crop, hide background."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.3), squeeze=False)
    for axis, panel in zip(axes[0], panels):
        shown = np.ma.masked_invalid(np.where(panel["support"], panel["values"], np.nan))
        image = axis.imshow(shown.T, origin="lower", extent=extent, cmap=panel["cmap"], vmin=panel["range_ms"][0], vmax=panel["range_ms"][1], interpolation="none", aspect="equal")
        if contour is not None and np.any(contour):
            axis.contour(contour.T.astype(float), levels=[0.5], colors="white", linewidths=.55, origin="lower", extent=extent)
        axis.set_title(panel["label"], fontsize=8)
        axis.set_xticks([]); axis.set_yticks([])
    figure.suptitle(title, fontsize=10)
    figure.colorbar(image, ax=list(axes[0]), fraction=.025, pad=.02, label="ms; shared parameter map scale")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _file_record(path: str | Path) -> dict[str, str]:
    item = Path(path)
    return {"path": str(item.resolve()), "sha256": sha256(item)}


def _render_longitudinal(config: Mapping[str, Any], output: Path, inputs: dict[str, Any]) -> list[dict[str, Any]]:
    section = config["longitudinal"]
    records: list[dict[str, Any]] = []
    methods = tuple(config["methods"])
    for parameter, display_parameter in PARAMETERS.items():
        mask_path = _load_mask_spec(config, parameter)
        mask_image = _image(mask_path)
        mask = np.asarray(mask_image.get_fdata(), bool)
        plane = longitudinal_plane(mask, mask_image.affine, axis=int(section["axis"]), index=int(section["index"]))
        sampled_mask, in_mask = _world_sample(mask.astype(np.uint8), mask_image.affine, plane["world"], order=0)
        support = in_mask & np.isfinite(sampled_mask) & (sampled_mask > .5)
        cmap, limits = colorbars().mapping[display_parameter], DISPLAY_RANGES_MS[display_parameter]
        panels: list[dict[str, Any]] = []
        method_inputs: dict[str, Any] = {}
        for method in methods:
            map_path = Path(config["methods"][method][parameter])
            image = _image(map_path)
            values, valid = _world_sample(image.get_fdata(), image.affine, plane["world"], order=1)
            panels.append({"label": method, "values": values, "support": support & valid & np.isfinite(values), "cmap": cmap, "range_ms": limits})
            method_inputs[method] = _file_record(map_path)
        filename = f"longitudinal_{display_parameter}.png"
        _plot_panels(output / filename, panels, extent=plane["extent_mm"], title=f"{config['subject']} {config['condition']} {config['endpoint']} — frozen L100 longitudinal {display_parameter}")
        inputs[f"l100_mask_{parameter}"] = _file_record(mask_path)
        inputs.update({f"{method}_{parameter}": value for method, value in method_inputs.items()})
        records.append({"parameter": display_parameter, "file": filename, "frozen_l100_source": _file_record(mask_path), "plane": {key: value for key, value in plane.items() if key != "world"}, "world_coordinate_sampling": True, "interpolation": {"map": "linear", "mask": "nearest-neighbor"}, "no_auto_crop": True, "lut_range_ms": list(limits)})
    return records


def _render_native(config: Mapping[str, Any], output: Path, inputs: dict[str, Any]) -> list[dict[str, Any]]:
    native = config["native"]
    sparse_manifest = Path(native["sparse_manifest"])
    sparse = json.loads(sparse_manifest.read_text(encoding="utf-8"))
    stack, original_group = str(native["stack"]), int(native["original_group_idx"])
    selected = select_verified_native_cache_groups(native["r0_native_maps_root"], sparse_manifest, stack, _heldout_ids(sparse, stack))
    matching = np.flatnonzero(np.asarray(selected["group_idx"], int) == original_group)
    if len(matching) != 1:
        raise ValueError(f"requested native group {original_group} is unavailable for {stack}")
    index = int(matching[0])
    affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ np.asarray(selected["affine_lps_rc"][index], float)
    source_support = np.asarray(selected["source_support"][index], bool)
    contour = None
    if native.get("myocardium_contour"):
        contour_path = Path(native["myocardium_contour"])
        contour = np.asarray(np.load(contour_path), bool)
        if contour.shape != source_support.shape:
            raise ValueError("myocardium contour must match the selected native plane")
        inputs["myocardium_contour"] = _file_record(contour_path)
    inputs["sparse_manifest"] = _file_record(sparse_manifest)
    provenance = dict(selected["provenance"])
    for key in ("native_maps_path", "geometry_path", "manifest_path"):
        inputs[f"native_cache_{key}"] = _file_record(provenance[key])
    records: list[dict[str, Any]] = []
    methods = tuple(config["methods"])
    for parameter, display_parameter in PARAMETERS.items():
        mask_path = _load_mask_spec(config, parameter)
        mask_image = _image(mask_path)
        l100 = reproject_binary_mask_to_native_plane(mask_image.get_fdata() > .5, mask_image.affine, affine, source_support.shape)
        support = source_support & l100
        cmap, limits = colorbars().mapping[display_parameter], DISPLAY_RANGES_MS[display_parameter]
        reference = np.asarray(selected[f"{parameter}_ms"][index], float)
        panels = [{"label": "Native reference", "values": reference, "support": support & np.isfinite(reference), "cmap": cmap, "range_ms": limits}]
        for method in methods:
            map_path = Path(config["methods"][method][parameter])
            image = _image(map_path)
            values, valid = reproject_volume_to_native_plane(image.get_fdata(), image.affine, affine, source_support.shape)
            available = support & valid & np.isfinite(reference) & np.isfinite(values)
            panels.extend((
                {"label": method, "values": values, "support": available, "cmap": cmap, "range_ms": limits},
                {"label": f"|{method} − Native|", "values": np.abs(values - reference), "support": available, "cmap": cmap, "range_ms": limits},
            ))
        extent = [0.0, float(source_support.shape[0] - 1), 0.0, float(source_support.shape[1] - 1)]
        filename = f"native_{stack}_group{original_group}_{display_parameter}.png"
        _plot_panels(output / filename, panels, extent=extent, title=f"{config['subject']} {config['condition']} {config['endpoint']} — {stack} native group {original_group} {display_parameter}", contour=contour)
        inputs[f"native_l100_mask_{parameter}"] = _file_record(mask_path)
        records.append({"parameter": display_parameter, "file": filename, "stack": stack, "original_group_idx": original_group, "native_reference": provenance, "frozen_l100_source": _file_record(mask_path), "plane_shape_rc": list(source_support.shape), "display_support": "source_support AND reprojected_frozen_L100_mask; myocardium is contour-only", "interpolation": {"map": "linear", "mask": "nearest-neighbor"}, "no_auto_crop": True, "error_display": {"quantity": "absolute_error_ms", "colormap": "same_parameter_map_lut", "range_ms": list(limits), "no_residual_colorbar": True}})
    return records


def render_final_figures(config_path: str | Path, output: str | Path) -> Path:
    """Create immutable, read-only CYJ figures from explicit formal artifact paths."""
    config_path, output = Path(config_path), Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("subject") != "CYJ" or config.get("endpoint") not in ENDPOINTS:
        raise ValueError("final figures are CYJ-only and require an explicit matched endpoint")
    if not isinstance(config.get("condition"), str):
        raise ValueError("final figures require an explicit CYJ condition/fold")
    _case(config["condition"])
    if not isinstance(config.get("methods"), Mapping) or not config["methods"]:
        raise ValueError("at least one explicit method map set is required")
    for method, maps in config["methods"].items():
        if method not in METHODS and method != "Fit-first":
            raise ValueError(f"unsupported method label: {method}")
        if set(maps) != set(PARAMETERS):
            raise ValueError(f"{method} must provide exactly t1 and t2 maps")
    if "native" not in config or "longitudinal" not in config or "l100_masks" not in config:
        raise ValueError("figure config requires longitudinal, native, and frozen L100-mask inputs")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        inputs = {"config": _file_record(config_path)}
        longitudinal = _render_longitudinal(config, staging, inputs)
        native = _render_native(config, staging, inputs)
        manifest = {"schema": "phase4r_final_cyj_figures/v1", "subject": "CYJ", "condition": config.get("condition"), "endpoint": config["endpoint"], "methods": list(config["methods"]), "inputs": inputs, "longitudinal_figures": longitudinal, "native_heldout_figures": native, "shared_parameter_lut_ranges_ms": {key: list(value) for key, value in DISPLAY_RANGES_MS.items()}, "error_display_contract": "absolute_error_ms using same parameter map LUT/range; no magma residual scale or residual colorbar", "code_git_sha": _git_sha()}
        write_immutable_json(staging / "figure_manifest.json", manifest)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--method-root", action="append", required=True, metavar="METHOD=ROOT")
    aggregate.add_argument("--fit-first-source-method", required=True); aggregate.add_argument("--output", required=True)
    figures = commands.add_parser("figures")
    figures.add_argument("--config", required=True); figures.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "figures":
        print(render_final_figures(args.config, args.output))
    else:
        roots = {}
        for item in args.method_root:
            method, separator, root = item.partition("=")
            if not separator or not method or not root:
                raise ValueError("--method-root must be METHOD=ROOT")
            if method in roots:
                raise ValueError(f"duplicate --method-root method: {method}")
            roots[method] = root
        print(aggregate_final_cyj(roots, args.output, fit_first_source_method=args.fit_first_source_method))


if __name__ == "__main__":
    main()
