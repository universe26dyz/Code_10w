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
from .current_native_central_component_mask import load_validated_current_native_mask_plane
from .fit_first import NATIVE_FIT_METHOD, _complete_source_group, _dictionary_contract, select_verified_native_cache_groups
from .heldout import _heldout_ids
from .visualization import DISPLAY_RANGES_MS, colorbars


METHODS = ("Q002S640", "Q002S640_K256", "Q004S640")
ENDPOINTS = ("central", "psf128")
EXPECTED_FOLDS = {"L100": (0,), "L50": (0, 1), "L25": (0, 1, 2, 3)}
PARAMETERS = {"t1": "T1", "t2": "T2"}
NATIVE_MOSAIC_METHODS = ("Fit-first", "Q002S640", "Q004S640")


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


def meaningful_longitudinal_index(mask: np.ndarray, *, axis: int, index: int | str | None = None) -> tuple[int, str]:
    """Keep an explicit plane, or select the median frozen-L100 support plane."""
    shape = np.asarray(mask).shape
    if len(shape) != 3 or axis not in (0, 1, 2):
        raise ValueError("invalid frozen-L100 plane axis")
    if index not in (None, "support_median"):
        chosen = int(index)
        if not 0 <= chosen < shape[axis]:
            raise ValueError("invalid frozen-L100 plane definition")
        return chosen, "configured_index"
    supported = np.flatnonzero(np.any(np.asarray(mask, bool), axis=tuple(item for item in range(3) if item != axis)))
    if not len(supported):
        raise ValueError("frozen-L100 mask has no support for a longitudinal plane")
    return int(supported[(len(supported) - 1) // 2]), "support_median"


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


def native_mosaic_methods(methods: Mapping[str, Any]) -> tuple[str, ...]:
    """Put the formal comparison methods first, independent of JSON key order."""
    formal = tuple(method for method in NATIVE_MOSAIC_METHODS if method in methods)
    return formal + tuple(method for method in methods if method not in formal)


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

    figure = plt.figure(figsize=(3.0 * len(panels) + .65, 3.3))
    grid = figure.add_gridspec(1, len(panels) + 1, width_ratios=[1.0] * len(panels) + [.055], wspace=.12)
    axes = [figure.add_subplot(grid[0, index]) for index in range(len(panels))]
    colorbar_axis = figure.add_subplot(grid[0, -1])
    for axis, panel in zip(axes, panels):
        shown = np.ma.masked_invalid(np.where(panel["support"], panel["values"], np.nan))
        image = axis.imshow(shown.T, origin="lower", extent=extent, cmap=panel["cmap"], vmin=panel["range_ms"][0], vmax=panel["range_ms"][1], interpolation="none", aspect="equal")
        if contour is not None and np.any(contour):
            axis.contour(contour.T.astype(float), levels=[0.5], colors="white", linewidths=.55, origin="lower", extent=extent)
        axis.set_title(panel["label"], fontsize=8)
        axis.set_xticks([]); axis.set_yticks([])
    figure.suptitle(title, fontsize=10)
    figure.colorbar(image, cax=colorbar_axis, label="ms; shared parameter map scale")
    figure.subplots_adjust(top=.83, bottom=.04, left=.02, right=.98)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _native_mosaic_row(row: Any) -> tuple[int, str, list[dict[str, Any]], list[float], np.ndarray | None]:
    if isinstance(row, Mapping):
        group, panels = int(row["group_idx"]), list(row["panels"])
        extent, contour = row.get("extent"), row.get("contour")
        label = str(row.get("group_label", f"group {group}"))
    else:
        group, panels, *extra = row
        group, panels = int(group), list(panels)
        extent, contour = (extra + [None, None])[:2]
        label = f"group {group}"
    values = np.asarray(panels[0]["values"])
    if extent is None:
        extent = [0.0, float(values.shape[0] - 1), 0.0, float(values.shape[1] - 1)]
    return group, label, panels, list(extent), None if contour is None else np.asarray(contour, bool)


def _plot_native_mosaic(path: Path, rows: list[Any], *, title: str) -> None:
    """Render all requested native planes with one shared parameter colorbar."""
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("native mosaic requires at least one group")
    normalized = [_native_mosaic_row(row) for row in rows]
    panel_count = len(normalized[0][2])
    if not panel_count or any(len(panels) != panel_count for _, _, panels, _, _ in normalized):
        raise ValueError("native mosaic rows must have an identical non-empty panel layout")
    figure = plt.figure(figsize=(2.5 * panel_count + .65, max(3.2, 2.55 * len(normalized) + .5)))
    grid = figure.add_gridspec(len(normalized), panel_count + 1, width_ratios=[1.0] * panel_count + [.06], wspace=.12, hspace=.12)
    map_handle = None
    for row_index, (_, group_label, panels, extent, contour) in enumerate(normalized):
        for column, panel in enumerate(panels):
            axis = figure.add_subplot(grid[row_index, column])
            shown = np.ma.masked_invalid(np.where(panel["support"], panel["values"], np.nan))
            image = axis.imshow(shown.T, origin="lower", extent=extent, cmap=panel["cmap"], vmin=panel["range_ms"][0], vmax=panel["range_ms"][1], interpolation="none", aspect="equal")
            if contour is not None and np.any(contour):
                axis.contour(contour.T.astype(float), levels=[.5], colors="white", linewidths=.55, origin="lower", extent=extent)
            if row_index == 0:
                axis.set_title(panel["label"], fontsize=8)
            if column == 0:
                axis.set_ylabel(group_label, fontsize=8)
            axis.set_xticks([]); axis.set_yticks([])
            if not panel["label"].startswith("|"):
                map_handle = image
    map_colorbar_axis = figure.add_subplot(grid[:, panel_count])
    if map_handle is None:
        raise ValueError("native mosaic needs a quantitative-map panel")
    figure.colorbar(map_handle, cax=map_colorbar_axis, label="parameter (ms); shared map/error scale")
    figure.suptitle(title, fontsize=10)
    figure.subplots_adjust(top=.9, bottom=.03, left=.03, right=.98)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def native_preview_groups(native: Mapping[str, Any], sparse: Mapping[str, Any], *, source_groups: list[int] | None = None) -> tuple[str, list[int]]:
    """Resolve display groups while retaining the sparse manifest as authority."""
    mode = str(native.get("mode", "single"))
    stack = str(native["stack"])
    authoritative = _heldout_ids(dict(sparse), stack)
    if mode == "all_groups":
        return mode, authoritative
    if mode == "all_source_groups":
        if source_groups is None:
            raise ValueError("native all_source_groups mode requires cached source group IDs")
        return mode, sorted({int(group) for group in source_groups})
    if mode != "single":
        raise ValueError("native.mode must be 'single', 'all_groups', or 'all_source_groups'")
    if "original_group_idx" not in native:
        raise ValueError("native single mode requires original_group_idx")
    group = int(native["original_group_idx"])
    if group not in authoritative:
        raise ValueError(f"requested native group {group} is unavailable for {stack}")
    return mode, [group]


def native_group_statuses(groups: list[int], sparse: Mapping[str, Any], stack: str) -> dict[int, str]:
    """Label source groups without changing sparse membership or numerical support."""
    if sparse.get("condition") == "L100":
        return {int(group): "source" for group in groups}
    heldout = set(_heldout_ids(dict(sparse), stack))
    return {int(group): "held-out" if int(group) in heldout else "observed" for group in groups}


def native_group_label(group: int, status: str) -> str:
    return f"group {int(group):02d} [{status}]"


def select_sax_reference_group(groups: list[int], section: Mapping[str, Any]) -> tuple[int, str]:
    """Choose a reproducible source slice for SCMR-style centre-line previews."""
    ordered = sorted({int(group) for group in groups})
    if not ordered:
        raise ValueError("SCMR-style longitudinal preview requires SAX source groups")
    if "sax_reference_group_idx" in section:
        chosen = int(section["sax_reference_group_idx"])
        if chosen not in ordered:
            raise ValueError("configured sax_reference_group_idx is not a SAX source group")
        return chosen, "configured_sax_reference_group_idx"
    return ordered[(len(ordered) - 1) // 2], "sorted_source_group_lower_median"


def sax_centerline_cuts(shape_rc: tuple[int, int]) -> tuple[tuple[str, int], tuple[str, int]]:
    if len(shape_rc) != 2 or min(shape_rc) <= 0:
        raise ValueError("invalid SAX source-plane shape")
    return (("center_row", (int(shape_rc[0]) - 1) // 2), ("center_column", (int(shape_rc[1]) - 1) // 2))


def scmr_longitudinal_panel_labels() -> tuple[str, str, str, str, str]:
    return ("SAX source + centerline overlay", "Original SAX-stack longitudinal reformatting", "Fit-first", "Q002S640", "Q004S640")


def native_preview_stack_configs(native: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Accept the original single-stack config or explicit 2CH/4CH stack entries."""
    if "stacks" not in native:
        return [dict(native)]
    base = {key: value for key, value in native.items() if key != "stacks"}
    entries = native["stacks"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("native.stacks must be a non-empty list")
    result: list[dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, str):
            result.append(base | {"stack": entry})
        elif isinstance(entry, Mapping):
            result.append(base | dict(entry))
        else:
            raise ValueError("native.stacks entries must be stack names or mappings")
    stacks = [str(item.get("stack", "")) for item in result]
    if len(stacks) != len(set(stacks)) or any(not stack for stack in stacks):
        raise ValueError("native.stacks must name each stack exactly once")
    return result


def _file_record(path: str | Path) -> dict[str, str]:
    item = Path(path)
    return {"path": str(item.resolve()), "sha256": sha256(item)}


def _load_all_verified_native_cache_groups(native_root: str | Path, sparse_manifest: str | Path, stack: str) -> dict[str, Any]:
    """Read every cached source group after the existing sparse provenance check passes.

    This is report-only: it does not alter the held-out subset used by the
    numerical evaluator or by reconstruction.
    """
    root, manifest_path = Path(native_root), Path(sparse_manifest)
    sparse = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sparse.get("status") != "ELIGIBLE":
        raise ValueError("all-source native reporting requires an ELIGIBLE sparse manifest")
    sparse_stack = sparse.get("stacks", {}).get(stack)
    if not isinstance(sparse_stack, Mapping):
        raise ValueError(f"{stack}: sparse manifest stack is missing")
    cache_root = root / stack
    maps_path, geometry_path, cache_manifest_path = cache_root / "native_maps.npz", cache_root / "geometry.npz", cache_root / "manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    if cache_manifest.get("method") != NATIVE_FIT_METHOD or cache_manifest.get("dictionary_contract") != _dictionary_contract():
        raise ValueError(f"{stack}: native cache does not satisfy the frozen dictionary contract")
    if cache_manifest.get("native_maps_sha256") != sha256(maps_path) or cache_manifest.get("geometry_sha256") != sha256(geometry_path):
        raise ValueError(f"{stack}: native cache SHA256 mismatch")
    sparse_source = sparse_stack.get("source_observation", {})
    if cache_manifest.get("source_observations", {}).get("sha256") != sparse_source.get("sha256"):
        raise ValueError(f"{stack}: sparse and cached source observation identities differ")
    source = Path(str(cache_manifest["source_observations"]["path"]))
    if not source.is_file() or sha256(source) != cache_manifest["source_observations"]["sha256"]:
        raise ValueError(f"{stack}: cached source observations are unavailable or changed")
    with np.load(maps_path, allow_pickle=False) as maps, np.load(geometry_path, allow_pickle=False) as geometry, np.load(source, allow_pickle=False) as observations:
        if any(name not in maps.files for name in ("t1_ms", "t2_ms", "b1", "group_idx")) or any(name not in geometry.files for name in ("affine_lps_rc", "pixel_spacing_rc_mm", "slice_thickness_mm")):
            raise ValueError(f"{stack}: native cache is incomplete")
        groups = [int(group) for group in maps["group_idx"]]
        if len(groups) != len(set(groups)):
            raise ValueError(f"{stack}: native cache group IDs are not unique")
        if any(array.shape[0] != len(groups) for array in (maps["t1_ms"], maps["t2_ms"], maps["b1"], geometry["affine_lps_rc"], geometry["pixel_spacing_rc_mm"], geometry["slice_thickness_mm"])):
            raise ValueError(f"{stack}: native cache group dimensions disagree")
        order = np.argsort(groups)
        groups = [groups[index] for index in order]
        support = []
        for group in groups:
            rows = np.flatnonzero(observations["group_idx"] == group)
            if not len(rows):
                raise ValueError(f"{stack}: cached group {group} is missing from source observations")
            _complete_source_group(observations, group, source)
            support.append(np.all(observations["masks"][rows].astype(bool), axis=0))
        result = {name: np.asarray(maps[name])[order].copy() for name in ("t1_ms", "t2_ms", "b1")}
        result.update({name: np.asarray(geometry[name])[order].copy() for name in ("affine_lps_rc", "pixel_spacing_rc_mm", "slice_thickness_mm")})
    result["group_idx"] = np.asarray(groups, dtype=int)
    result["source_support"] = np.asarray(support, dtype=bool)
    result["provenance"] = {
        "native_reference_source": "R0_native_cache_all_source_groups_report_only",
        "native_reference_cache_root": str(root.resolve()),
        "native_maps_path": str(maps_path.resolve()), "native_maps_sha256": sha256(maps_path),
        "geometry_path": str(geometry_path.resolve()), "geometry_sha256": sha256(geometry_path),
        "manifest_path": str(cache_manifest_path.resolve()), "manifest_sha256": sha256(cache_manifest_path),
        "source_observations": _file_record(source), "selected_original_group_ids": groups,
        "report_only_all_source_groups": True,
    }
    return result


def _render_longitudinal(config: Mapping[str, Any], output: Path, inputs: dict[str, Any]) -> list[dict[str, Any]]:
    section = config["longitudinal"]
    records: list[dict[str, Any]] = []
    methods = tuple(config["methods"])
    for parameter, display_parameter in PARAMETERS.items():
        mask_path = _load_mask_spec(config, parameter)
        mask_image = _image(mask_path)
        mask = np.asarray(mask_image.get_fdata(), bool)
        index, selection_mode = meaningful_longitudinal_index(mask, axis=int(section["axis"]), index=section.get("index", "support_median"))
        plane = longitudinal_plane(mask, mask_image.affine, axis=int(section["axis"]), index=index)
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
        records.append({"parameter": display_parameter, "file": filename, "frozen_l100_source": _file_record(mask_path), "plane": {key: value for key, value in plane.items() if key != "world"}, "plane_selection": selection_mode, "world_coordinate_sampling": True, "interpolation": {"map": "linear", "mask": "nearest-neighbor"}, "no_auto_crop": True, "lut_range_ms": list(limits)})
    return records


def _sax_stack_affine_ras(sax: Mapping[str, Any]) -> np.ndarray:
    affines = np.diag([-1.0, -1.0, 1.0, 1.0]) @ np.asarray(sax["affine_lps_rc"], float)
    result = np.asarray(affines[0], float).copy()
    if len(affines) > 1:
        result[:3, 2] = np.median(np.diff(affines[:, :3, 3], axis=0), axis=0)
    else:
        normal = np.cross(result[:3, 0], result[:3, 1]); normal /= np.linalg.norm(normal)
        result[:3, 2] = normal * float(np.asarray(sax["slice_thickness_mm"], float)[0])
    if not np.any(result[:3, 2]):
        raise ValueError("SAX source groups do not define a through-plane direction")
    return result


def _sax_cut_plane(stack_shape: tuple[int, int, int], affine: np.ndarray, *, cut: str, index: int) -> dict[str, Any]:
    rows, columns, groups = stack_shape
    if cut == "center_row":
        first, second = np.indices((columns, groups), dtype=float)
        indices = np.stack((np.full_like(first, index), first, second, np.ones_like(first)))
        in_plane_axes = (1, 2)
    elif cut == "center_column":
        first, second = np.indices((rows, groups), dtype=float)
        indices = np.stack((first, np.full_like(first, index), second, np.ones_like(first)))
        in_plane_axes = (0, 2)
    else:
        raise ValueError(f"unsupported SAX centre-line cut: {cut}")
    world = np.asarray(affine, float) @ indices.reshape(4, -1)
    world = world.reshape(indices.shape)
    spacing = [float(np.linalg.norm(np.asarray(affine, float)[:3, axis])) for axis in in_plane_axes]
    return {"world": world, "extent_mm": [0.0, (indices.shape[1] - 1) * spacing[0], 0.0, (indices.shape[2] - 1) * spacing[1]], "shape": list(indices.shape[1:]), "cut": cut, "index": int(index), "in_plane_axes": list(in_plane_axes)}


def _dense_physical_plane(source_plane: Mapping[str, Any], *, spacing_mm: float = 1.0) -> dict[str, Any]:
    """Resample one physical SAX cut at approximately 1-mm in-plane spacing.

    The source stack and every reconstruction retain their own raster, but all
    reconstruction samples are taken on this one common physical plane.
    """
    if spacing_mm <= 0:
        raise ValueError("dense physical plane spacing must be positive")
    source_world = np.asarray(source_plane["world"], float)
    origin = source_world[:3, 0, 0]
    first_vector = source_world[:3, -1, 0] - origin
    second_vector = source_world[:3, 0, -1] - origin
    first_length, second_length = float(np.linalg.norm(first_vector)), float(np.linalg.norm(second_vector))
    if first_length <= 0 or second_length <= 0:
        raise ValueError("SAX centre-line plane must span two non-zero physical axes")
    first_count, second_count = int(np.ceil(first_length / spacing_mm)) + 1, int(np.ceil(second_length / spacing_mm)) + 1
    first_positions = np.linspace(0.0, first_length, first_count)
    second_positions = np.linspace(0.0, second_length, second_count)
    first_unit, second_unit = first_vector / first_length, second_vector / second_length
    first, second = np.meshgrid(first_positions, second_positions, indexing="ij")
    world = np.empty((4, first_count, second_count), float)
    world[:3] = origin[:, None, None] + first_unit[:, None, None] * first + second_unit[:, None, None] * second
    world[3] = 1.0
    return {"world": world, "extent_mm": [0.0, first_length, 0.0, second_length], "shape": [first_count, second_count], "requested_spacing_mm": float(spacing_mm), "actual_spacing_mm": [first_length / (first_count - 1), second_length / (second_count - 1)], "same_physical_plane_as_source": True}


def _plot_scmr_longitudinal(path: Path, rows: list[dict[str, Any]], *, title: str, limits: tuple[float, float], cmap: Any) -> None:
    """Two physical SAX centre-line cuts with one parameter colorbar."""
    import matplotlib.pyplot as plt

    labels = scmr_longitudinal_panel_labels()
    figure = plt.figure(figsize=(15.2, 6.1))
    grid = figure.add_gridspec(2, 6, width_ratios=[1.0] * 5 + [.06], wspace=.12, hspace=.20)
    handle = None
    for row_index, row in enumerate(rows):
        source = row["source"]
        source_axis = figure.add_subplot(grid[row_index, 0])
        shown = np.ma.masked_invalid(np.where(source["support"], source["values"], np.nan))
        handle = source_axis.imshow(shown.T, origin="lower", extent=source["extent_mm"], cmap=cmap, vmin=limits[0], vmax=limits[1], interpolation="none", aspect="equal")
        if row["cut"] == "center_row":
            source_axis.axvline(row["source_line_coordinate_mm"], color="cyan", linewidth=1.2)
        else:
            source_axis.axhline(row["source_line_coordinate_mm"], color="cyan", linewidth=1.2)
        source_axis.set_ylabel(row["cut"].replace("_", " "), fontsize=8)
        source_axis.set_title(labels[0], fontsize=8)
        source_axis.set_xticks([]); source_axis.set_yticks([])
        for column, panel in enumerate(row["longitudinal"], start=1):
            axis = figure.add_subplot(grid[row_index, column])
            shown = np.ma.masked_invalid(np.where(panel["support"], panel["values"], np.nan))
            handle = axis.imshow(shown.T, origin="lower", extent=row["extent_mm"], cmap=cmap, vmin=limits[0], vmax=limits[1], interpolation="none", aspect="equal")
            if row_index == 0:
                axis.set_title(labels[column], fontsize=8)
            axis.set_xticks([]); axis.set_yticks([])
    colorbar_axis = figure.add_subplot(grid[:, -1])
    figure.colorbar(handle, cax=colorbar_axis, label="ms; shared parameter map scale")
    figure.suptitle(title, fontsize=10)
    figure.subplots_adjust(top=.91, bottom=.04, left=.03, right=.98)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _render_scmr_style_longitudinal(config: Mapping[str, Any], output: Path, inputs: dict[str, Any]) -> list[dict[str, Any]]:
    """Optional report-only SAX source centre-line preview; primary figures stay unchanged."""
    section = config.get("scmr_longitudinal")
    if section is None:
        return []
    if not isinstance(section, Mapping):
        raise ValueError("scmr_longitudinal must be a mapping")
    native = config.get("native", {})
    native_root = section.get("r0_native_maps_root", native.get("r0_native_maps_root"))
    sparse_manifest = section.get("sparse_manifest", native.get("sparse_manifest"))
    if not native_root or not sparse_manifest:
        raise ValueError("scmr_longitudinal requires r0_native_maps_root and sparse_manifest")
    sax = _load_all_verified_native_cache_groups(native_root, sparse_manifest, "sax")
    groups = [int(group) for group in sax["group_idx"]]
    reference_group, selection_rule = select_sax_reference_group(groups, section)
    reference_index = groups.index(reference_group)
    source_affine = _sax_stack_affine_ras(sax)
    source_shape = tuple(int(value) for value in np.asarray(sax["source_support"]).shape[1:])
    stack_shape = (source_shape[0], source_shape[1], len(groups))
    source_spacing = np.asarray(sax["pixel_spacing_rc_mm"][reference_index], float)
    source_extent = [0.0, (source_shape[0] - 1) * float(source_spacing[0]), 0.0, (source_shape[1] - 1) * float(source_spacing[1])]
    methods = tuple(scmr_longitudinal_panel_labels()[2:])
    if any(method not in config["methods"] for method in methods):
        raise ValueError("scmr_longitudinal requires Fit-first, Q002S640, and Q004S640 map inputs")
    records: list[dict[str, Any]] = []
    for parameter, display_parameter in PARAMETERS.items():
        source_stack = np.moveaxis(np.asarray(sax[f"{parameter}_ms"], float), 0, -1)
        support_stack = np.moveaxis(np.asarray(sax["source_support"], bool), 0, -1)
        method_images = {method: _image(config["methods"][method][parameter]) for method in methods}
        rows: list[dict[str, Any]] = []
        cut_records: list[dict[str, Any]] = []
        for cut, index in sax_centerline_cuts(source_shape):
            plane = _sax_cut_plane(stack_shape, source_affine, cut=cut, index=index)
            dense_plane = _dense_physical_plane(plane, spacing_mm=float(section.get("reconstruction_spacing_mm", 1.0)))
            native_values, native_valid = _world_sample(source_stack, source_affine, plane["world"], order=0)
            native_support, native_support_valid = _world_sample(support_stack.astype(np.uint8), source_affine, plane["world"], order=0)
            longitudinal = [{"label": scmr_longitudinal_panel_labels()[1], "values": native_values, "support": native_valid & native_support_valid & (native_support > .5) & np.isfinite(native_values)}]
            for method, image in method_images.items():
                values, valid = _world_sample(image.get_fdata(), image.affine, dense_plane["world"], order=1)
                longitudinal.append({"label": method, "values": values, "support": valid & np.isfinite(values)})
            line_coordinate = float(index * source_spacing[0 if cut == "center_row" else 1])
            rows.append({"cut": cut, "source": {"values": np.asarray(sax[f"{parameter}_ms"][reference_index], float), "support": np.asarray(sax["source_support"][reference_index], bool), "extent_mm": source_extent}, "source_line_coordinate_mm": line_coordinate, "longitudinal": longitudinal, "extent_mm": plane["extent_mm"]})
            cut_records.append({"cut": cut, "source_line_index": index, "source_line_coordinate_mm": line_coordinate, "source_plane_group_idx": reference_group, "line_origin_world_ras_mm": [float(value) for value in plane["world"][:3, 0, 0]], "source_plane": {key: value for key, value in plane.items() if key != "world"}, "reconstruction_plane": {key: value for key, value in dense_plane.items() if key != "world"}})
        cmap, limits = colorbars().mapping[display_parameter], DISPLAY_RANGES_MS[display_parameter]
        filename = f"longitudinal_sax_centerline_{display_parameter}.png"
        _plot_scmr_longitudinal(output / filename, rows, title=f"{config['subject']} {config['condition']} {config['endpoint']} — SAX centre-line longitudinal {display_parameter}", limits=limits, cmap=cmap)
        records.append({"parameter": display_parameter, "file": filename, "layout": {"rows": 2, "columns": 5, "column_labels": list(scmr_longitudinal_panel_labels())}, "sax_reference_group_idx": reference_group, "sax_reference_group_selection": selection_rule, "cuts": cut_records, "sampling": {"source": "nearest-neighbor on source-stack grid", "reconstruction": "linear on shared approximately-1-mm physical plane", "same_physical_plane_for_source_reformat_and_all_reconstructions": True, "reconstruction_support": "in_volume AND finite only", "optional_reconstruction_display_roi": bool(section.get("reconstruction_display_roi", False))}, "no_auto_crop": True, "physical_aspect_ratio_preserved": True, "colorbars": {"shared_parameter": "dedicated_axis", "range_ms": list(limits), "colormap": "same_parameter_map_lut"}, "native_source": sax["provenance"]})
    for key in ("native_maps_path", "geometry_path", "manifest_path"):
        inputs[f"scmr_sax_{key}"] = _file_record(sax["provenance"][key])
    return records


def _render_native_stack(config: Mapping[str, Any], output: Path, inputs: dict[str, Any], native: Mapping[str, Any]) -> list[dict[str, Any]]:
    sparse_manifest = Path(native["sparse_manifest"])
    sparse = json.loads(sparse_manifest.read_text(encoding="utf-8"))
    stack = str(native["stack"])
    if str(native.get("mode", "single")) == "all_source_groups":
        selected = _load_all_verified_native_cache_groups(native["r0_native_maps_root"], sparse_manifest, stack)
        mode, display_groups = native_preview_groups(native, sparse, source_groups=[int(group) for group in selected["group_idx"]])
    else:
        mode, display_groups = native_preview_groups(native, sparse)
        selected = select_verified_native_cache_groups(native["r0_native_maps_root"], sparse_manifest, stack, _heldout_ids(sparse, stack))
    selected_indices = {int(group): index for index, group in enumerate(np.asarray(selected["group_idx"], int))}
    if any(group not in selected_indices for group in display_groups):
        raise ValueError(f"requested native groups are unavailable for {stack}")
    contour = None
    if native.get("myocardium_contour"):
        if mode != "single":
            raise ValueError("myocardium_contour is only supported for native.mode='single'")
        contour_path = Path(native["myocardium_contour"])
        contour = np.asarray(np.load(contour_path), bool)
        if contour.shape != np.asarray(selected["source_support"][selected_indices[display_groups[0]]], bool).shape:
            raise ValueError("myocardium contour must match the selected native plane")
        inputs["myocardium_contour"] = _file_record(contour_path)
    inputs.setdefault("sparse_manifest", _file_record(sparse_manifest))
    provenance = dict(selected["provenance"])
    for key in ("native_maps_path", "geometry_path", "manifest_path"):
        record = _file_record(provenance[key])
        inputs.setdefault(f"native_cache_{key}", record)
        inputs[f"native_cache_{stack}_{key}"] = record
    records: list[dict[str, Any]] = []
    group_statuses = native_group_statuses(display_groups, sparse, stack)
    methods = native_mosaic_methods(config["methods"])
    formal_mask_root = Path(config["native_mask_root"])
    for parameter, display_parameter in PARAMETERS.items():
        cmap, limits = colorbars().mapping[display_parameter], DISPLAY_RANGES_MS[display_parameter]
        method_sources: dict[str, tuple[str, Any]] = {}
        for method in methods:
            configured = config["methods"][method]
            if "fit_first_native_bundle" in configured:
                root = Path(configured["fit_first_native_bundle"])
                manifest_path = root / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("schema") != "fit_first_native_prediction_bundle/v1":
                    raise ValueError(f"{method}: Fit-first final-pose bundle schema mismatch")
                method_sources[method] = ("fit_first_native_bundle", root)
                inputs.setdefault(f"{method}_fit_first_native_bundle_manifest", _file_record(manifest_path))
            elif "native_bundle" in configured:
                root = Path(configured["native_bundle"])
                manifest_path = root / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("schema") != "direct_native_prediction_bundle/v1" or manifest.get("endpoint") != config["endpoint"]:
                    raise ValueError(f"{method}: direct-native bundle endpoint/schema mismatch")
                method_sources[method] = ("direct_native_bundle", root)
                inputs.setdefault(f"{method}_direct_native_bundle_manifest", _file_record(manifest_path))
            else:
                method_sources[method] = ("volume", _image(configured[parameter]))
        mosaic_rows: list[dict[str, Any]] = []
        shapes: dict[str, list[int]] = {}
        mask_artifacts: list[dict[str, Any]] = []
        for group in display_groups:
            index = selected_indices[group]
            affine = np.diag([-1.0, -1.0, 1.0, 1.0]) @ np.asarray(selected["affine_lps_rc"][index], float)
            source_support = np.asarray(selected["source_support"][index], bool)
            reference = np.asarray(selected[f"{parameter}_ms"][index], float)
            masks, mask_provenance = load_validated_current_native_mask_plane(formal_mask_root, parameter=parameter, stack=stack, original_group_idx=group, shape_rc=tuple(reference.shape), source_support=source_support)
            support = np.asarray(masks["primary_native_mask"], bool)
            mask_artifacts.append(mask_provenance)
            panels = [{"label": "Native reference", "values": reference, "support": support & np.isfinite(reference), "cmap": cmap, "range_ms": limits}]
            method_panels: list[tuple[str, np.ndarray, np.ndarray]] = []
            for method, (kind, source) in method_sources.items():
                if kind == "direct_native_bundle":
                    plane_path = source / "planes" / stack / f"group{group:03d}.npz"
                    with np.load(plane_path, allow_pickle=False) as bundle:
                        field = f"{parameter}_ms"
                        if field not in bundle.files:
                            raise ValueError(f"{plane_path}: missing {field}")
                        values = np.asarray(bundle[field], float)
                    if values.shape != source_support.shape:
                        raise ValueError(f"{plane_path}: native plane shape does not match current reference")
                    valid = np.isfinite(values)
                elif kind == "fit_first_native_bundle":
                    plane_path = source / "planes" / parameter / stack / f"group{group:03d}.npz"
                    with np.load(plane_path, allow_pickle=False) as bundle:
                        if not {"prediction_ms", "valid"}.issubset(bundle.files):
                            raise ValueError(f"{plane_path}: missing Fit-first prediction_ms/valid")
                        values, valid = np.asarray(bundle["prediction_ms"], float), np.asarray(bundle["valid"], bool)
                    if values.shape != source_support.shape or valid.shape != source_support.shape:
                        raise ValueError(f"{plane_path}: final-pose native plane shape does not match current reference")
                else:
                    image = source
                    values, valid = reproject_volume_to_native_plane(image.get_fdata(), image.affine, affine, source_support.shape)
                available = support & valid & np.isfinite(reference) & np.isfinite(values)
                method_panels.append((method, values, available))
            panels.extend({"label": method, "values": values, "support": available, "cmap": cmap, "range_ms": limits} for method, values, available in method_panels)
            panels.extend({"label": f"|{method} − Native|", "values": np.abs(values - reference), "support": available, "cmap": cmap, "range_ms": limits} for method, values, available in method_panels)
            spacing = np.asarray(selected["pixel_spacing_rc_mm"][index], float)
            extent = [0.0, float(source_support.shape[0] - 1) * spacing[0], 0.0, float(source_support.shape[1] - 1) * spacing[1]]
            mosaic_rows.append({"group_idx": group, "group_label": native_group_label(group, group_statuses[group]), "panels": panels, "extent": extent, "contour": contour})
            shapes[str(group)] = list(source_support.shape)
        filename = f"native_{stack}_group{display_groups[0]}_{display_parameter}.png" if mode == "single" else f"native_{stack}_{mode}_{display_parameter}.png"
        _plot_native_mosaic(output / filename, mosaic_rows, title=f"{config['subject']} {config['condition']} {config['endpoint']} — {stack} native {mode} {display_parameter}")
        inputs.setdefault("native_mask_manifest", _file_record(formal_mask_root / "manifest.json"))
        records.append({"parameter": display_parameter, "file": filename, "stack": stack, "mode": mode, "original_group_idx": display_groups[0] if mode == "single" else None, "original_group_ids": display_groups, "displayed_groups": [{"group_idx": group, "status": group_statuses[group], "label": native_group_label(group, group_statuses[group])} for group in display_groups], "native_reference": provenance, "primary_native_mask": "primary_native_mask", "mask_artifacts": mask_artifacts, "plane_shapes_rc": shapes, "display_support": "primary_native_mask AND finite(reference); per-method panels additionally require endpoint-valid AND finite(prediction)", "interpolation": {"map": "linear", "mask": "artifact-defined native-plane support"}, "no_auto_crop": True, "physical_aspect_ratio_preserved": True, "colorbars": {"shared_parameter": "dedicated_axis", "range_ms": list(limits), "colormap": "same_parameter_map_lut"}, "error_display": {"quantity": "absolute_error_ms", "colormap": "same_parameter_map_lut", "range_ms": list(limits), "no_residual_colorbar": True}})
    return records


def _render_native(config: Mapping[str, Any], output: Path, inputs: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for native in native_preview_stack_configs(config["native"]):
        records.extend(_render_native_stack(config, output, inputs, native))
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
        allowed = set(PARAMETERS) | {"native_bundle", "fit_first_native_bundle"}
        if not set(maps).issubset(allowed) or not set(maps):
            raise ValueError(f"{method} contains unsupported map/bundle keys")
        if not (set(PARAMETERS).issubset(maps) or "native_bundle" in maps or "fit_first_native_bundle" in maps):
            raise ValueError(f"{method} must provide t1/t2 maps or a native bundle")
        if method == "Fit-first" and "fit_first_native_bundle" not in maps:
            raise ValueError("formal Fit-first figures require fit_first_native_bundle; R0 volume reprojection is not a formal endpoint")
    if "native" not in config or "longitudinal" not in config or "l100_masks" not in config:
        raise ValueError("figure config requires longitudinal, native, and frozen L100-mask inputs")
    if config["native"] and not config.get("native_mask_root"):
        raise ValueError("formal native figures require immutable native_mask_root")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        inputs = {"config": _file_record(config_path)}
        longitudinal = _render_longitudinal(config, staging, inputs)
        scmr_longitudinal = _render_scmr_style_longitudinal(config, staging, inputs)
        native = _render_native(config, staging, inputs)
        manifest = {"schema": "phase4r_final_cyj_figures/v1", "subject": "CYJ", "condition": config.get("condition"), "endpoint": config["endpoint"], "methods": list(config["methods"]), "inputs": inputs, "longitudinal_figures": longitudinal, "scmr_style_longitudinal_figures": scmr_longitudinal, "native_heldout_figures": native, "shared_parameter_lut_ranges_ms": {key: list(value) for key, value in DISPLAY_RANGES_MS.items()}, "error_display_contract": "absolute_error_ms using same parameter map LUT/range; no magma residual scale; native mosaics provide one shared parameter colorbar", "code_git_sha": _git_sha()}
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
