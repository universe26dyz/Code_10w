"""Read-only final matched summaries and fixed-domain paper figures."""
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
from .visualization import DISPLAY_RANGES_MS, colorbars, final5_paired_display_support


PROPOSED_METHODS = ("Q002S640", "Q002S640_K256", "Q005S640_K4", "Q004S640")
METHOD_DISPLAY_ORDER = ("Fit-first", *PROPOSED_METHODS)
METHODS = PROPOSED_METHODS
ENDPOINTS = ("central", "psf128")
EXPECTED_FOLDS = {"L100": (0,), "L50": (0, 1), "L25": (0, 1, 2, 3)}
PARAMETERS = {"t1": "T1", "t2": "T2"}
NATIVE_MOSAIC_METHODS = METHOD_DISPLAY_ORDER


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _case(condition: str, *, profile_id: str | None = None, case_id: str | None = None) -> tuple[str, int, bool]:
    if profile_id is not None or case_id is not None:
        if not isinstance(profile_id, str) or not profile_id or not isinstance(case_id, str) or not case_id or condition != case_id:
            raise ValueError("custom profile results require matching explicit profile_id and case_id")
        try:
            fold = int(case_id.rsplit("_fold", 1)[1]) if "_fold" in case_id else 0
        except ValueError as error:
            raise ValueError(f"invalid custom case_id: {case_id}") from error
        return profile_id, fold, True
    try:
        level, fold = condition.split("_fold", 1)
        fold_int = int(fold)
    except ValueError as error:
        raise ValueError(f"invalid CYJ condition: {condition}") from error
    if level not in EXPECTED_FOLDS or fold_int not in EXPECTED_FOLDS[level]:
        raise ValueError(f"unsupported CYJ condition: {condition}")
    return level, fold_int, False


def _recovered_custom_identity(manifest: Mapping[str, Any], metrics: Mapping[str, Any], *, condition: str, subject: str) -> dict[str, Any] | None:
    """Recover identity only from a verified custom sparse manifest, never a path name."""
    if manifest.get("profile_id") is not None or manifest.get("case_id") is not None:
        return None
    if manifest.get("schema") != "final_stratified_native_bundle_metrics/v1" or metrics.get("schema") != "final_stratified_native_common_support_metrics/v1":
        return None
    try:
        level, _fold = condition.split("_fold", 1)
    except ValueError:
        level = condition
    if level in EXPECTED_FOLDS:
        return None
    record = manifest.get("sparse_manifest")
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str) or not isinstance(record.get("sha256"), str):
        raise ValueError("custom stratified metrics require a recorded sparse manifest path and SHA256")
    sparse_path = Path(record["path"])
    if not sparse_path.is_file():
        raise FileNotFoundError(sparse_path)
    if sha256(sparse_path) != record["sha256"]:
        raise ValueError("custom stratified metrics sparse manifest SHA256 mismatch")
    sparse = json.loads(sparse_path.read_text(encoding="utf-8"))
    if sparse.get("schema_version") != "phase4r_sparse_profile_v2":
        raise ValueError("custom stratified metrics sparse manifest schema mismatch")
    profile_id, case_id = sparse.get("profile_id"), sparse.get("case_id")
    fold_id, fold_count = sparse.get("fold_id"), sparse.get("fold_count")
    if sparse.get("subject") != subject or not isinstance(profile_id, str) or not profile_id or not isinstance(case_id, str) or case_id != condition:
        raise ValueError("custom stratified metrics sparse manifest subject/case identity mismatch")
    valid_fold = isinstance(fold_count, int) and fold_count >= 1 and (
        (fold_count == 1 and fold_id is None) or (isinstance(fold_id, int) and 0 <= fold_id < fold_count)
    )
    if not valid_fold:
        raise ValueError("custom stratified metrics sparse manifest fold identity mismatch")
    parsed_fold = _case(condition, profile_id=profile_id, case_id=case_id)[1]
    if fold_id is not None and parsed_fold != fold_id:
        raise ValueError("custom stratified metrics sparse manifest case/fold mismatch")
    return {"profile_id": profile_id, "case_id": case_id, "fold_id": fold_id, "fold_count": fold_count}


def _metric_records(root: str | Path, method: str, side: str, *, subject: str) -> list[dict[str, Any]]:
    root = Path(root)
    if method not in METHODS:
        raise ValueError(f"unsupported Proposed method: {method}")
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for manifest_path in sorted(root.rglob("manifest.json")):
        metrics_path = manifest_path.with_name("aggregate_metrics.json")
        if not metrics_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("subject") != subject:
            raise ValueError(f"final subject summary rejects a mismatched result: {manifest_path}")
        endpoint, condition = manifest.get("endpoint"), manifest.get("condition")
        if endpoint not in ENDPOINTS or not isinstance(condition, str):
            raise ValueError(f"missing endpoint/condition in {manifest_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        schema = str(metrics.get("schema", "legacy_primary"))
        identity = (endpoint, condition, schema)
        if identity in seen:
            raise ValueError(f"duplicate {method} result for {endpoint}/{condition}/{schema}")
        seen.add(identity)
        recovered = _recovered_custom_identity(manifest, metrics, condition=condition, subject=subject)
        profile_id = manifest.get("profile_id") if recovered is None else recovered["profile_id"]
        case_id = manifest.get("case_id") if recovered is None else recovered["case_id"]
        fold_id = manifest.get("fold_id") if recovered is None else recovered["fold_id"]
        fold_count = manifest.get("fold_count") if recovered is None else recovered["fold_count"]
        level, fold, custom_profile = _case(condition, profile_id=profile_id, case_id=case_id)
        for parameter, display_parameter in PARAMETERS.items():
            entries: list[tuple[str, str, Mapping[str, Any]]] = []
            if schema == "final_stratified_native_common_support_metrics/v1":
                for region, region_values in metrics.get("regions", {}).get(parameter, {}).items():
                    if region_values.get("status") == "COMPLETE" and side in region_values:
                        entries.append(("STRATIFIED_ORIGINAL_NATIVE", region, region_values[side]))
            else:
                try:
                    values = metrics["primary_aggregate"][parameter][side]
                except KeyError:
                    try:
                        values = metrics["regions"][parameter]["all_lax"]["primary_native_mask"][side]
                    except KeyError as error:
                        try:
                            values = metrics["regions"][parameter]["all_lax"]["l100_full"][side]
                        except KeyError:
                            try:
                                values = metrics["regions"][parameter]["all_lax"][side]
                            except KeyError:
                                raise ValueError(f"{metrics_path} lacks formal primary metrics for {parameter}/{side}") from error
                entries.append(("PRIMARY_FINAL", "primary_native_mask", values))
            for metric_family, region, values in entries:
                rows.append({
                    "method": method if side == "proposed" else "Fit-first", "endpoint": endpoint, "condition": condition,
                    "level": level, "fold": fold, "parameter": display_parameter, "metric_family": metric_family,
                    "profile_id": profile_id if custom_profile else None, "case_id": case_id if custom_profile else None, "custom_profile": custom_profile,
                    "fold_id": fold if custom_profile and fold_id is None else fold_id, "fold_count": fold_count if custom_profile else None,
                    "region": region, "rmse": values["rmse"], "mae": values["mae"], "bias": values["bias"],
                    "voxel_count": values.get("voxel_count", values.get("support_voxels")),
                    "coverage_fraction": values.get("coverage_fraction"), "source_manifest": str(manifest_path.resolve()),
                    "source_metrics": str(metrics_path.resolve()), "source_method": method,
                })
    if not rows:
        raise ValueError(f"no formal aggregate results under {root}")
    return rows


def _sample_sd(values: list[float]) -> float | None:
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def _format_mean_sd(mean: float, sample_sd: float | None) -> str:
    return f"{mean:.4g}" if sample_sd is None else f"{mean:.4g} ± {sample_sd:.3g}"


def _summaries(rows: list[dict[str, Any]], fit_rows: dict[tuple[str, str, str, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str, str | None], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["endpoint"], row["level"], row["parameter"], row["metric_family"], row["region"], row["case_id"] if row.get("custom_profile") else None)].append(row)
    output: list[dict[str, Any]] = []
    for (method, endpoint, level, parameter, metric_family, region, case_id), values in sorted(grouped.items()):
        values.sort(key=lambda row: row["fold"])
        observed = [int(row["fold"]) for row in values]
        custom_profile = all(bool(row.get("custom_profile")) for row in values)
        expected = observed if custom_profile else list(EXPECTED_FOLDS[level])
        missing = sorted(set(expected).difference(observed))
        metric = lambda name: [float(row[name]) for row in values]
        row: dict[str, Any] = {
            "method": method, "endpoint": endpoint, "level": level, "parameter": parameter,
            "metric_family": metric_family, "region": region,
            "expected_folds": expected, "observed_folds": observed, "missing_folds": missing,
            "status": "COMPLETE_EXPLICIT_CASE" if custom_profile else ("COMPLETE" if not missing else "INCOMPLETE_MISSING_FOLDS"),
        }
        if custom_profile:
            row["profile_id"] = values[0]["profile_id"]
            row["case_id"] = case_id
            row["fold_id"] = values[0]["fold_id"]
            row["fold_count"] = values[0]["fold_count"]
        for name in ("rmse", "mae", "bias"):
            numbers = metric(name)
            row[f"{name}_mean"] = float(np.mean(numbers))
            row[f"{name}_sample_sd"] = _sample_sd(numbers)
        if method != "Fit-first":
            paired = [fit_rows.get((endpoint, value["condition"], parameter, metric_family, region)) for value in values]
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


def _profile_summaries(case_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate explicit custom cases only after preserving case-level rows."""
    grouped: dict[tuple[str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in case_summaries:
        if row.get("profile_id"):
            grouped[(row["method"], row["endpoint"], row["profile_id"], row["parameter"], row["metric_family"], row["region"])].append(row)
    output: list[dict[str, Any]] = []
    for (method, endpoint, profile_id, parameter, metric_family, region), values in sorted(grouped.items()):
        result = {"method": method, "endpoint": endpoint, "profile_id": profile_id, "parameter": parameter, "metric_family": metric_family, "region": region, "case_ids": sorted(str(item["case_id"]) for item in values), "case_count": len(values), "status": "PROFILE_LEVEL_EXPLICIT_CASES"}
        for metric in ("rmse", "mae", "bias"):
            numbers = [float(item[f"{metric}_mean"]) for item in values]
            result[f"{metric}_mean_across_cases"] = float(np.mean(numbers))
            result[f"{metric}_sample_sd_across_cases"] = _sample_sd(numbers)
        output.append(result)
    return output


def _robustness(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["method"], row["endpoint"], row["parameter"], row["metric_family"], row["region"], row["level"]): row for row in summaries if not row.get("profile_id")}
    rows: list[dict[str, Any]] = []
    identities = sorted({key[:5] for key in by_key})
    for method, endpoint, parameter, metric_family, region in identities:
        levels = {level: by_key.get((method, endpoint, parameter, metric_family, region, level)) for level in EXPECTED_FOLDS}
        result = {"method": method, "endpoint": endpoint, "parameter": parameter, "metric_family": metric_family, "region": region}
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


def _paper_table(rows: list[dict[str, Any]], *, subject: str) -> tuple[list[dict[str, Any]], str]:
    output: list[dict[str, Any]] = []
    for row in rows:
        for metric in ("rmse", "mae"):
            value, sd = row[f"{metric}_mean"], row[f"{metric}_sample_sd"]
            output.append({"method": row["method"], "endpoint": row["endpoint"], "level_or_profile": row["level"], "parameter": row["parameter"], "metric_family": row["metric_family"], "region": row["region"], "metric": metric.upper(), "value": f"{value:.4g}" if sd is None else f"{value:.4g} ± {sd:.3g}"})
    has_custom = any(row.get("profile_id") for row in rows)
    note = "L100 is a single value and L50/L25 use observed-fold mean ± sample SD."
    if has_custom:
        note += " Explicit custom-profile cases are preserved case-by-case; profile-level summaries aggregate those cases only after case-level results are retained."
    header = f"# {subject} final matched table\n\nAll values use frozen `primary_native_mask`; {note}\n\n"
    lines = [header, "| Method | Endpoint | Level / Profile | Parameter | Metric family | Region | Metric | Value |", "|---|---|---|---|---|---|---|---:|"]
    lines.extend(f"| {row['method']} | {row['endpoint']} | {row['level_or_profile']} | {row['parameter']} | {row['metric_family']} | {row['region']} | {row['metric']} | {row['value']} |" for row in output)
    return output, "\n".join(lines) + "\n"


def _summary_markdown(rows: list[dict[str, Any]], robustness: list[dict[str, Any]], *, subject: str) -> str:
    has_custom = any(row.get("profile_id") for row in rows)
    note = "L100 is a single value; L50/L25 are observed-fold mean ± sample SD. Missing folds are not imputed."
    if has_custom:
        note += " Explicit custom-profile cases are preserved case-by-case; profile-level summaries aggregate explicit cases only after case-level results are retained."
    lines = [
        f"# {subject} final matched summary", "",
        f"All values use the frozen current-native `primary_native_mask`. {note}", "",
        "| Method | Endpoint | Level / Profile | Parameter | Metric family | Region | RMSE | MAE | Bias | ΔRMSE vs Fit-first | ΔMAE vs Fit-first | RMSE wins | MAE wins | Status |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        delta_rmse = row.get("delta_rmse_vs_fit_first_mean")
        delta_mae = row.get("delta_mae_vs_fit_first_mean")
        wins = row.get("proposed_rmse_win_count")
        total = row.get("win_total_folds")
        rmse_wins = "—" if wins is None else f"{wins}/{total}"
        mae_wins = "—" if wins is None else f"{row['proposed_mae_win_count']}/{total}"
        lines.append(
            f"| {row['method']} | {row['endpoint']} | {row['level']} | {row['parameter']} | {row['metric_family']} | {row['region']} | "
            f"{_format_mean_sd(row['rmse_mean'], row['rmse_sample_sd'])} | {_format_mean_sd(row['mae_mean'], row['mae_sample_sd'])} | "
            f"{_format_mean_sd(row['bias_mean'], row['bias_sample_sd'])} | "
            f"{'—' if delta_rmse is None else f'{delta_rmse:.4g}'} | {'—' if delta_mae is None else f'{delta_mae:.4g}'} | "
            f"{rmse_wins} | {mae_wins} | {row['status']} |"
        )
    lines.extend(("", "## Sparsity robustness", "", "| Method | Endpoint | Parameter | Metric family | Region | RMSE L100→L50 | RMSE L50→L25 | RMSE L100→L25 | MAE L100→L50 | MAE L50→L25 | MAE L100→L25 |", "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|"))
    for row in robustness:
        values = [row[f"{metric}_{first}_to_{second}_change"] for metric in ("rmse", "mae") for first, second in (("L100", "L50"), ("L50", "L25"), ("L100", "L25"))]
        lines.append(f"| {row['method']} | {row['endpoint']} | {row['parameter']} | {row['metric_family']} | {row['region']} | " + " | ".join("—" if value is None else f"{value:.4g}" for value in values) + " |")
    return "\n".join(lines) + "\n"


def aggregate_final_subject(subject: str, method_roots: Mapping[str, str | Path], output: str | Path, *, fit_first_source_method: str) -> Path:
    """Aggregate one explicit subject without fabricating methods or folds."""
    if not isinstance(subject, str) or not subject:
        raise ValueError("subject must be a non-empty explicit identity")
    if fit_first_source_method not in method_roots:
        raise ValueError("fit_first_source_method must be one explicitly supplied Proposed method root")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    proposed_rows = [row for method, root in sorted(method_roots.items()) for row in _metric_records(root, method, "proposed", subject=subject)]
    fit_rows_list = _metric_records(method_roots[fit_first_source_method], fit_first_source_method, "fit_first", subject=subject)
    fit_index = {(row["endpoint"], row["condition"], row["parameter"], row["metric_family"], row["region"]): row for row in fit_rows_list}
    all_rows = proposed_rows + fit_rows_list
    summaries = _summaries(all_rows, fit_index)
    profile_summaries = _profile_summaries(summaries)
    robustness = _robustness(summaries)
    paper_rows, paper_markdown = _paper_table(summaries, subject=subject)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        _write_csv(staging / "summary.csv", all_rows + [{"row_type": "level_summary", **row} for row in summaries] + [{"row_type": "robustness", **row} for row in robustness])
        _write_csv(staging / "paper_table.csv", paper_rows)
        (staging / "paper_table.md").write_text(paper_markdown, encoding="utf-8")
        (staging / "summary.md").write_text(_summary_markdown(summaries, robustness, subject=subject), encoding="utf-8")
        write_immutable_json(staging / "summary.json", {"schema": "phase4r_final_subject_summary/v4", "subject": subject, "primary_region": "primary_native_mask", "fit_first_source_method": fit_first_source_method, "method_roots": {key: str(Path(value).resolve()) for key, value in method_roots.items()}, "fold_rows": all_rows, "level_summaries": summaries, "profile_summaries": profile_summaries, "sparsity_robustness": robustness})
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True); raise
    return output


def aggregate_final_cyj(method_roots: Mapping[str, str | Path], output: str | Path, *, fit_first_source_method: str) -> Path:
    """Compatibility wrapper for existing CYJ command lines and outputs."""
    return aggregate_final_subject("CYJ", method_roots, output, fit_first_source_method=fit_first_source_method)


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
    unknown = set(methods).difference(METHOD_DISPLAY_ORDER)
    if unknown:
        raise ValueError(f"unsupported reporting method(s): {', '.join(sorted(unknown))}")
    return tuple(method for method in METHOD_DISPLAY_ORDER if method in methods)


def _load_mask_spec(config: Mapping[str, Any], parameter: str) -> Path:
    value = config["l100_masks"][parameter.lower()]
    return Path(value["path"] if isinstance(value, Mapping) else value)


def _image(path: str | Path) -> Any:
    import nibabel as nib
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return nib.load(str(path))


def _component_path(component_directory: Path, *, row: int, column: int, label: str) -> Path:
    safe_label = "".join(character.lower() if character.isalnum() else "_" for character in label).strip("_")
    return component_directory / f"r{row:02d}_c{column:02d}_{safe_label}.png"


def _component_record_path(component_directory: Path, path: Path) -> str:
    """Return an output-root-relative component path for the fixed components/<stem>/ layout."""
    return path.relative_to(component_directory.parents[1]).as_posix()


def _save_text_free_image_component(path: Path, shown: np.ndarray, *, extent: list[float], cmap: Any, limits: tuple[float, float], contour: np.ndarray | None = None, centerline: tuple[str, float] | None = None) -> None:
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(3.0, 3.0))
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.imshow(shown.T, origin="lower", extent=extent, cmap=cmap, vmin=limits[0], vmax=limits[1], interpolation="none", aspect="equal")
    if contour is not None and np.any(contour):
        axis.contour(contour.T.astype(float), levels=[.5], colors="white", linewidths=.55, origin="lower", extent=extent)
    if centerline is not None:
        direction, coordinate = centerline
        (axis.axvline if direction == "vertical" else axis.axhline)(coordinate, color="cyan", linewidth=1.2)
    axis.set_axis_off()
    figure.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0)
    plt.close(figure)


def _save_text_free_colorbar(path: Path, *, cmap: Any, limits: tuple[float, float]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize

    figure = plt.figure(figsize=(.24, 3.0))
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0])
    colorbar = ColorbarBase(axis, cmap=cmap, norm=Normalize(vmin=limits[0], vmax=limits[1]), orientation="vertical")
    colorbar.set_ticks([])
    colorbar.set_label("")
    axis.set_axis_off()
    figure.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0)
    plt.close(figure)


def _empty_component_assets() -> dict[str, Any]:
    return {"components": [], "colorbar": None}


def _plot_panels(path: Path, panels: list[dict[str, Any]], *, extent: list[float], title: str, contour: np.ndarray | None = None, component_directory: Path | None = None) -> dict[str, Any]:
    """Render full supplied plane bounds; only masks, never crop, hide background."""
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(3.0 * len(panels) + .65, 3.3))
    grid = figure.add_gridspec(1, len(panels) + 1, width_ratios=[1.0] * len(panels) + [.055], wspace=.12)
    axes = [figure.add_subplot(grid[0, index]) for index in range(len(panels))]
    colorbar_axis = figure.add_subplot(grid[0, -1])
    assets = _empty_component_assets()
    if component_directory is not None:
        component_directory.mkdir(parents=True, exist_ok=True)
    for column, (axis, panel) in enumerate(zip(axes, panels)):
        shown = np.ma.masked_invalid(np.where(panel["support"], panel["values"], np.nan))
        image = axis.imshow(shown.T, origin="lower", extent=extent, cmap=panel["cmap"], vmin=panel["range_ms"][0], vmax=panel["range_ms"][1], interpolation="none", aspect="equal")
        if contour is not None and np.any(contour):
            axis.contour(contour.T.astype(float), levels=[0.5], colors="white", linewidths=.55, origin="lower", extent=extent)
        axis.set_title(panel["label"], fontsize=8)
        axis.set_xticks([]); axis.set_yticks([])
        if component_directory is not None:
            component_path = _component_path(component_directory, row=0, column=column, label=str(panel["label"]))
            _save_text_free_image_component(component_path, shown, extent=extent, cmap=panel["cmap"], limits=panel["range_ms"], contour=contour)
            assets["components"].append({"file": _component_record_path(component_directory, component_path), "kind": "panel", "row": 0, "column": column, "label": panel["label"], "cmap": panel["cmap"].name, "range_ms": list(panel["range_ms"]), "extent_mm": list(extent), "contour_overlay": bool(contour is not None and np.any(contour)), "text_free": True, "derived_from_same_render_data": True})
    figure.suptitle(title, fontsize=10)
    figure.colorbar(image, cax=colorbar_axis, label="ms; shared parameter map scale")
    figure.subplots_adjust(top=.83, bottom=.04, left=.02, right=.98)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    if component_directory is not None:
        colorbar_path = component_directory / "colorbar.png"
        _save_text_free_colorbar(colorbar_path, cmap=panels[-1]["cmap"], limits=panels[-1]["range_ms"])
        assets["colorbar"] = {"file": _component_record_path(component_directory, colorbar_path), "cmap": panels[-1]["cmap"].name, "range_ms": list(panels[-1]["range_ms"]), "orientation": "vertical", "text_free": True, "derived_from_same_render_data": True}
    return assets


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


def _plot_native_mosaic(path: Path, rows: list[Any], *, title: str, component_directory: Path | None = None) -> dict[str, Any]:
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
    assets = _empty_component_assets()
    if component_directory is not None:
        component_directory.mkdir(parents=True, exist_ok=True)
    for row_index, (group, group_label, panels, extent, contour) in enumerate(normalized):
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
            panel_kind = str(panel.get("kind", "map_ms"))
            if panel_kind != "absolute_error_ms":
                map_handle = image
            if component_directory is not None:
                component_path = _component_path(component_directory, row=row_index, column=column, label=str(panel["label"]))
                _save_text_free_image_component(component_path, shown, extent=extent, cmap=panel["cmap"], limits=panel["range_ms"], contour=contour)
                component = {"file": _component_record_path(component_directory, component_path), "kind": panel_kind, "row": row_index, "column": column, "original_group_idx": group, "group_label": group_label, "label": panel["label"], "displayed_quantity": panel.get("displayed_quantity", "quantitative_map_ms"), "cmap": panel["cmap"].name, "range_ms": list(panel["range_ms"]), "extent_mm": list(extent), "contour_overlay": bool(contour is not None and np.any(contour)), "text_free": True, "derived_from_same_render_data": True}
                if panel.get("method") is not None:
                    component["method"] = panel["method"]
                assets["components"].append(component)
    map_colorbar_axis = figure.add_subplot(grid[:, panel_count])
    if map_handle is None:
        raise ValueError("native mosaic needs a quantitative-map panel")
    figure.colorbar(map_handle, cax=map_colorbar_axis, label="parameter (ms); shared map/error scale")
    figure.suptitle(title, fontsize=10)
    figure.subplots_adjust(top=.9, bottom=.03, left=.03, right=.98)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    if component_directory is not None:
        colorbar_path = component_directory / "colorbar.png"
        _save_text_free_colorbar(colorbar_path, cmap=map_handle.get_cmap(), limits=(map_handle.norm.vmin, map_handle.norm.vmax))
        assets["colorbar"] = {"file": _component_record_path(component_directory, colorbar_path), "cmap": map_handle.get_cmap().name, "range_ms": [map_handle.norm.vmin, map_handle.norm.vmax], "orientation": "vertical", "text_free": True, "derived_from_same_render_data": True}
    return assets


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
    if stack == "sax" or sparse.get("condition") == "L100":
        return {int(group): "observed" for group in groups}
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


def scmr_longitudinal_panel_labels(methods: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    comparison = ("Fit-first", "Q002S640", "Q004S640") if methods is None else native_mosaic_methods(methods)
    return ("SAX source + centerline overlay", "Original SAX-stack longitudinal reformatting", *comparison)


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
            item = dict(base)
            item["stack"] = entry
            result.append(item)
        elif isinstance(entry, Mapping):
            item = dict(base)
            item.update(entry)
            result.append(item)
        else:
            raise ValueError("native.stacks entries must be stack names or mappings")
    stacks = [str(item.get("stack", "")) for item in result]
    if len(stacks) != len(set(stacks)) or any(not stack for stack in stacks):
        raise ValueError("native.stacks must name each stack exactly once")
    return result


def _file_record(path: str | Path) -> dict[str, str]:
    item = Path(path)
    return {"path": str(item.resolve()), "sha256": sha256(item)}


def _final5_case_record(config: Mapping[str, Any]) -> tuple[Path, Any]:
    """Recover the exact Final5 CaseRecord from the explicit config source."""
    final5 = config.get("final5")
    if not isinstance(final5, Mapping):
        raise ValueError("Final5 validation requires an explicit final5 config section")
    root_value, sparse_value = final5.get("root"), final5.get("sparse_manifest")
    if not isinstance(root_value, str) or not isinstance(sparse_value, str):
        raise ValueError("Final5 config requires explicit root and sparse_manifest paths")
    sparse_path = Path(sparse_value)
    sparse = json.loads(sparse_path.read_text(encoding="utf-8"))
    if not isinstance(sparse, Mapping):
        raise ValueError("Final5 sparse manifest must be a JSON object")
    subject, case_id, profile_id = config.get("subject"), config.get("case_id"), config.get("profile_id")
    if not all(isinstance(value, str) and value for value in (subject, case_id, profile_id)):
        raise ValueError("Final5 config requires subject, case_id, and profile_id")
    if (sparse.get("subject") not in (None, subject) or sparse.get("case_id") != case_id or sparse.get("profile_id") != profile_id):
        raise ValueError("Final5 config and sparse manifest identity mismatch")
    generator_git_sha, prepared_root = sparse.get("generator_git_sha", sparse.get("git_sha")), sparse.get("prepared_root")
    fold_id, fold_count = sparse.get("fold_id"), sparse.get("fold_count")
    if not isinstance(generator_git_sha, str) or not generator_git_sha or not isinstance(prepared_root, str) or not prepared_root:
        raise ValueError("Final5 sparse manifest requires generator_git_sha and prepared_root")
    if fold_id is not None and not isinstance(fold_id, int) or fold_count is not None and not isinstance(fold_count, int):
        raise ValueError("Final5 sparse manifest fold identity is invalid")
    from .final5_batch_runner import CaseRecord
    return Path(root_value), CaseRecord(subject, case_id, profile_id, fold_id, fold_count, sparse_path, sha256(sparse_path), generator_git_sha, Path(prepared_root))


def _validate_final5_native_bundle(config: Mapping[str, Any], *, kind: str, path: Path) -> None:
    """Delegate Final5 bundle admission to the production provenance validators."""
    if "final5" not in config:
        return
    root, record = _final5_case_record(config)
    from .final5_batch_runner import validate_fit_first_native_bundle, validate_proposed_native_bundle
    result = validate_fit_first_native_bundle(path, record, root) if kind == "fit_first" else validate_proposed_native_bundle(path, record, root)
    if not result.valid:
        raise ValueError(f"Final5 {kind} native bundle provenance rejected: {result.reason}")


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
        sampled_methods: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        method_inputs: dict[str, Any] = {}
        for method in methods:
            map_path = Path(config["methods"][method][parameter])
            image = _image(map_path)
            values, valid = _world_sample(image.get_fdata(), image.affine, plane["world"], order=1)
            sampled_methods[method] = (np.asarray(values), np.asarray(valid, bool))
            method_inputs[method] = _file_record(map_path)
        shared_support = support.copy()
        for values, valid in sampled_methods.values():
            shared_support &= valid & np.isfinite(values)
        panels = [{"label": method, "values": values, "support": shared_support, "cmap": cmap, "range_ms": limits} for method, (values, _valid) in sampled_methods.items()]
        filename = f"longitudinal_{display_parameter}.png"
        assets = _plot_panels(output / filename, panels, extent=plane["extent_mm"], title=f"{config['subject']} {config['condition']} {config['endpoint']} — frozen L100 longitudinal {display_parameter}", component_directory=output / "components" / Path(filename).stem)
        inputs[f"l100_mask_{parameter}"] = _file_record(mask_path)
        inputs.update({f"{method}_{parameter}": value for method, value in method_inputs.items()})
        records.append({"parameter": display_parameter, "file": filename, "component_directory": f"components/{Path(filename).stem}", "components": [{**component, "parameter": display_parameter, "displayed_quantity": "parameter_map_ms", "method": component["label"], "support_semantics": "frozen_L100_display_mask AND in_volume(all configured comparison methods) AND finite(all configured comparison methods)"} for component in assets["components"]], "colorbar_component": assets["colorbar"], "frozen_l100_source": _file_record(mask_path), "plane": {key: value for key, value in plane.items() if key != "world"}, "plane_selection": selection_mode, "world_coordinate_sampling": True, "comparison_support_contract": "frozen_L100_display_mask AND in_volume(all configured comparison methods) AND finite(all configured comparison methods)", "common_support_voxels": int(shared_support.sum()), "all_methods_share_identical_support": True, "interpolation": {"map": "linear", "mask": "nearest-neighbor"}, "no_auto_crop": True, "lut_range_ms": list(limits)})
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


def _plot_scmr_longitudinal(path: Path, rows: list[dict[str, Any]], *, title: str, limits: tuple[float, float], cmap: Any, column_labels: tuple[str, ...], component_directory: Path | None = None) -> dict[str, Any]:
    """Two physical SAX centre-line cuts with one parameter colorbar."""
    import matplotlib.pyplot as plt

    comparison_columns = len(column_labels)
    figure = plt.figure(figsize=(3.0 * comparison_columns + .65, 6.1))
    grid = figure.add_gridspec(2, comparison_columns + 1, width_ratios=[1.0] * comparison_columns + [.06], wspace=.12, hspace=.20)
    handle = None
    assets = _empty_component_assets()
    if component_directory is not None:
        component_directory.mkdir(parents=True, exist_ok=True)
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
        source_axis.set_title(column_labels[0], fontsize=8)
        source_axis.set_xticks([]); source_axis.set_yticks([])
        if component_directory is not None:
            component_path = _component_path(component_directory, row=row_index, column=0, label=column_labels[0])
            centerline = ("vertical", row["source_line_coordinate_mm"]) if row["cut"] == "center_row" else ("horizontal", row["source_line_coordinate_mm"])
            _save_text_free_image_component(component_path, shown, extent=source["extent_mm"], cmap=cmap, limits=limits, centerline=centerline)
            assets["components"].append({"file": _component_record_path(component_directory, component_path), "kind": "source_with_centerline", "row": row_index, "column": 0, "cut": row["cut"], "label": column_labels[0], "cmap": cmap.name, "range_ms": list(limits), "extent_mm": list(source["extent_mm"]), "centerline_overlay": True, "text_free": True, "derived_from_same_render_data": True})
        for column, panel in enumerate(row["longitudinal"], start=1):
            axis = figure.add_subplot(grid[row_index, column])
            shown = np.ma.masked_invalid(np.where(panel["support"], panel["values"], np.nan))
            handle = axis.imshow(shown.T, origin="lower", extent=row["extent_mm"], cmap=cmap, vmin=limits[0], vmax=limits[1], interpolation="none", aspect="equal")
            if row_index == 0:
                axis.set_title(column_labels[column], fontsize=8)
            axis.set_xticks([]); axis.set_yticks([])
            if component_directory is not None:
                component_path = _component_path(component_directory, row=row_index, column=column, label=column_labels[column])
                _save_text_free_image_component(component_path, shown, extent=row["extent_mm"], cmap=cmap, limits=limits)
                assets["components"].append({"file": _component_record_path(component_directory, component_path), "kind": "longitudinal_reformat", "row": row_index, "column": column, "cut": row["cut"], "label": column_labels[column], "method": panel.get("label"), "cmap": cmap.name, "range_ms": list(limits), "extent_mm": list(row["extent_mm"]), "text_free": True, "derived_from_same_render_data": True})
    colorbar_axis = figure.add_subplot(grid[:, -1])
    figure.colorbar(handle, cax=colorbar_axis, label="ms; shared parameter map scale")
    figure.suptitle(title, fontsize=10)
    figure.subplots_adjust(top=.91, bottom=.04, left=.03, right=.98)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    if component_directory is not None:
        colorbar_path = component_directory / "colorbar.png"
        _save_text_free_colorbar(colorbar_path, cmap=cmap, limits=limits)
        assets["colorbar"] = {"file": _component_record_path(component_directory, colorbar_path), "cmap": cmap.name, "range_ms": list(limits), "orientation": "vertical", "text_free": True, "derived_from_same_render_data": True}
    return assets


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
    methods = native_mosaic_methods(config["methods"])
    if "Fit-first" not in methods or len(methods) < 2:
        raise ValueError("scmr_longitudinal requires Fit-first plus at least one configured Proposed method")
    column_labels = scmr_longitudinal_panel_labels(config["methods"])
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
            native_values, native_valid = _world_sample(source_stack, source_affine, dense_plane["world"], order=0)
            native_support, native_support_valid = _world_sample(support_stack.astype(np.uint8), source_affine, dense_plane["world"], order=0)
            shared_support = native_valid & native_support_valid & (native_support > .5) & np.isfinite(native_values)
            sampled_methods: dict[str, np.ndarray] = {}
            for method, image in method_images.items():
                values, valid = _world_sample(image.get_fdata(), image.affine, dense_plane["world"], order=1)
                values = np.asarray(values)
                sampled_methods[method] = values
                shared_support &= np.asarray(valid, bool) & np.isfinite(values)
            longitudinal = [{"label": column_labels[1], "values": native_values, "support": shared_support}]
            longitudinal.extend({"label": method, "values": values, "support": shared_support} for method, values in sampled_methods.items())
            line_coordinate = float(index * source_spacing[0 if cut == "center_row" else 1])
            rows.append({"cut": cut, "source": {"values": np.asarray(sax[f"{parameter}_ms"][reference_index], float), "support": np.asarray(sax["source_support"][reference_index], bool), "extent_mm": source_extent}, "source_line_coordinate_mm": line_coordinate, "longitudinal": longitudinal, "extent_mm": plane["extent_mm"]})
            cut_records.append({"cut": cut, "source_line_index": index, "source_line_coordinate_mm": line_coordinate, "source_plane_group_idx": reference_group, "line_origin_world_ras_mm": [float(value) for value in plane["world"][:3, 0, 0]], "source_plane": {key: value for key, value in plane.items() if key != "world"}, "reconstruction_plane": {key: value for key, value in dense_plane.items() if key != "world"}, "comparison_support_contract": "source comparison domain AND in_volume(all configured comparison methods) AND finite(all configured comparison methods)", "common_support_voxels": int(shared_support.sum()), "all_methods_share_identical_support": True})
        cmap, limits = colorbars().mapping[display_parameter], DISPLAY_RANGES_MS[display_parameter]
        filename = f"longitudinal_sax_centerline_{display_parameter}.png"
        assets = _plot_scmr_longitudinal(output / filename, rows, title=f"{config['subject']} {config['condition']} {config['endpoint']} — SAX centre-line longitudinal {display_parameter}", limits=limits, cmap=cmap, column_labels=column_labels, component_directory=output / "components" / Path(filename).stem)
        records.append({"parameter": display_parameter, "file": filename, "component_directory": f"components/{Path(filename).stem}", "components": [{**component, "parameter": display_parameter, "displayed_quantity": "parameter_map_ms", "support_semantics": "source comparison domain AND in_volume(all configured comparison methods) AND finite(all configured comparison methods)"} for component in assets["components"]], "colorbar_component": assets["colorbar"], "layout": {"rows": 2, "columns": len(column_labels), "column_labels": list(column_labels), "configured_comparison_methods": list(methods)}, "sax_reference_group_idx": reference_group, "sax_reference_group_selection": selection_rule, "cuts": cut_records, "sampling": {"source": "nearest-neighbor on source-stack grid", "reconstruction": "linear on shared approximately-1-mm physical plane", "same_physical_plane_for_source_reformat_and_all_reconstructions": True, "reconstruction_support": "source comparison domain AND in_volume(all configured comparison methods) AND finite(all configured comparison methods)", "optional_reconstruction_display_roi": bool(section.get("reconstruction_display_roi", False))}, "all_methods_share_identical_support": True, "no_auto_crop": True, "physical_aspect_ratio_preserved": True, "colorbars": {"shared_parameter": "dedicated_axis", "range_ms": list(limits), "colormap": "same_parameter_map_lut"}, "native_source": sax["provenance"]})
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
                if manifest.get("schema") not in ("fit_first_native_prediction_bundle/v2", "fit_first_native_prediction_bundle/v4", "fit_first_native_prediction_bundle/v6") or manifest.get("endpoint") != config["endpoint"]:
                    raise ValueError(f"{method}: Fit-first final-pose bundle schema mismatch")
                if manifest.get("schema") == "fit_first_native_prediction_bundle/v6":
                    _validate_final5_native_bundle(config, kind="fit_first", path=root)
                method_sources[method] = ("fit_first_native_bundle", root)
                inputs.setdefault(f"{method}_fit_first_native_bundle_manifest", _file_record(manifest_path))
            elif "native_bundle" in configured:
                root = Path(configured["native_bundle"])
                manifest_path = root / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("schema") != "direct_native_prediction_bundle/v1" or manifest.get("endpoint") != config["endpoint"]:
                    raise ValueError(f"{method}: direct-native bundle endpoint/schema mismatch")
                if "final5" in config:
                    _validate_final5_native_bundle(config, kind="proposed", path=root)
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
            panels: list[dict[str, Any]] = []
            method_values: dict[str, np.ndarray] = {}
            method_valid: dict[str, np.ndarray] = {}
            fit_first_volume_geometry_valid: np.ndarray | None = None
            fit_first_query_effective_mask: np.ndarray | None = None
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
                        if "final5" in config:
                            if not {"volume_geometry_valid", "query_effective_mask"}.issubset(bundle.files):
                                raise ValueError(f"{plane_path}: Final5 v6 bundle is missing output-sampling validity fields")
                            fit_first_volume_geometry_valid = np.asarray(bundle["volume_geometry_valid"], bool)
                            fit_first_query_effective_mask = np.asarray(bundle["query_effective_mask"], bool)
                    if values.shape != source_support.shape or valid.shape != source_support.shape:
                        raise ValueError(f"{plane_path}: final-pose native plane shape does not match current reference")
                    if fit_first_volume_geometry_valid is not None and (fit_first_volume_geometry_valid.shape != source_support.shape or fit_first_query_effective_mask is None or fit_first_query_effective_mask.shape != source_support.shape):
                        raise ValueError(f"{plane_path}: Final5 v6 validity shape does not match current reference")
                    if "final5" in config and not np.array_equal(valid, fit_first_volume_geometry_valid & fit_first_query_effective_mask & np.isfinite(values)):
                        raise ValueError(f"{plane_path}: Final5 v6 valid must equal volume_geometry_valid AND query_effective_mask AND finite(prediction)")
                else:
                    image = source
                    values, valid = reproject_volume_to_native_plane(image.get_fdata(), image.affine, affine, source_support.shape)
                method_values[method], method_valid[method] = values, valid
            if "final5" in config:
                proposed_methods = [method for method, (kind, _source) in method_sources.items() if kind == "direct_native_bundle"]
                if len(proposed_methods) != 1 or "Fit-first" not in method_values or fit_first_volume_geometry_valid is None or fit_first_query_effective_mask is None:
                    raise ValueError("Final5 native figures require one Proposed native bundle and one v6 Fit-first native bundle")
                proposed_method = proposed_methods[0]
                paired_support = final5_paired_display_support(
                    reference, method_values[proposed_method], method_values["Fit-first"], support,
                    method_valid[proposed_method], fit_first_volume_geometry_valid, fit_first_query_effective_mask,
                )
                panels.append({"label": "Native reference", "kind": "reference_ms", "displayed_quantity": "quantitative_map_ms", "values": reference, "support": paired_support, "cmap": cmap, "range_ms": limits})
                method_panels = [(proposed_method, method_values[proposed_method], paired_support), ("Fit-first", method_values["Fit-first"], paired_support)]
                display_support_definition = "primary_native_mask AND finite(reference) AND proposed_valid AND fit_first_volume_geometry_valid AND fit_first_query_effective_mask AND finite(proposed, fit_first)"
            else:
                panels.append({"label": "Native reference", "kind": "reference_ms", "displayed_quantity": "quantitative_map_ms", "values": reference, "support": support & np.isfinite(reference), "cmap": cmap, "range_ms": limits})
                method_panels = [(method, values, support & method_valid[method] & np.isfinite(reference) & np.isfinite(values)) for method, values in method_values.items()]
                display_support_definition = "primary_native_mask AND finite(reference); per-method panels additionally require endpoint-valid AND finite(prediction)"
            panels.extend({"label": method, "kind": "map_ms", "method": method, "displayed_quantity": "quantitative_map_ms", "values": values, "support": available, "cmap": cmap, "range_ms": limits} for method, values, available in method_panels)
            panels.extend({"label": f"|{method} − Native|", "kind": "absolute_error_ms", "method": method, "displayed_quantity": "absolute_error_ms", "values": np.abs(values - reference), "support": available, "cmap": cmap, "range_ms": limits} for method, values, available in method_panels)
            spacing = np.asarray(selected["pixel_spacing_rc_mm"][index], float)
            extent = [0.0, float(source_support.shape[0] - 1) * spacing[0], 0.0, float(source_support.shape[1] - 1) * spacing[1]]
            mosaic_rows.append({"group_idx": group, "group_label": native_group_label(group, group_statuses[group]), "panels": panels, "extent": extent, "contour": contour})
            shapes[str(group)] = list(source_support.shape)
        filename = f"native_{stack}_group{display_groups[0]}_{display_parameter}.png" if mode == "single" else f"native_{stack}_{mode}_{display_parameter}.png"
        assets = _plot_native_mosaic(output / filename, mosaic_rows, title=f"{config['subject']} {config['condition']} {config['endpoint']} — {stack} native {mode} {display_parameter}", component_directory=output / "components" / Path(filename).stem)
        inputs.setdefault("native_mask_manifest", _file_record(formal_mask_root / "manifest.json"))
        records.append({"parameter": display_parameter, "file": filename, "component_directory": f"components/{Path(filename).stem}", "components": [{**component, "parameter": display_parameter, "stack": stack, "support_semantics": display_support_definition} for component in assets["components"]], "colorbar_component": assets["colorbar"], "stack": stack, "mode": mode, "original_group_idx": display_groups[0] if mode == "single" else None, "original_group_ids": display_groups, "displayed_groups": [{"group_idx": group, "status": group_statuses[group], "label": native_group_label(group, group_statuses[group])} for group in display_groups], "native_reference": provenance, "primary_native_mask": "primary_native_mask", "mask_artifacts": mask_artifacts, "plane_shapes_rc": shapes, "display_support": display_support_definition, "interpolation": {"map": "linear", "mask": "artifact-defined native-plane support"}, "no_auto_crop": True, "physical_aspect_ratio_preserved": True, "colorbars": {"shared_parameter": "dedicated_axis", "range_ms": list(limits), "colormap": "same_parameter_map_lut"}, "error_display": {"quantity": "absolute_error_ms", "colormap": "same_parameter_map_lut", "range_ms": list(limits), "no_residual_colorbar": True}})
    return records


def _render_native(config: Mapping[str, Any], output: Path, inputs: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for native in native_preview_stack_configs(config["native"]):
        records.extend(_render_native_stack(config, output, inputs, native))
    return records


def render_final_figures(config_path: str | Path, output: str | Path) -> Path:
    """Create immutable, read-only subject figures from explicit formal artifact paths."""
    config_path, output = Path(config_path), Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config.get("subject"), str) or not config["subject"] or config.get("endpoint") not in ENDPOINTS:
        raise ValueError("final figures require an explicit subject and matched endpoint")
    if not isinstance(config.get("condition"), str):
        raise ValueError("final figures require an explicit condition/fold")
    _case(config["condition"], profile_id=config.get("profile_id"), case_id=config.get("case_id"))
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
        if method == "Fit-first" and config["condition"] in {"L100_fold0", "L100"} and "fit_first_native_bundle" not in maps:
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
        manifest = {"schema": "phase4r_final_subject_figures/v2", "subject": config["subject"], "condition": config.get("condition"), "endpoint": config["endpoint"], "methods": list(config["methods"]), "inputs": inputs, "longitudinal_figures": longitudinal, "scmr_style_longitudinal_figures": scmr_longitudinal, "native_heldout_figures": native, "shared_parameter_lut_ranges_ms": {key: list(value) for key, value in DISPLAY_RANGES_MS.items()}, "error_display_contract": "absolute_error_ms using same parameter map LUT/range; no magma residual scale; native mosaics provide one shared parameter colorbar", "code_git_sha": _git_sha()}
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
    aggregate.add_argument("--subject", default="CYJ"); aggregate.add_argument("--fit-first-source-method", required=True); aggregate.add_argument("--output", required=True)
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
        print(aggregate_final_subject(args.subject, roots, args.output, fit_first_source_method=args.fit_first_source_method))


if __name__ == "__main__":
    main()
