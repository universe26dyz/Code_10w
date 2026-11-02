"""Read-only Final5 metric, mask, and native-physical-coordinate QC audit.

The tool deliberately consumes published artifacts only.  It neither trains,
reconstructs, regenerates masks, nor writes below Final5 formal artifact
directories.  Final-pose (L100) and sparse held-out native-grid evidence are
kept in separate records because they are different coordinate contracts.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from .contracts import sha256, write_immutable_json
from .current_native_central_component_mask import load_validated_current_native_mask_plane
from .evaluation import _metrics, reproject_binary_mask_to_native_plane, reproject_volume_to_native_plane
from .final5_batch_runner import CaseRecord, _paths, build_batch_plan, load_final5_registry, validate_l100_metric, validate_sparse_metric
from .final_cyj_reporting import _load_all_verified_native_cache_groups
from .final_sparse_native_metrics import validate_sparse_endpoint_roots, validate_sparse_heldout_admission
from .fit_first_route_contract import EXP1_1_SCHEMA, resolve_fit_first_route
from .matched_output import verified_fit_first_effective_query_mask


FINAL_POSE_COORDINATE_SEMANTICS = "direct bundle world_train_ras converted with checkpoint model-space train_ras_to_physical; compared with Fit-first world_ras_mm in physical_RAS_mm"
SPARSE_NATIVE_GRID_COORDINATE_SEMANTICS = "Proposed volume, Fit-first volume, and Fit-first effective query mask independently sampled on the same original native RAS affine grid"
PRIMARY_SUPPORT_EQUATION = "primary_native_mask AND finite(reference) AND proposed_valid AND fit_first_volume_geometry_valid AND fit_first_query_effective_mask AND finite(proposed, fit_first)"
FORMAL_MUTABLE_ROOTS = ("04_reconstruction", "05_formal_endpoints", "06_metrics")


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    """Return a named-mask ratio; empty denominators are explicitly undefined."""
    return None if not denominator else float(numerator / denominator)


def _same_shape(*arrays: np.ndarray) -> None:
    if len({np.asarray(value).shape for value in arrays}) != 1:
        raise ValueError("audit arrays must share one native-plane shape")


def mask_count_summary(*, source_support: np.ndarray, phase4r_l100: np.ndarray, base_native_support: np.ndarray, primary_native_mask: np.ndarray, reference: np.ndarray, proposed: np.ndarray, proposed_valid: np.ndarray, fit_first: np.ndarray, fit_first_volume_geometry_valid: np.ndarray, fit_first_query_effective_mask: np.ndarray) -> dict[str, Any]:
    """Count distinct validity gates without conflating them into ``valid``."""
    arrays = (source_support, phase4r_l100, base_native_support, primary_native_mask, reference, proposed, proposed_valid, fit_first, fit_first_volume_geometry_valid, fit_first_query_effective_mask)
    _same_shape(*arrays)
    source, l100, base, primary = (np.asarray(value, bool) for value in arrays[:4])
    reference, proposed, fit_first = (np.asarray(value, float) for value in (reference, proposed, fit_first))
    proposed_valid, geometry, query = (np.asarray(value, bool) for value in (proposed_valid, fit_first_volume_geometry_valid, fit_first_query_effective_mask))
    fit_valid = geometry & query & np.isfinite(fit_first)
    paired = primary & np.isfinite(reference) & proposed_valid & geometry & query & np.isfinite(proposed) & np.isfinite(fit_first)
    proposed_region, fit_region = primary & proposed_valid & np.isfinite(proposed), primary & fit_valid
    union = proposed_region | fit_region
    return {
        "source_support_voxels": int(source.sum()), "phase4r_l100_voxels": int(l100.sum()),
        "base_native_support_voxels": int(base.sum()), "primary_native_mask_voxels": int(primary.sum()),
        "reference_finite_voxels": int(np.isfinite(reference).sum()), "proposed_valid_voxels": int(proposed_valid.sum()),
        "fit_first_volume_geometry_valid_voxels": int(geometry.sum()), "fit_first_query_effective_mask_voxels": int(query.sum()),
        "fit_first_valid_voxels": int(fit_valid.sum()), "paired_common_support_voxels": int(paired.sum()),
        "primary_over_source_support": safe_ratio(int(primary.sum()), int(source.sum())),
        "primary_over_base_native_support": safe_ratio(int(primary.sum()), int(base.sum())),
        "paired_over_primary_native_mask": safe_ratio(int(paired.sum()), int(primary.sum())),
        "paired_over_source_support": safe_ratio(int(paired.sum()), int(source.sum())),
        "proposed_valid_over_primary_native_mask": safe_ratio(int(proposed_region.sum()), int(primary.sum())),
        "fit_first_valid_over_primary_native_mask": safe_ratio(int(fit_region.sum()), int(primary.sum())),
        "method_valid_iou": safe_ratio(int((proposed_region & fit_region).sum()), int(union.sum())),
        "method_valid_identical": bool(np.array_equal(proposed_region, fit_region)),
        "fit_first_valid": fit_valid, "paired_common_support": paired,
    }


def _xyz(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, float)
    if values.ndim < 2:
        raise ValueError("coordinate array must include three physical components")
    if values.shape[-1] == 3:
        return values
    if values.shape[0] == 3:
        return np.moveaxis(values, 0, -1)
    raise ValueError("coordinate array must have a three-component axis")


def train_to_physical_ras(world_train_ras: np.ndarray, converter: Callable[[np.ndarray], Any]) -> np.ndarray:
    """Convert direct INR training coordinates before any physical comparison."""
    shape = _xyz(world_train_ras).shape
    converted = converter(_xyz(world_train_ras).reshape(-1, 3))
    if hasattr(converted, "detach"):
        converted = converted.detach().cpu().numpy()
    return np.asarray(converted, float).reshape(shape)


def coordinate_displacement_summary(proposed_physical_ras: np.ndarray, fit_first_physical_ras: np.ndarray, source_support: np.ndarray, *, tolerance_mm: float = 1e-3) -> tuple[dict[str, Any], np.ndarray]:
    proposed, fit_first = _xyz(proposed_physical_ras), _xyz(fit_first_physical_ras)
    _same_shape(proposed, fit_first)
    support = np.asarray(source_support, bool)
    if support.shape != proposed.shape[:-1]:
        raise ValueError("source support must match coordinate plane shape")
    displacement = np.linalg.norm(proposed - fit_first, axis=-1)
    values = displacement[support & np.isfinite(displacement)]
    summary: dict[str, Any] = {"comparison_space": "physical_RAS_mm", "coordinate_shape": list(proposed.shape), "finite_coordinate_voxels": int(values.size), "source_support_voxels": int(support.sum()), "tolerance_mm": float(tolerance_mm)}
    if values.size:
        summary.update({"mean_displacement_mm": float(values.mean()), "median_displacement_mm": float(np.median(values)), "p95_displacement_mm": float(np.percentile(values, 95)), "p99_displacement_mm": float(np.percentile(values, 99)), "max_displacement_mm": float(values.max()), "within_tolerance_fraction": float(np.mean(values <= tolerance_mm))})
    else:
        summary.update({key: None for key in ("mean_displacement_mm", "median_displacement_mm", "p95_displacement_mm", "p99_displacement_mm", "max_displacement_mm", "within_tolerance_fraction")})
    return summary, displacement


def require_new_qc_output(final5_root: str | Path, output: str | Path) -> None:
    root, target = Path(final5_root).resolve(), Path(output).resolve()
    if target.exists():
        raise FileExistsError(f"refusing overwrite: {target}")
    for name in FORMAL_MUTABLE_ROOTS:
        protected = root / name
        if target == protected or protected in target.parents:
            raise ValueError(f"QC output must not overlap formal artifact root: {protected}")


def _metric_path(root: Path, record: CaseRecord) -> tuple[str, Path]:
    paths = _paths(root, record)
    return ("L100_FINAL_POSE", paths["l100_metrics"]) if record.case_id == "L100" else ("SPARSE_HELDOUT", paths["sparse_metrics"])


def case_inventory_row(final5_root: str | Path, record: CaseRecord) -> dict[str, Any]:
    root = Path(final5_root); metric_kind, metric = _metric_path(root, record)
    manifest, aggregate = metric / "manifest.json", metric / "aggregate_metrics.json"
    if not manifest.is_file() or not aggregate.is_file():
        return {"subject": record.subject, "case_id": record.case_id, "metric_kind": metric_kind, "metric_path": str(metric), "metric_status": "MISSING", "auditable": False, "reason": "metric manifest or aggregate_metrics.json is absent"}
    try:
        result = validate_l100_metric(metric, record, root) if record.case_id == "L100" else validate_sparse_metric(metric, record, root)
    except (FileNotFoundError, ValueError, TypeError) as exc:
        return {"subject": record.subject, "case_id": record.case_id, "metric_kind": metric_kind, "metric_path": str(metric), "metric_status": "INVALID", "auditable": False, "reason": str(exc)}
    return {"subject": record.subject, "case_id": record.case_id, "metric_kind": metric_kind, "metric_path": str(metric), "metric_status": "COMPLETE_VALID" if result.valid else "INVALID_OR_BLOCKED", "auditable": bool(result.valid), "reason": result.reason, "validation_state": result.state}


def _native_ras_affine(cache: Mapping[str, Any], index: int) -> np.ndarray:
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ np.asarray(cache["affine_lps_rc"][index], float)


def _native_world(affine: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    row, col = np.indices(shape, dtype=float)
    indices = np.stack((row, col, np.zeros_like(row), np.ones_like(row)), axis=-1)
    return (np.asarray(affine, float) @ indices.reshape(-1, 4).T).T[:, :3].reshape(*shape, 3)


def _component_metadata(mask_root: Path, parameter: str, stack: str, group: int) -> dict[str, Any]:
    manifest = json.loads((mask_root / "manifest.json").read_text(encoding="utf-8"))
    matches = [item for item in manifest.get("planes", []) if item.get("parameter") == parameter and item.get("stack") == stack and int(item.get("original_group_idx", -1)) == group]
    return dict(matches[0].get("component_selection", {})) if len(matches) == 1 else {}


def _metric_comparison(*, stored: Mapping[str, Any] | None, recomputed: Mapping[str, Any] | None, subject: str, case_id: str, parameter: str, stack: str, group: int, mask_name: str, method: str) -> dict[str, Any]:
    row: dict[str, Any] = {"subject": subject, "case_id": case_id, "parameter": parameter, "stack": stack, "original_group_idx": group, "mask_name": mask_name, "method": method, "reproducible": False}
    if stored is None or recomputed is None:
        row["status"] = "NO_COMMON_SUPPORT"; return row
    deltas = {name: abs(float(stored.get(name, float("nan"))) - float(recomputed.get(name, float("nan")))) for name in ("rmse", "mae", "bias", "support_voxels")}
    finite = [value for value in deltas.values() if np.isfinite(value)]
    row.update({"status": "COMPARED", "max_abs_difference": max(finite) if finite else None, "reproducible": bool(finite and max(finite) <= 1e-6)})
    row.update({f"stored_{key}": stored.get(key) for key in ("rmse", "mae", "bias", "support_voxels")})
    row.update({f"recomputed_{key}": recomputed.get(key) for key in ("rmse", "mae", "bias", "support_voxels")})
    return row


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows); keys = sorted({key for row in materialized for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(materialized)


def _plot_plane(path: Path, panels: list[tuple[str, np.ndarray, np.ndarray]], *, parameter: str, title: str, extent: list[float]) -> None:
    import matplotlib.pyplot as plt
    from .visualization import DISPLAY_RANGES_MS, colorbars
    cmap, limits = colorbars().mapping[parameter.upper()], DISPLAY_RANGES_MS[parameter.upper()]
    figure = plt.figure(figsize=(2.55 * len(panels) + .6, 3.25)); grid = figure.add_gridspec(1, len(panels) + 1, width_ratios=[1] * len(panels) + [.055], wspace=.12)
    image = None
    for index, (label, values, support) in enumerate(panels):
        axis = figure.add_subplot(grid[0, index]); image = axis.imshow(np.ma.masked_invalid(np.where(support, values, np.nan)).T, origin="lower", extent=extent, cmap=cmap, vmin=limits[0], vmax=limits[1], aspect="equal", interpolation="none")
        axis.set_title(label, fontsize=8); axis.set_xticks([]); axis.set_yticks([])
    figure.colorbar(image, cax=figure.add_subplot(grid[0, -1]), label="ms; shared parameter scale")
    figure.suptitle(title, fontsize=9); figure.subplots_adjust(top=.80, bottom=.03, left=.02, right=.98); path.parent.mkdir(parents=True, exist_ok=True); figure.savefig(path, dpi=180); plt.close(figure)


def _plot_masks(path: Path, masks: list[tuple[str, np.ndarray]], *, title: str, extent: list[float]) -> None:
    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=(2.4 * len(masks), 2.9))
    for index, (label, values) in enumerate(masks, start=1):
        axis = figure.add_subplot(1, len(masks), index); axis.imshow(np.asarray(values, bool).T, origin="lower", extent=extent, cmap="gray", vmin=0, vmax=1, aspect="equal", interpolation="none")
        axis.set_title(label, fontsize=8); axis.set_xticks([]); axis.set_yticks([])
    figure.suptitle(title, fontsize=9); figure.subplots_adjust(top=.80, bottom=.03, left=.02, right=.98); path.parent.mkdir(parents=True, exist_ok=True); figure.savefig(path, dpi=180); plt.close(figure)


def _plot_reference_support_overlay(path: Path, reference: np.ndarray, masks: list[tuple[str, np.ndarray]], *, parameter: str, title: str, extent: list[float]) -> None:
    """Show every support over the same unmodified native reference plane."""
    import matplotlib.pyplot as plt
    from .visualization import DISPLAY_RANGES_MS, colorbars
    cmap, limits = colorbars().mapping[parameter.upper()], DISPLAY_RANGES_MS[parameter.upper()]
    figure = plt.figure(figsize=(2.5 * len(masks), 3.0))
    for index, (label, support) in enumerate(masks, start=1):
        axis = figure.add_subplot(1, len(masks), index); axis.imshow(reference.T, origin="lower", extent=extent, cmap=cmap, vmin=limits[0], vmax=limits[1], aspect="equal", interpolation="none")
        axis.imshow(np.ma.masked_where(~np.asarray(support, bool).T, np.asarray(support, bool).T), origin="lower", extent=extent, cmap="Reds", vmin=0, vmax=1, alpha=.45, aspect="equal", interpolation="none")
        axis.set_title(label, fontsize=8); axis.set_xticks([]); axis.set_yticks([])
    figure.suptitle(title, fontsize=9); figure.subplots_adjust(top=.80, bottom=.03, left=.02, right=.98); path.parent.mkdir(parents=True, exist_ok=True); figure.savefig(path, dpi=180); plt.close(figure)


def _plot_displacement(path: Path, displacement: np.ndarray, support: np.ndarray, *, title: str, extent: list[float]) -> None:
    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=(3.2, 3.1)); axis = figure.add_axes([.07, .08, .76, .80]); shown = np.ma.masked_invalid(np.where(support, displacement, np.nan))
    image = axis.imshow(shown.T, origin="lower", extent=extent, cmap="viridis", aspect="equal", interpolation="none")
    axis.set_xticks([]); axis.set_yticks([]); axis.set_title(title, fontsize=8); figure.colorbar(image, cax=figure.add_axes([.87, .11, .04, .72]), label="mm")
    path.parent.mkdir(parents=True, exist_ok=True); figure.savefig(path, dpi=180); plt.close(figure)


def _extent(cache: Mapping[str, Any], index: int, shape: tuple[int, int]) -> list[float]:
    spacing = np.asarray(cache["pixel_spacing_rc_mm"][index], float)
    return [0.0, (shape[0] - 1) * float(spacing[0]), 0.0, (shape[1] - 1) * float(spacing[1])]


def _proposed_physical_converter(direct_root: Path, manifest: Mapping[str, Any], *, device: str) -> Callable[[np.ndarray], Any]:
    """Load only the published checkpoint/model-space transform on the requested device."""
    from .direct_native_endpoint_audit import _load_model
    checkpoint = Path(manifest["checkpoint"]["path"])
    prepared = [Path(manifest["prepared_observations"][stack]["path"]) for stack in ("sax", "2ch", "4ch")]
    _model, space, torch = _load_model(checkpoint, prepared, device)
    return lambda points: space.train_ras_to_physical(torch.from_numpy(np.asarray(points, np.float32)))


def _l100_audit(record: CaseRecord, root: Path, qc: Path, *, figures: bool, device: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    import nibabel as nib
    paths = _paths(root, record); direct_root, fit_root, mask_root, metric_root = paths["direct_bundle"], paths["fit_bundle"], paths["native_mask"], paths["l100_metrics"]
    direct_manifest = json.loads((direct_root / "manifest.json").read_text(encoding="utf-8")); metric_manifest = json.loads((metric_root / "manifest.json").read_text(encoding="utf-8"))
    route = resolve_fit_first_route(paths["fit_first"])
    existing = {(row["parameter"], row["stack"], int(row["original_group_idx"])): row for row in metric_manifest.get("per_plane_records", [])}
    caches = {stack: _load_all_verified_native_cache_groups(root / "02_standardized_12_15_10" / "native_dictionary" / record.subject, record.sparse_manifest, stack) for stack in ("sax", "2ch", "4ch")}
    converter: Callable[[np.ndarray], Any] | None
    converter_error: dict[str, str] | None = None
    try:
        converter = _proposed_physical_converter(direct_root, direct_manifest, device=device)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, KeyError) as exc:
        converter = None
        converter_error = {"exception_type": type(exc).__name__, "exception_message": str(exc)}
    endpoint_paths, _endpoint_provenance = validate_sparse_endpoint_roots(paths["proposed_endpoint"], paths["fit_endpoint"], record.case_id, subject=record.subject)
    endpoint_images = {parameter: {name: nib.load(str(path)) for name, path in item.items()} for parameter, item in endpoint_paths.items()}
    endpoint_masks = {parameter: nib.load(str(verified_fit_first_effective_query_mask(paths["fit_endpoint"], parameter)[0])) for parameter in ("t1", "t2")}
    counts: list[dict[str, Any]] = []; reproduction: list[dict[str, Any]] = []; coordinates: list[dict[str, Any]] = []; warnings: list[dict[str, Any]] = []
    for plane in direct_manifest.get("planes", []):
        stack, group = str(plane["stack"]), int(plane["original_group_idx"]); cache = caches[stack]
        indices = np.flatnonzero(np.asarray(cache["group_idx"], int) == group)
        if len(indices) != 1:
            warnings.append({"subject": record.subject, "case_id": record.case_id, "stack": stack, "original_group_idx": group, "warning": "R0_REFERENCE_GROUP_UNAVAILABLE"}); continue
        index = int(indices[0]); affine = _native_ras_affine(cache, index)
        with np.load(direct_root / plane["path"], allow_pickle=False) as direct:
            train = np.asarray(direct["world_train_ras"], float)
            for parameter in ("t1", "t2"):
                reference, source = np.asarray(cache[f"{parameter}_ms"][index], float), np.asarray(cache["source_support"][index], bool)
                masks, _provenance = load_validated_current_native_mask_plane(mask_root, parameter=parameter, stack=stack, original_group_idx=group, shape_rc=tuple(reference.shape), source_support=source)
                with np.load(fit_root / "planes" / parameter / stack / f"group{group:03d}.npz", allow_pickle=False) as fit:
                    proposed, fit_first = np.asarray(direct[f"{parameter}_ms"], float), np.asarray(fit["prediction_ms"], float)
                    geometry, query = np.asarray(fit["volume_geometry_valid"], bool), np.asarray(fit["query_effective_mask"], bool)
                    summary = mask_count_summary(source_support=source, phase4r_l100=masks["phase4r_l100"], base_native_support=masks["base_native_support"], primary_native_mask=masks["primary_native_mask"], reference=reference, proposed=proposed, proposed_valid=np.isfinite(proposed), fit_first=fit_first, fit_first_volume_geometry_valid=geometry, fit_first_query_effective_mask=query)
                    paired = np.asarray(summary.pop("paired_common_support"), bool); summary.pop("fit_first_valid")
                    component = _component_metadata(mask_root, parameter, stack, group)
                    row = {"subject": record.subject, "case_id": record.case_id, "metric_kind": "L100_FINAL_POSE", "parameter": parameter, "stack": stack, "original_group_idx": group, **summary, **{f"component_{key}": value for key, value in component.items()}}
                    counts.append(row)
                    for mask_name, support in (("primary_native_mask", paired), ("phase4r_l100", np.asarray(masks["phase4r_l100"], bool) & np.isfinite(reference) & np.isfinite(proposed) & geometry & query & np.isfinite(fit_first))):
                        saved = existing.get((parameter, stack, group), {}).get("masks", {}).get(mask_name, {}).get("metrics")
                        for method, values in (("proposed", proposed), ("fit_first", fit_first)):
                            recomputed = _metrics(reference, values, support) if support.any() else None
                            reproduction.append(_metric_comparison(stored=None if saved is None else saved.get(method), recomputed=recomputed, subject=record.subject, case_id=record.case_id, parameter=parameter, stack=stack, group=group, mask_name=mask_name, method=method))
                    if converter is None:
                        coordinates.append({"subject": record.subject, "case_id": record.case_id, "parameter": parameter, "stack": stack, "original_group_idx": group, "status": "MODEL_SPACE_CONVERTER_FAILED", "coordinate_semantics": FINAL_POSE_COORDINATE_SEMANTICS, "fit_first_route_kind": route.route_kind, "fit_first_route_schema": route.route_schema, "native_prediction_source": "Proposed direct native bundle vs Fit-first final NeSVoR pose bundle", "device": device, **(converter_error or {})})
                    else:
                        physical = train_to_physical_ras(train, converter); fit_world = _xyz(np.asarray(fit["world_ras_mm"], float)); coord, displacement = coordinate_displacement_summary(physical, fit_world, source)
                        native_world = _native_world(affine, reference.shape); proposed_native, _ = coordinate_displacement_summary(physical, native_world, source); fit_native, _ = coordinate_displacement_summary(fit_world, native_world, source)
                        coordinates.append({"subject": record.subject, "case_id": record.case_id, "parameter": parameter, "stack": stack, "original_group_idx": group, "status": "COMPARED", "coordinate_semantics": FINAL_POSE_COORDINATE_SEMANTICS, "fit_first_route_kind": route.route_kind, "fit_first_route_schema": route.route_schema, "native_prediction_source": "Proposed direct native bundle vs Fit-first final NeSVoR pose bundle", **coord, "proposed_vs_source_within_tolerance_fraction": proposed_native["within_tolerance_fraction"], "fit_first_vs_source_within_tolerance_fraction": fit_native["within_tolerance_fraction"]})
                    if figures:
                        base = qc / "figures" / record.subject / record.case_id / parameter / stack / f"group{group:03d}"; extent = _extent(cache, index, reference.shape)
                        _plot_reference_support_overlay(base / "A_reference_support_overlays.png", reference, [("source", source), ("frozen L100", masks["phase4r_l100"]), ("base", masks["base_native_support"]), ("primary", masks["primary_native_mask"])], parameter=parameter, title=f"{record.subject} {record.case_id} {parameter.upper()} {stack} group{group:03d} supports", extent=extent)
                        _plot_masks(base / "B_validity_masks.png", [("Proposed valid", np.isfinite(proposed)), ("Fit geometry", geometry), ("Fit effective", query), ("Fit valid", geometry & query & np.isfinite(fit_first)), ("paired", paired)], title=f"{record.subject} {record.case_id} {parameter.upper()} {stack} group{group:03d} validity", extent=extent)
                        _plot_plane(base / "C_final_pose_native.png", [("Native", reference, paired), ("Proposed pose", proposed, paired), ("Fit-first pose", fit_first, paired), ("|P-N|", np.abs(proposed-reference), paired), ("|F-N|", np.abs(fit_first-reference), paired)], parameter=parameter, title=f"Final-pose values; {FINAL_POSE_COORDINATE_SEMANTICS}", extent=extent)
                        endpoint_proposed, endpoint_pvalid = reproject_volume_to_native_plane(endpoint_images[parameter]["proposed"].get_fdata(), endpoint_images[parameter]["proposed"].affine, affine, reference.shape)
                        endpoint_fit, endpoint_fgeom = reproject_volume_to_native_plane(endpoint_images[parameter]["fit_first"].get_fdata(), endpoint_images[parameter]["fit_first"].affine, affine, reference.shape)
                        endpoint_query = reproject_binary_mask_to_native_plane(endpoint_masks[parameter].get_fdata(), endpoint_masks[parameter].affine, affine, reference.shape)
                        endpoint_paired = np.asarray(masks["primary_native_mask"], bool) & np.isfinite(reference) & endpoint_pvalid & endpoint_fgeom & endpoint_query & np.isfinite(endpoint_proposed) & np.isfinite(endpoint_fit)
                        _plot_plane(base / "D_original_native_physical_grid.png", [("Native", reference, endpoint_paired), ("Proposed", endpoint_proposed, endpoint_paired), ("Fit-first", endpoint_fit, endpoint_paired), ("|P-N|", np.abs(endpoint_proposed-reference), endpoint_paired), ("|F-N|", np.abs(endpoint_fit-reference), endpoint_paired)], parameter=parameter, title=SPARSE_NATIVE_GRID_COORDINATE_SEMANTICS, extent=extent)
                        if converter is not None:
                            _plot_displacement(base / "E_physical_coordinate_displacement_mm.png", displacement, source, title="Proposed vs Fit-first physical RAS displacement", extent=extent)
                    coverage = row["primary_over_source_support"]
                    if not row["primary_native_mask_voxels"] or (coverage is not None and (coverage < .05 or coverage > .98)) or component.get("component_count", 1) > 1:
                        warnings.append({"subject": record.subject, "case_id": record.case_id, "parameter": parameter, "stack": stack, "original_group_idx": group, "warning": "PRIMARY_MASK_REVIEW", "detail": "primary_native_mask is a source/L100 central-component support, not a myocardium segmentation"})
    return counts, reproduction, coordinates, warnings


def _sparse_audit(record: CaseRecord, root: Path, qc: Path, *, figures: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    import nibabel as nib
    paths = _paths(root, record); sparse = json.loads(record.sparse_manifest.read_text(encoding="utf-8")); admission = validate_sparse_heldout_admission(sparse); route = resolve_fit_first_route(paths["fit_first"])
    volume_paths, _endpoint = validate_sparse_endpoint_roots(paths["proposed_endpoint"], paths["fit_endpoint"], record.case_id, subject=record.subject)
    images = {parameter: {name: nib.load(str(path)) for name, path in item.items()} for parameter, item in volume_paths.items()}; masks3d = {parameter: nib.load(str(verified_fit_first_effective_query_mask(paths["fit_endpoint"], parameter)[0])) for parameter in ("t1", "t2")}
    metric_payload = json.loads((paths["sparse_metrics"] / "aggregate_metrics.json").read_text(encoding="utf-8")); saved = {(parameter, item["stack"], int(item["original_group_idx"])): item["parameters"].get(parameter, {}) for item in metric_payload.get("per_plane_records", []) for parameter in ("t1", "t2")}
    counts: list[dict[str, Any]] = []; reproduction: list[dict[str, Any]] = []; coordinates: list[dict[str, Any]] = []; warnings: list[dict[str, Any]] = []
    native_root = root / "02_standardized_12_15_10" / "native_dictionary" / record.subject
    for stack, groups in admission["heldout"].items():
        if not groups: continue
        from .fit_first import select_verified_native_cache_groups
        cache = select_verified_native_cache_groups(native_root, record.sparse_manifest, stack, groups)
        for index, group in enumerate(groups):
            affine = _native_ras_affine(cache, index)
            for parameter in ("t1", "t2"):
                reference, source = np.asarray(cache[f"{parameter}_ms"][index], float), np.asarray(cache["source_support"][index], bool)
                masks, _ = load_validated_current_native_mask_plane(paths["native_mask"], parameter=parameter, stack=stack, original_group_idx=int(group), shape_rc=tuple(reference.shape), source_support=source)
                proposed, pvalid = reproject_volume_to_native_plane(images[parameter]["proposed"].get_fdata(), images[parameter]["proposed"].affine, affine, reference.shape)
                fit, fgeom = reproject_volume_to_native_plane(images[parameter]["fit_first"].get_fdata(), images[parameter]["fit_first"].affine, affine, reference.shape)
                query = reproject_binary_mask_to_native_plane(masks3d[parameter].get_fdata(), masks3d[parameter].affine, affine, reference.shape)
                summary = mask_count_summary(source_support=source, phase4r_l100=masks["phase4r_l100"], base_native_support=masks["base_native_support"], primary_native_mask=masks["primary_native_mask"], reference=reference, proposed=proposed, proposed_valid=pvalid, fit_first=fit, fit_first_volume_geometry_valid=fgeom, fit_first_query_effective_mask=query)
                paired = summary.pop("paired_common_support"); summary.pop("fit_first_valid")
                counts.append({"subject": record.subject, "case_id": record.case_id, "metric_kind": "SPARSE_HELDOUT", "parameter": parameter, "stack": stack, "original_group_idx": int(group), **summary})
                stored = saved.get((parameter, stack, int(group)), {}).get("metrics")
                for method, values in (("proposed", proposed), ("fit_first", fit)):
                    reproduction.append(_metric_comparison(stored=None if stored is None else stored.get(method), recomputed=_metrics(reference, values, paired) if paired.any() else None, subject=record.subject, case_id=record.case_id, parameter=parameter, stack=stack, group=int(group), mask_name="primary_native_mask", method=method))
                coordinates.append({"subject": record.subject, "case_id": record.case_id, "parameter": parameter, "stack": stack, "original_group_idx": int(group), "status": "COMMON_ORIGINAL_NATIVE_GRID", "coordinate_semantics": SPARSE_NATIVE_GRID_COORDINATE_SEMANTICS, "fit_first_route_kind": route.route_kind, "fit_first_route_schema": route.route_schema, "native_prediction_source": "central endpoint volumes sampled on original native physical grid", "native_affine_ras": json.dumps(affine.tolist()), "native_shape_rc": json.dumps(list(reference.shape))})
                if figures:
                    base = qc / "figures" / record.subject / record.case_id / parameter / stack / f"group{int(group):03d}"; extent = _extent(cache, index, reference.shape)
                    _plot_reference_support_overlay(base / "A_reference_support_overlays.png", reference, [("source", source), ("frozen L100", masks["phase4r_l100"]), ("base", masks["base_native_support"]), ("primary", masks["primary_native_mask"])], parameter=parameter, title=f"{record.subject} {record.case_id} {parameter.upper()} {stack} group{int(group):03d} supports", extent=extent)
                    _plot_masks(base / "B_validity_masks.png", [("Proposed valid", pvalid), ("Fit geometry", fgeom), ("Fit effective", query), ("Fit valid", fgeom & query & np.isfinite(fit)), ("paired", paired)], title="Original-native-grid validity", extent=extent)
                    _plot_plane(base / "D_original_native_grid.png", [("Native", reference, paired), ("Proposed", proposed, paired), ("Fit-first", fit, paired), ("|P-N|", np.abs(proposed-reference), paired), ("|F-N|", np.abs(fit-reference), paired)], parameter=parameter, title=SPARSE_NATIVE_GRID_COORDINATE_SEMANTICS, extent=extent)
                if not paired.any(): warnings.append({"subject": record.subject, "case_id": record.case_id, "parameter": parameter, "stack": stack, "original_group_idx": int(group), "warning": "EMPTY_PAIRED_SUPPORT"})
    return counts, reproduction, coordinates, warnings


def run_metric_mask_coordinate_audit(*, registry: str | Path, final5_root: str | Path, output: str | Path, device: str, figures: bool = True, legacy_cyj_l100_metrics: str | Path | None = None, legacy_cyj_sparse_metrics: str | Path | None = None) -> Path:
    """Audit all registry cases into a new QC directory without formal writes."""
    root, output = Path(final5_root), Path(output); require_new_qc_output(root, output)
    records = load_final5_registry(registry); inventory = [case_inventory_row(root, record) for record in records]
    plan = build_batch_plan(registry, root, device="0", records=records)
    for item in inventory:
        # The QC readers below intentionally retain legacy-artifact diagnostics,
        # while batch status reports the current formal native-grid metric.
        stage = "final_native_grid_metrics"
        state = plan["cases"][f"{item['subject']}/{item['case_id']}"]["stages"][stage]
        item["pipeline_state"], item["pipeline_reason"] = state["state"], state.get("reason")
    output.parent.mkdir(parents=True, exist_ok=True); staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        count_rows: list[dict[str, Any]] = []; metric_rows: list[dict[str, Any]] = []; coordinate_rows: list[dict[str, Any]] = []; warnings: list[dict[str, Any]] = []
        for record, item in zip(records, inventory):
            if not item["auditable"]: continue
            try:
                values = _l100_audit(record, root, staging, figures=figures, device=device) if record.case_id == "L100" else _sparse_audit(record, root, staging, figures=figures)
                for target, produced in zip((count_rows, metric_rows, coordinate_rows, warnings), values): target.extend(produced)
            except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
                item["auditable"] = False; item["metric_status"] = "AUDIT_BLOCKED"; item["reason"] = str(exc)
        summary_rows = []
        for key in sorted({(row["subject"], row["case_id"]) for row in count_rows}):
            rows = [row for row in count_rows if (row["subject"], row["case_id"]) == key]
            summary_rows.append({"subject": key[0], "case_id": key[1], "planes_parameters": len(rows), "primary_native_mask_voxels": sum(row["primary_native_mask_voxels"] for row in rows), "paired_common_support_voxels": sum(row["paired_common_support_voxels"] for row in rows), "paired_over_primary_native_mask": safe_ratio(sum(row["paired_common_support_voxels"] for row in rows), sum(row["primary_native_mask_voxels"] for row in rows)), "anatomical_note": "primary_native_mask is not myocardium-only; no trusted myocardium-only conclusion is available"})
        _write_csv(staging / "case_inventory.csv", inventory); _write_csv(staging / "per_plane_mask_counts.csv", count_rows); _write_csv(staging / "per_plane_metric_reproduction.csv", metric_rows); _write_csv(staging / "per_plane_coordinate_displacement.csv", coordinate_rows); _write_csv(staging / "mask_summary_by_subject_case.csv", summary_rows); _write_csv(staging / "qc_warnings.csv", warnings)
        legacy_rows = []
        for name, value in (("cyj_l100_metrics", legacy_cyj_l100_metrics), ("cyj_sparse_metrics", legacy_cyj_sparse_metrics)):
            path = None if value is None else Path(value)
            legacy_rows.append({"legacy_scope": name, "path": None if path is None else str(path.resolve()), "status": "NOT_REQUESTED" if path is None else "PRESENT" if path.exists() else "MISSING", "separate_from_final5": True})
        _write_csv(staging / "legacy_inventory.csv", legacy_rows)
        legacy = {"inventory_file": "legacy_inventory.csv", "semantics": "legacy paths are reported separately and never combined with Final5 rows"}
        write_immutable_json(staging / "audit_manifest.json", {"schema": "phase4r_final5_metric_mask_coordinate_audit/v1", "created_utc": datetime.now(timezone.utc).isoformat(), "final5_root": str(root.resolve()), "registry": {"path": str(Path(registry).resolve()), "sha256": sha256(registry)}, "read_only": True, "formal_artifacts_modified": False, "device": device, "final_pose_coordinate_semantics": FINAL_POSE_COORDINATE_SEMANTICS, "sparse_native_grid_coordinate_semantics": SPARSE_NATIVE_GRID_COORDINATE_SEMANTICS, "primary_support_equation": PRIMARY_SUPPORT_EQUATION, "legacy_cyj": legacy, "inventory_counts": {"total": len(inventory), "auditable": sum(bool(row["auditable"]) for row in inventory), "missing_or_blocked": sum(not bool(row["auditable"]) for row in inventory)}, "figures_requested": bool(figures)})
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True); raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True); parser.add_argument("--final5-root", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--device", required=True); parser.add_argument("--no-figures", action="store_true"); parser.add_argument("--legacy-cyj-l100-metrics"); parser.add_argument("--legacy-cyj-sparse-metrics")
    args = parser.parse_args(); print(run_metric_mask_coordinate_audit(registry=args.registry, final5_root=args.final5_root, output=args.output, device=args.device, figures=not args.no_figures, legacy_cyj_l100_metrics=args.legacy_cyj_l100_metrics, legacy_cyj_sparse_metrics=args.legacy_cyj_sparse_metrics))


if __name__ == "__main__":
    main()
