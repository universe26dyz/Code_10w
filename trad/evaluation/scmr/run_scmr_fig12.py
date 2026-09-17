"""Generate SCMR Figure 1/2 and concise Trad consistency metrics.

Native maps are accepted only from a provenance-verified bundle generated from
the exact current ``MP-PCA(MIND_mag_reg)`` preprocessed MAT files.  Quantitative
agreement is native dictionary-map consistency, not ground-truth accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import METRIC_COLUMNS, agreement_metrics
from .reference_2d import NativeReference, STACKS, load_verified_reference, sha256


LEGACY_FILES = [
    "python/visualization/scmr/make_figure1_throughplane_svr.py",
    "python/visualization/qc/make_figure1_throughplane_candidates.py",
    "python/visualization/scmr/make_figure2_reprojection_residual.py",
    "python/evaluation/reprojection/prepare_common_evaluation_mask.py",
    "python/evaluation/reprojection/refine_common_support_to_central_component.py",
    "python/evaluation/utils/metrics.py",
]


def _write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0]) if rows else ["subject_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _groups_from_reprojection(data: dict[str, np.ndarray], stack: str, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = {"group_idx", "weight_idx", "masks"}
    if kind == "quantitative":
        required.update(("t1_ms", "t2_ms"))
    else:
        required.update(("observed", "predicted"))
    missing = required.difference(data)
    if missing:
        raise ValueError(f"{stack} {kind} reprojection lacks {sorted(missing)}")
    groups = np.asarray(data["group_idx"], dtype=np.int64)
    weights = np.asarray(data["weight_idx"], dtype=np.int64)
    masks = np.asarray(data["masks"], dtype=bool)
    shape = np.asarray(data["t1_ms"] if kind == "quantitative" else data["observed"]).shape
    if len(shape) != 3 or masks.shape != shape or groups.shape != (shape[0],) or weights.shape != (shape[0],):
        raise ValueError(f"{stack} {kind} reprojection shape/metadata mismatch.")
    expected = np.arange(int(groups.max()) + 1) if groups.size else np.empty(0, dtype=int)
    if not np.array_equal(np.unique(groups), expected):
        raise ValueError(f"{stack} {kind} group_idx is not contiguous zero-based.")
    return groups, weights, masks, np.asarray(shape[1:], dtype=int)


def _extract_quantitative_stack(reference: NativeReference, run: Path, stack: str) -> dict[str, np.ndarray]:
    data = _load_npz(run / f"t1_t2_native_plane_{stack}.npz")
    groups, weights, masks, image_shape = _groups_from_reprojection(data, stack, "quantitative")
    native = reference.stacks[stack]
    if native.t1_ms.shape != (len(np.unique(groups)), *image_shape):
        raise ValueError(
            f"{stack} native reference shape {native.t1_ms.shape} does not match "
            f"Trad groups/images {(len(np.unique(groups)), *image_shape)}."
        )
    t1, t2 = [], []
    support = []
    for group in native.group_idx:
        rows = np.flatnonzero(groups == group)
        if rows.size != 10 or not np.array_equal(np.sort(weights[rows]), np.arange(10)):
            raise ValueError(f"{stack} group {group} must contain one observation for each of 10 weights.")
        weight_zero = rows[weights[rows] == 0]
        if weight_zero.size != 1:
            raise ValueError(f"{stack} group {group} has no unique weight 0 quantitative reprojection.")
        index = int(weight_zero[0])
        t1.append(np.asarray(data["t1_ms"][index], dtype=np.float32))
        t2.append(np.asarray(data["t2_ms"][index], dtype=np.float32))
        support.append(np.all(masks[rows], axis=0))
    return {"t1_ms": np.stack(t1), "t2_ms": np.stack(t2), "support": np.stack(support)}


def _quantitative_metrics(reference: NativeReference, run: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    per_slice: list[dict[str, Any]] = []
    per_stack: list[dict[str, Any]] = []
    loaded: dict[str, dict[str, np.ndarray]] = {}
    pooled: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {"T1": [], "T2": []}
    for stack in STACKS:
        value = _extract_quantitative_stack(reference, run, stack)
        loaded[stack] = value
        native = reference.stacks[stack]
        for parameter, key in (("T1", "t1_ms"), ("T2", "t2_ms")):
            refs, preds = getattr(native, key), value[key]
            masks = native.valid_mask & value["support"] & np.isfinite(refs) & np.isfinite(preds)
            for group in native.group_idx:
                metrics = agreement_metrics(refs[group], preds[group], masks[group])
                per_slice.append({"subject_id": reference.manifest["subject_id"], "parameter": parameter, "stack": stack,
                                  "group_idx": int(group), "mask_definition": "native_valid AND Trad_support AND finite(reference,prediction)",
                                  "units": "ms", **metrics})
            per_stack.append({"subject_id": reference.manifest["subject_id"], "parameter": parameter, "stack": stack,
                              "group_idx": "ALL", "mask_definition": "native_valid AND Trad_support AND finite(reference,prediction)",
                              "units": "ms", **agreement_metrics(refs, preds, masks)})
            pooled[parameter].append((refs, preds, masks))
    overall = {}
    for parameter, arrays in pooled.items():
        ref = np.concatenate([x[0].ravel() for x in arrays])
        pred = np.concatenate([x[1].ravel() for x in arrays])
        mask = np.concatenate([x[2].ravel() for x in arrays])
        overall[parameter] = agreement_metrics(ref, pred, mask)
    return per_slice, per_stack, {"definition": "quantitative agreement / consistency with native 2-D dictionary maps; not ground-truth accuracy", "overall": overall}, loaded


def _signal_metrics(run: Path, subject_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pools: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for stack in STACKS:
        data = _load_npz(run / f"signal_reprojection_{stack}.npz")
        groups, weights, masks, _ = _groups_from_reprojection(data, stack, "signal")
        observed, predicted = np.asarray(data["observed"], dtype=np.float32), np.asarray(data["predicted"], dtype=np.float32)
        for weight in range(10):
            indices = np.flatnonzero(weights == weight)
            if not indices.size:
                raise ValueError(f"{stack} signal reprojection lacks weight {weight}.")
            metrics = agreement_metrics(observed[indices], predicted[indices], masks[indices])
            rows.append({"subject_id": subject_id, "stack": stack, "weight_idx": weight, "units": "signal intensity", **metrics})
        pools.append((observed, predicted, masks))
    observed = np.concatenate([x[0].ravel() for x in pools])
    predicted = np.concatenate([x[1].ravel() for x in pools])
    masks = np.concatenate([x[2].ravel() for x in pools])
    overall = agreement_metrics(observed, predicted, masks)
    rows.append({"subject_id": subject_id, "stack": "ALL", "weight_idx": "ALL", "units": "signal intensity", **overall})
    return rows, {"definition": "observed weighted image versus existing Trad forward-predicted weighted image on prepared support", "overall": overall}


def _select_groups(loaded: dict[str, dict[str, np.ndarray]], requested: dict[str, int] | None) -> dict[str, int]:
    selected: dict[str, int] = {}
    for stack, value in loaded.items():
        if requested and stack in requested:
            group = requested[stack]
            if not 0 <= group < value["support"].shape[0]:
                raise IndexError(f"Selected {stack} group {group} is unavailable.")
        else:
            group = int(np.argmax(value["support"].sum(axis=(1, 2))))
        selected[stack] = group
    return selected


def _crop(data: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.where(mask)
    if not rows.size:
        return data, mask
    r0, r1 = max(0, rows.min() - 3), min(data.shape[0], rows.max() + 4)
    c0, c1 = max(0, cols.min() - 3), min(data.shape[1], cols.max() + 4)
    return data[r0:r1, c0:c1], mask[r0:r1, c0:c1]


def _render_figure2(reference: NativeReference, loaded: dict[str, dict[str, np.ndarray]], selected: dict[str, int], output: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {"selected_groups": selected, "mask_definition": "native_valid AND Trad_support AND finite(reference,prediction)"}
    masks_to_save: dict[str, np.ndarray] = {}
    for parameter, key, display_range in (("T1", "t1_ms", (0, 2500)), ("T2", "t2_ms", (0, 200))):
        panels = []
        for stack in STACKS:
            group, native = selected[stack], reference.stacks[stack]
            ref, pred = native.__getattribute__(key)[group], loaded[stack][key][group]
            mask = native.valid_mask[group] & loaded[stack]["support"][group] & np.isfinite(ref) & np.isfinite(pred)
            panels.append((stack, ref, pred, mask))
            masks_to_save[f"mask_{parameter.lower()}_{stack}"] = mask
        residual_values = [np.abs(pred[mask] - ref[mask]) for _, ref, pred, mask in panels if mask.any()]
        if not residual_values:
            raise ValueError(f"No common valid/support pixels are available for Figure 2 {parameter}.")
        residual_max = float(np.percentile(np.concatenate(residual_values), 99))
        figure, axes = plt.subplots(3, 4, figsize=(11.4, 8.4), gridspec_kw={"width_ratios": [1, 1, 1, 0.055]})
        image = None
        residual_image = None
        for row, (stack, ref, pred, mask) in enumerate(panels):
            cropped_ref, cropped_mask = _crop(ref, mask)
            cropped_pred, _ = _crop(pred, mask)
            residual, _ = _crop(np.abs(pred - ref), mask)
            for axis, data, title, cmap, limits in ((axes[row, 0], cropped_ref, "Native 2-D", "viridis", display_range),
                                                     (axes[row, 1], cropped_pred, "Trad 3-D → native plane", "viridis", display_range),
                                                     (axes[row, 2], residual, "|Difference|", "magma", (0, residual_max))):
                rendered = np.ma.masked_where(~cropped_mask, data)
                handle = axis.imshow(rendered, cmap=cmap, vmin=limits[0], vmax=limits[1], origin="lower", interpolation="nearest", aspect="equal")
                axis.set_title(title if row == 0 else "")
                axis.set_ylabel(stack.upper())
                axis.axis("off")
                if title == "|Difference|": residual_image = handle
                else: image = handle
        figure.colorbar(image, cax=axes[0, 3], label=f"{parameter} (ms)")
        figure.colorbar(residual_image, cax=axes[2, 3], label="absolute difference (ms)")
        figure.suptitle(f"{reference.manifest['subject_id']} {parameter}: quantitative agreement with native 2-D dictionary maps")
        png = output / f"Figure2_{parameter}_native_vs_reprojection.png"
        figure.savefig(png, dpi=320, bbox_inches="tight")
        figure.savefig(png.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)
        metadata[parameter] = {"mapping_range_ms": list(display_range), "residual_range_ms": [0, residual_max], "png": str(png), "pdf": str(png.with_suffix('.pdf'))}
    np.savez_compressed(output / "selected_common_masks.npz", **masks_to_save)
    _write_json(metadata, output / "figure2_metadata.json")
    return metadata


def _require_nifti() -> tuple[Any, Any]:
    try:
        import nibabel as nib
        from scipy.ndimage import map_coordinates
    except ImportError as exc:
        raise RuntimeError("Figure 1 requires nibabel and scipy (the knesvr_torch environment provides both).") from exc
    return nib, map_coordinates


def _plane_world(affine: np.ndarray, shape: tuple[int, int, int], axis: int, index: int) -> tuple[np.ndarray, tuple[int, int], tuple[float, float]]:
    other = 1 - axis
    nu = shape[other]
    spacing_u = float(np.linalg.norm(affine[:3, other]))
    spacing_z = float(np.linalg.norm(affine[:3, 2]))
    nv = max(2, int(round((shape[2] - 1) * spacing_z)) + 1)
    grid_u, grid_z = np.meshgrid(np.arange(nu), np.linspace(0, shape[2] - 1, nv), indexing="xy")
    voxels = np.zeros((grid_u.size, 4), dtype=float)
    voxels[:, axis] = index
    voxels[:, other] = grid_u.ravel()
    voxels[:, 2] = grid_z.ravel()
    voxels[:, 3] = 1
    return (affine @ voxels.T).T, (nv, nu), (spacing_u * (nu - 1), spacing_z * (shape[2] - 1))


def _sample_plane_shaped(source: np.ndarray, inverse_affine: np.ndarray, world: np.ndarray, plane_shape: tuple[int, int], order: int) -> np.ndarray:
    _, map_coordinates = _require_nifti()
    voxels = (inverse_affine @ world.T).T[:, :3].T
    return map_coordinates(source, voxels, order=order, mode="constant", cval=np.nan).reshape(plane_shape)


def _foreground_center(data: np.ndarray, axis: int) -> int:
    foreground = np.isfinite(data) & (data > 0)
    projected = foreground.any(axis=tuple(i for i in range(3) if i != axis))
    indices = np.flatnonzero(projected)
    return int(np.median(indices)) if indices.size else data.shape[axis] // 2


def _render_figure1_candidates(native: np.ndarray, native_affine: np.ndarray, trad: np.ndarray, trad_affine: np.ndarray, output: Path, axis_name: str) -> dict[str, Any]:
    """Save a compact, deterministic candidate montage for manual plane QC."""

    import matplotlib.pyplot as plt

    axis = {"x": 0, "y": 1}[axis_name]
    center = _foreground_center(native, axis)
    radius = max(1, native.shape[axis] // 6)
    indices = sorted(set(int(np.clip(value, 0, native.shape[axis] - 1)) for value in np.linspace(center - radius, center + radius, 5)))
    figure, axes = plt.subplots(len(indices), 2, figsize=(7.6, 2.5 * len(indices)), squeeze=False)
    paths = []
    for row, index in enumerate(indices):
        world, plane_shape, extent = _plane_world(native_affine, native.shape, axis, index)
        native_plane = _sample_plane_shaped(native, np.linalg.inv(native_affine), world, plane_shape, order=0)
        trad_plane = _sample_plane_shaped(trad, np.linalg.inv(trad_affine), world, plane_shape, order=1)
        mask = np.isfinite(native_plane) & (native_plane > 0)
        for column, (data, title) in enumerate(((native_plane, "Native (nearest)"), (trad_plane, "Trad (linear)"))):
            axes[row, column].imshow(np.ma.masked_where(~mask, data), cmap="viridis", vmin=0, vmax=2500,
                                     origin="lower", interpolation="nearest", extent=[0, extent[0], 0, extent[1]], aspect="equal")
            axes[row, column].set_title(title if row == 0 else "")
            axes[row, column].set_ylabel(f"{axis_name}={index}")
            axes[row, column].axis("off")
    figure.suptitle("Figure 1 candidate through-plane montage (T1)")
    target = output / "candidates" / f"Figure1_T1_{axis_name}_candidates.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=220, bbox_inches="tight"); plt.close(figure)
    paths.append(str(target))
    return {"plane_axis": axis_name, "candidate_indices": indices, "selection_rule": "foreground-centre unless --plane-index is supplied", "outputs": paths}


def _render_figure1(reference: NativeReference, run: Path, output: Path, axis_name: str, requested_index: int | None, reference_group: int) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    nib, _ = _require_nifti()
    axis = {"x": 0, "y": 1}[axis_name]
    native_images = {p: nib.load(str(reference.figure1_stacks[p])).get_fdata(dtype=np.float32) for p in ("t1", "t2")}
    native_affines = {p: nib.load(str(reference.figure1_stacks[p])).affine for p in ("t1", "t2")}
    trad_images = {p: nib.load(str(run / f"{p.upper()}_3D.nii.gz")).get_fdata(dtype=np.float32) for p in ("t1", "t2")}
    trad_affines = {p: nib.load(str(run / f"{p.upper()}_3D.nii.gz")).affine for p in ("t1", "t2")}
    if native_images["t1"].shape != native_images["t2"].shape or not np.allclose(native_affines["t1"], native_affines["t2"], atol=1e-4):
        raise ValueError("Figure 1 native T1/T2 stacks must share exact geometry.")
    if not 0 <= reference_group < native_images["t1"].shape[2]:
        raise IndexError(f"Figure 1 reference SAX group {reference_group} is unavailable.")
    index = _foreground_center(native_images["t1"], axis) if requested_index is None else requested_index
    if not 0 <= index < native_images["t1"].shape[axis]:
        raise IndexError(f"Plane index {index} is outside native axis {axis_name}.")
    world, plane_shape, extent = _plane_world(native_affines["t1"], native_images["t1"].shape, axis, index)
    output.mkdir(parents=True, exist_ok=True)
    candidates = _render_figure1_candidates(native_images["t1"], native_affines["t1"], trad_images["t1"], trad_affines["t1"], output, axis_name)
    figure, axes = plt.subplots(2, 4, figsize=(13, 7), gridspec_kw={"width_ratios": [1, 1, 1, 0.055]})
    planes: dict[str, dict[str, np.ndarray]] = {}
    for row, (parameter, key, limits) in enumerate((("T1", "t1", (0, 2500)), ("T2", "t2", (0, 200)))):
        native = _sample_plane_shaped(native_images[key], np.linalg.inv(native_affines[key]), world, plane_shape, order=0)
        trad = _sample_plane_shaped(trad_images[key], np.linalg.inv(trad_affines[key]), world, plane_shape, order=1)
        mask = np.isfinite(native) & (native > 0)
        native = np.ma.masked_where(~mask, native)
        trad = np.ma.masked_where(~mask, trad)
        source = native_images[key][:, :, reference_group].T
        axes[row, 0].imshow(np.ma.masked_where(~np.isfinite(source) | (source <= 0), source), cmap="viridis", vmin=limits[0], vmax=limits[1], origin="lower", aspect="equal")
        if axis == 0:
            axes[row, 0].axvline(index, color="white", linewidth=0.8)
        else:
            axes[row, 0].axhline(index, color="white", linewidth=0.8)
        axes[row, 0].set_title("Native SAX + cut line" if row == 0 else "")
        axes[row, 0].set_ylabel(parameter)
        for column, data, title, interpolation in ((1, native, "Native through-plane (nearest)", "nearest"), (2, trad, "Trad 1-mm through-plane (linear)", "nearest")):
            image = axes[row, column].imshow(data, cmap="viridis", vmin=limits[0], vmax=limits[1], origin="lower", interpolation=interpolation, extent=[0, extent[0], 0, extent[1]], aspect="equal")
            axes[row, column].set_title(title if row == 0 else "")
            axes[row, column].axis("off")
        axes[row, 0].axis("off")
        figure.colorbar(image, cax=axes[row, 3], label=f"{parameter} (ms)")
        planes[parameter] = {"native": np.asarray(native), "trad": np.asarray(trad)}
    figure.suptitle(f"{reference.manifest['subject_id']}: native thick-slice versus Trad through-plane visualization")
    png = output / "Figure1_native_vs_trad_through_plane.png"
    figure.savefig(png, dpi=320, bbox_inches="tight"); figure.savefig(png.with_suffix(".pdf"), bbox_inches="tight"); plt.close(figure)
    metadata = {"plane_axis": axis_name, "plane_index": index, "reference_sax_group": reference_group,
                "world_plane_affine_source": str(reference.figure1_stacks["t1"]), "world_point_count": int(world.shape[0]),
                "plane_shape": list(plane_shape), "physical_extent_mm": list(extent), "same_world_plane_check": "PASS",
                "native_interpolation": "nearest-neighbor", "trad_interpolation": "linear", "display_ranges_ms": {"T1": [0, 2500], "T2": [0, 200]},
                "png": str(png), "pdf": str(png.with_suffix(".pdf")), "candidates": candidates}
    _write_json(metadata, output / "figure1_metadata.json")
    return metadata


def _git_value(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE"


def run(args: argparse.Namespace) -> Path | None:
    run_root = Path(args.trad_run).expanduser().resolve()
    prepared_root = Path(args.prepared_root).expanduser().resolve()
    preprocessed_root = Path(args.preprocessed_root).expanduser().resolve() if args.preprocessed_root else prepared_root.parent / "Code_10w_preprocessed"
    reference = load_verified_reference(args.native_reference_root, args.subject_id, preprocessed_root)
    for filename in ("T1_3D.nii.gz", "T2_3D.nii.gz", "experiment_manifest.json"):
        if not (run_root / filename).is_file():
            raise FileNotFoundError(run_root / filename)
    per_slice, per_stack, agreement_summary, loaded = _quantitative_metrics(reference, run_root)
    signal_rows, signal_summary = _signal_metrics(run_root, args.subject_id)
    selected = _select_groups(loaded, _parse_selected_groups(args.selected_group))
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_PASS", "selected_groups": selected, "quantitative_rows": len(per_slice), "signal_rows": len(signal_rows)}, indent=2))
        return None
    output = Path(args.output).expanduser().resolve() if args.output else run_root / "evaluation_scmr"
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty evaluation output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    figure1 = _render_figure1(reference, run_root, output / "figure1", args.plane_axis, args.plane_index, args.figure1_reference_group if args.figure1_reference_group is not None else selected["sax"])
    figure2 = _render_figure2(reference, loaded, selected, output / "figure2")
    _write_csv(per_slice, output / "metrics" / "quantitative_agreement_per_slice.csv")
    _write_csv(per_stack, output / "metrics" / "quantitative_agreement_per_stack.csv")
    _write_json(agreement_summary, output / "metrics" / "quantitative_agreement_summary.json")
    _write_csv(signal_rows, output / "metrics" / "signal_reprojection_metrics.csv")
    _write_json(signal_summary, output / "metrics" / "signal_reprojection_summary.json")
    repo = Path(__file__).resolve().parents[3]
    manifest = {"status": "COMPLETE", "subject_id": args.subject_id, "code_10w_git_commit": _git_value(repo, "rev-parse", "HEAD"),
                "git_dirty_status": _git_value(repo, "status", "--short"), "legacy_2d_fit_first_source_path": str(Path(args.legacy_source).resolve()),
                "legacy_files_inspected_and_adapted": LEGACY_FILES, "native_reference": {"root": str(reference.root), "manifest": str(reference.manifest_path), "same_mind_mppca_provenance_verified": True, "sha256": reference.file_hashes},
                "trad_run": str(run_root), "trad_experiment_manifest_sha256": sha256(run_root / "experiment_manifest.json"),
                "prepared_observations_sha256": {stack: sha256(prepared_root / args.subject_id / stack / "observations.npz") for stack in STACKS},
                "figure1": figure1, "figure2": figure2, "metrics": {"quantitative": "bias, MAE, RMSE, range-NRMSE, Pearson r, NCC, ROI-bounded SSIM", "signal": "RMSE, range-NRMSE, MAE, NCC"},
                "figure3": "SKIPPED / ROI_PENDING: no aligned myocardial/AHA masks", "command": sys.argv, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    _write_json(manifest, output / "evaluation_manifest.json")
    (output / "README.md").write_text("# SCMR Figure 1/2 evaluation\n\nFigure 1 shows through-plane continuity only. Figure 2 and quantitative metrics report consistency with native 2-D dictionary maps, not ground-truth accuracy. Figure 3 is planned, pending aligned myocardial/AHA masks.\n", encoding="utf-8")
    return output


def _parse_selected_groups(values: list[str] | None) -> dict[str, int] | None:
    if not values:
        return None
    result: dict[str, int] = {}
    for value in values:
        stack, separator, group = value.partition("=")
        if separator != "=" or stack not in STACKS:
            raise ValueError("--selected-group expects sax=N, 2ch=N, or 4ch=N")
        result[stack] = int(group)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--preprocessed-root")
    parser.add_argument("--trad-run", required=True)
    parser.add_argument("--native-reference-root", required=True)
    parser.add_argument("--output")
    parser.add_argument("--legacy-source", default="/home/universe/SVR/multimap_postprogramming/MultiMapCode/method_repositories/2D_fit_first")
    parser.add_argument("--plane-axis", choices=("x", "y"), default="x")
    parser.add_argument("--plane-index", type=int)
    parser.add_argument("--figure1-reference-group", type=int)
    parser.add_argument("--selected-group", action="append")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args)
    except (FileNotFoundError, ValueError, IndexError) as exc:
        parser.error(str(exc))
    if result:
        print(result)


if __name__ == "__main__":
    main()
