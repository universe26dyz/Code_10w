"""Read-only Phase-4R publication/QC rendering on the current paired support.

This module deliberately retains the current world-coordinate evaluation and
its support definitions.  It imports the legacy SCMR *visual* contract only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from trad.evaluation.scmr.quality_control import (
    DISPLAY_RANGES_MS,
    RESIDUAL_RANGES_MS,
    SCMRColorbars,
    load_legacy_scmr_colorbars,
)

from .evaluation import _world_sample, reproject_volume_to_native_plane
from .fit_first import select_verified_native_cache_groups


R0_SUBJECTS = ("CYJ", "DYZ", "HHZ", "HJL")


@dataclass(frozen=True)
class Panel:
    data: np.ndarray
    mask: np.ndarray


def colorbars() -> SCMRColorbars:
    """Return the exact vendored SCMR LUTs and fixed display limits."""
    return load_legacy_scmr_colorbars()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _crop_bounds(mask: np.ndarray, margin: int = 3) -> tuple[int, int, int, int]:
    rows, cols = np.where(np.asarray(mask, bool))
    if not rows.size:
        raise ValueError("cannot crop an empty paired support")
    return (
        max(0, int(rows.min()) - margin),
        min(mask.shape[0], int(rows.max()) + margin + 1),
        max(0, int(cols.min()) - margin),
        min(mask.shape[1], int(cols.max()) + margin + 1),
    )


def paired_common_support(
    stack: str,
    reference: np.ndarray,
    proposed: np.ndarray,
    fit_first: np.ndarray,
    source_support: np.ndarray,
    *,
    reprojection_valid: np.ndarray | None = None,
    myocardium_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build the evaluator's immutable paired support for one native plane."""
    arrays = (reference, proposed, fit_first, source_support)
    if len({np.asarray(item).shape for item in arrays}) != 1:
        raise ValueError("native comparison arrays must share one plane shape")
    stack = str(stack).lower()
    if stack != "sax" and myocardium_mask is not None:
        raise ValueError("SAX myocardium masks must never be applied to 2CH/4CH")
    support = np.asarray(source_support, bool)
    if myocardium_mask is not None:
        support &= np.asarray(myocardium_mask, bool)
    if reprojection_valid is not None:
        support &= np.asarray(reprojection_valid, bool)
    return support & np.isfinite(reference) & np.isfinite(proposed) & np.isfinite(fit_first)


def native_display_support(
    reference: np.ndarray,
    proposed: np.ndarray,
    fit_first: np.ndarray,
    proposed_valid: np.ndarray,
    fit_first_valid: np.ndarray,
) -> np.ndarray:
    """Return the common displayable native foreground, independently of metrics.

    Formal R0 metrics remain restricted to the common SAX myocardium.  The
    native comparison panels instead show the full valid, non-zero native
    foreground that both reconstructed volumes can be sampled on.
    """
    arrays = (reference, proposed, fit_first, proposed_valid, fit_first_valid)
    if len({np.asarray(item).shape for item in arrays}) != 1:
        raise ValueError("native display arrays must share one plane shape")
    return (
        np.isfinite(reference)
        & (np.asarray(reference) != 0)
        & np.isfinite(proposed)
        & np.isfinite(fit_first)
        & np.asarray(proposed_valid, bool)
        & np.asarray(fit_first_valid, bool)
    )


def final5_paired_display_support(
    reference: np.ndarray,
    proposed: np.ndarray,
    fit_first: np.ndarray,
    primary_native_mask: np.ndarray,
    proposed_valid: np.ndarray,
    fit_first_volume_geometry_valid: np.ndarray,
    fit_first_query_effective_mask: np.ndarray,
) -> np.ndarray:
    """Return Final5's one shared native-plane display support.

    This is deliberately the same finite/validity intersection used for all
    five Proposed-vs-Fit-first panels.  The validity masks are output-sampling
    provenance carried by the native bundles; no support is estimated here.
    """
    arrays = (
        reference, proposed, fit_first, primary_native_mask, proposed_valid,
        fit_first_volume_geometry_valid, fit_first_query_effective_mask,
    )
    if len({np.asarray(item).shape for item in arrays}) != 1:
        raise ValueError("Final5 paired display arrays must share one plane shape")
    return (
        np.asarray(primary_native_mask, bool)
        & np.isfinite(reference)
        & np.asarray(proposed_valid, bool)
        & np.asarray(fit_first_volume_geometry_valid, bool)
        & np.asarray(fit_first_query_effective_mask, bool)
        & np.isfinite(proposed)
        & np.isfinite(fit_first)
    )


def crop_myocardium_contour(
    myocardium_mask: np.ndarray, crop: tuple[int, int, int, int]
) -> np.ndarray:
    """Return a cropped R0 ROI boundary source without masking displayed maps."""
    r0, r1, c0, c1 = crop
    return np.asarray(myocardium_mask, bool)[r0:r1, c0:c1]


def native_panel_specs(parameter: str) -> tuple[dict[str, Any], ...]:
    """Formal native-panel colours: one physical parameter scale for every panel."""
    parameter = parameter.upper()
    if parameter not in DISPLAY_RANGES_MS:
        raise ValueError(f"unsupported parameter: {parameter}")
    cmap, limits = colorbars().mapping[parameter], DISPLAY_RANGES_MS[parameter]
    return (
        {"key": "native", "label": "Native 2-D", "cmap": cmap, "limits": limits},
        {"key": "proposed", "label": "Proposed 3-D → native plane", "cmap": cmap, "limits": limits},
        {"key": "fit_first", "label": "Fit-first 3-D → native plane", "cmap": cmap, "limits": limits},
        {"key": "proposed_error", "label": "|Proposed − Native|", "cmap": cmap, "limits": limits},
        {"key": "fit_first_error", "label": "|Fit-first − Native|", "cmap": cmap, "limits": limits},
    )


def prepare_native_comparison_panels(
    reference: np.ndarray, proposed: np.ndarray, fit_first: np.ndarray, support: np.ndarray
) -> tuple[dict[str, Panel], tuple[int, int, int, int]]:
    """Crop all maps/errors once from the paired support, with the SCMR margin."""
    crop = _crop_bounds(support)
    r0, r1, c0, c1 = crop
    mask = np.asarray(support, bool)[r0:r1, c0:c1]
    values = {
        "native": np.asarray(reference, float)[r0:r1, c0:c1],
        "proposed": np.asarray(proposed, float)[r0:r1, c0:c1],
        "fit_first": np.asarray(fit_first, float)[r0:r1, c0:c1],
        "proposed_error": np.abs(np.asarray(proposed, float) - np.asarray(reference, float))[r0:r1, c0:c1],
        "fit_first_error": np.abs(np.asarray(fit_first, float) - np.asarray(reference, float))[r0:r1, c0:c1],
    }
    return {key: Panel(value, mask.copy()) for key, value in values.items()}, crop


def select_representative_group(records: list[Mapping[str, Any]], override: int | None = None) -> int:
    """Choose a central largest-support group deterministically, or honour an override."""
    if override is not None:
        return int(override)
    usable = [item for item in records if int(item.get("common_support_voxels", item.get("support_voxels", 0))) > 0]
    if not usable:
        raise ValueError("no non-empty groups are available for representative selection")
    ordered = sorted(usable, key=lambda item: int(item["group_index"]))
    maximum = max(int(item.get("common_support_voxels", item.get("support_voxels", 0))) for item in ordered)
    candidates = [item for item in ordered if int(item.get("common_support_voxels", item.get("support_voxels", 0))) == maximum]
    central = (len(ordered) - 1) / 2.0
    return int(min(candidates, key=lambda item: (abs(ordered.index(item) - central), int(item["group_index"]))) ["group_index"])


def sample_same_world_through_plane(
    native: np.ndarray,
    native_affine: np.ndarray,
    proposed: np.ndarray,
    proposed_affine: np.ndarray,
    fit_first: np.ndarray,
    fit_first_affine: np.ndarray,
    *,
    cut_column: int | None = None,
    common_support: np.ndarray | None = None,
) -> dict[str, Any]:
    """Sample a regular 1-mm native-world plane shared by all three volumes."""
    native = np.asarray(native, float)
    if native.ndim != 3:
        raise ValueError("Figure-1 native stack must be a 3-D [row,col,group] array")
    if common_support is not None:
        common_support = np.asarray(common_support, bool)
        if common_support.shape != native.shape:
            raise ValueError("through-plane common support must use the native [row,col,group] lattice")
    column = native.shape[1] // 2 if cut_column is None else int(cut_column)
    if not 0 <= column < native.shape[1]:
        raise ValueError("cut column is outside the native plane")
    def physical_affine(value: np.ndarray, label: str) -> np.ndarray:
        affine = np.asarray(value, float)
        if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
            raise ValueError(f"invalid physical through-plane geometry: {label} affine must be finite 4x4")
        if np.linalg.matrix_rank(affine[:3, :3]) != 3:
            raise ValueError(f"invalid physical through-plane geometry: {label} affine is singular")
        return affine

    affine = physical_affine(native_affine, "native")
    proposed_affine = physical_affine(proposed_affine, "proposed")
    fit_first_affine = physical_affine(fit_first_affine, "fit-first")
    linear = affine[:3, :3]
    row_vector, slice_vector = linear[:, 0], linear[:, 2]
    row_spacing, slice_spacing = np.linalg.norm(row_vector), np.linalg.norm(slice_vector)
    if row_spacing <= 0 or slice_spacing <= 0 or np.linalg.norm(np.cross(row_vector, slice_vector)) <= 1e-8 * row_spacing * slice_spacing:
        raise ValueError("invalid physical through-plane geometry: row and slice axes are degenerate")
    row_basis, slice_basis = row_vector / row_spacing, slice_vector / slice_spacing
    origin = (affine @ np.array((0.0, float(column), 0.0, 1.0)))[:3]
    row_extent = (native.shape[0] - 1) * row_spacing
    slice_extent = (native.shape[2] - 1) * slice_spacing
    grid_spacing = 1.0
    row_coordinates = np.arange(0.0, row_extent + grid_spacing * .5, grid_spacing)
    slice_coordinates = np.arange(0.0, slice_extent + grid_spacing * .5, grid_spacing)
    row_grid, slice_grid = np.meshgrid(row_coordinates, slice_coordinates, indexing="ij")
    world = np.empty((4, *row_grid.shape), dtype=float)
    world[:3] = (origin[:, None, None] + row_basis[:, None, None] * row_grid + slice_basis[:, None, None] * slice_grid)
    world[3] = 1.0
    native_values, native_geometry = _world_sample(native, affine, world, order=0)
    proposed_values, proposed_geometry = _world_sample(proposed, proposed_affine, world, order=1)
    fit_values, fit_geometry = _world_sample(fit_first, fit_first_affine, world, order=1)
    sampled_support = None
    if common_support is not None:
        support_values, support_geometry = _world_sample(common_support.astype(np.uint8), affine, world, order=0)
        sampled_support = (np.asarray(support_geometry, bool) & (np.asarray(support_values) > .5)
                           & np.asarray(native_geometry, bool) & np.asarray(proposed_geometry, bool)
                           & np.asarray(fit_geometry, bool) & np.isfinite(native_values)
                           & np.isfinite(proposed_values) & np.isfinite(fit_values))
    return {
        "world": world, "proposed_world": world, "fit_first_world": world,
        "native": native_values, "proposed": proposed_values, "fit_first": fit_values,
        "common_support": sampled_support,
        "cut_column": column,
        "interpolation": {"native": "nearest", "reconstructions": "linear"},
        "world_plane_origin_mm": origin.tolist(),
        "world_plane_basis_vectors": {"row": row_basis.tolist(), "slice": slice_basis.tolist()},
        "grid_spacing_mm": {"row": grid_spacing, "slice": grid_spacing},
        "physical_extent_mm": {"row": [0.0, float(row_coordinates[-1])], "slice": [0.0, float(slice_coordinates[-1])]},
        "grid_shape": list(row_grid.shape),
    }


def _masked(panel: Panel) -> np.ma.MaskedArray:
    return np.ma.masked_where(~panel.mask, panel.data)


def _write_figure(figure: Any, target: Path) -> dict[str, str]:
    figure.savefig(target, dpi=320, bbox_inches="tight")
    pdf = target.with_suffix(".pdf")
    figure.savefig(pdf, bbox_inches="tight")
    return {"png": str(target), "pdf": str(pdf)}


def render_native_plane_comparison(
    parameter: str, panels: Mapping[str, Panel], output: str | Path, *, title: str,
    metadata: Mapping[str, Any], myocardium_contour: np.ndarray | None = None,
) -> dict[str, str]:
    """Render formal native panels with one parameter colour scale and colourbar."""
    import matplotlib.pyplot as plt

    parameter = parameter.upper()
    if parameter not in DISPLAY_RANGES_MS:
        raise ValueError(f"unsupported parameter: {parameter}")
    target = Path(output)
    if target.exists() or target.with_suffix(".pdf").exists():
        raise FileExistsError(f"refusing overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=False)
    figure, axes = plt.subplots(1, 6, figsize=(13.7, 3.0), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, .07]})
    definitions = native_panel_specs(parameter)
    map_handle = None
    for axis, spec in zip(axes[:5], definitions):
        handle = axis.imshow(_masked(panels[spec["key"]]), cmap=spec["cmap"], vmin=spec["limits"][0], vmax=spec["limits"][1], origin="lower", interpolation="nearest", aspect="equal")
        axis.set_title(spec["label"], fontsize=8)
        axis.axis("off")
        map_handle = handle
    if myocardium_contour is not None:
        contour = np.asarray(myocardium_contour, bool)
        if contour.shape != panels["native"].mask.shape:
            raise ValueError("myocardium contour must match cropped display panels")
        for axis in axes[:3]:
            axis.contour(contour.astype(float), levels=(.5,), colors="white", linewidths=.5, origin="lower")
    figure.colorbar(map_handle, cax=axes[5], label=f"{parameter} (ms)")
    figure.suptitle(title, fontsize=10)
    written = _write_figure(figure, target)
    plt.close(figure)
    metadata_record = dict(metadata)
    metadata_record.update({"outputs": written, "formal_colorbar_count": 1, "myocardium_contour_overlay": myocardium_contour is not None})
    (target.parent / "metadata.json").write_text(json.dumps(metadata_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return written


def render_through_plane_continuity(
    parameter: str,
    native: np.ndarray,
    native_affine: np.ndarray,
    proposed: np.ndarray,
    proposed_affine: np.ndarray,
    fit_first: np.ndarray,
    fit_first_affine: np.ndarray,
    output: str | Path,
    *,
    reference_group: int,
    metadata: Mapping[str, Any],
    common_support: np.ndarray | None = None,
) -> dict[str, str]:
    """Render the same-world-plane through-plane continuity QC figure."""
    import matplotlib.pyplot as plt

    parameter = parameter.upper()
    target = Path(output)
    if target.exists() or target.with_suffix(".pdf").exists():
        raise FileExistsError(f"refusing overwrite: {target}")
    sampled = sample_same_world_through_plane(native, native_affine, proposed, proposed_affine, fit_first, fit_first_affine,
                                               common_support=common_support)
    bars = colorbars()
    figure, axes = plt.subplots(1, 5, figsize=(12, 3), gridspec_kw={"width_ratios": [1, 1, 1, 1, .07]})
    limit = DISPLAY_RANGES_MS[parameter]
    reference = np.asarray(native, float)[:, :, int(reference_group)]
    if common_support is not None:
        reference = np.ma.masked_where(~np.asarray(common_support, bool)[:, :, int(reference_group)], reference)
    h0 = axes[0].imshow(reference, cmap=bars.mapping[parameter], vmin=limit[0], vmax=limit[1], origin="lower", interpolation="nearest", aspect="equal")
    axes[0].axvline(sampled["cut_column"], color="white", linewidth=.75)
    axes[0].set_title("Native SAX + cut line", fontsize=8)
    extent = [*sampled["physical_extent_mm"]["slice"], *sampled["physical_extent_mm"]["row"]]
    for axis, value, label in zip(axes[1:4], (sampled["native"], sampled["proposed"], sampled["fit_first"]), ("Native through-plane", "Proposed 3-D through-plane", "Fit-first 3-D through-plane")):
        if sampled["common_support"] is not None:
            value = np.ma.masked_where(~np.asarray(sampled["common_support"], bool), value)
        axis.imshow(value, cmap=bars.mapping[parameter], vmin=limit[0], vmax=limit[1], origin="lower", interpolation="nearest", aspect="equal", extent=extent)
        axis.set_title(label, fontsize=8)
    for axis in axes[:4]:
        axis.axis("off")
    figure.colorbar(h0, cax=axes[4], label=f"{parameter} (ms)")
    figure.suptitle("Through-plane continuity QC; not native-plane quantitative accuracy", fontsize=10)
    written = _write_figure(figure, target)
    plt.close(figure)
    geometry = {key: sampled[key] for key in ("cut_column", "interpolation", "world_plane_origin_mm", "world_plane_basis_vectors", "grid_spacing_mm", "physical_extent_mm", "grid_shape")}
    metadata_record = dict(metadata)
    metadata_record.update(geometry)
    metadata_record.update({"reference_group": int(reference_group), "outputs": written,
                            "common_support_policy": "one supplied native-lattice common mask sampled nearest-neighbour and intersected with all three world-sample geometry domains" if common_support is not None else None})
    (target.parent / "through_plane_metadata.json").write_text(json.dumps(metadata_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return written


def render_all_slice_montage(
    parameter: str, planes: list[Mapping[str, Any]], output: str | Path, *, title: str, metadata: Mapping[str, Any]
) -> dict[str, str]:
    """Render all non-empty native planes as supplemental/QC rows on paired support."""
    import matplotlib.pyplot as plt

    parameter = parameter.upper()
    target = Path(output)
    if target.exists() or target.with_suffix(".pdf").exists():
        raise FileExistsError(f"refusing overwrite: {target}")
    usable = [plane for plane in planes if int(plane["common_support_voxels"]) > 0]
    if not usable:
        raise ValueError("all-slice montage has no non-empty paired support")
    target.parent.mkdir(parents=True, exist_ok=False)
    figure, axes = plt.subplots(len(usable), 6, figsize=(13.7, max(3, 2.7 * len(usable))), squeeze=False, gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, .07]})
    titles = ("Native 2-D", "Proposed 3-D → native plane", "Fit-first 3-D → native plane", "|Proposed − Native|", "|Fit-first − Native|")
    map_handle = None
    annotations = []
    for row, plane in enumerate(usable):
        display_support = np.asarray(plane.get("display_support", plane["support"]), bool)
        panels, _ = prepare_native_comparison_panels(plane["reference"], plane["proposed"], plane["fit_first"], display_support)
        keys = ("native", "proposed", "fit_first", "proposed_error", "fit_first_error")
        specs = {spec["key"]: spec for spec in native_panel_specs(parameter)}
        for column, key in enumerate(keys):
            spec = specs[key]
            handle = axes[row, column].imshow(_masked(panels[key]), cmap=spec["cmap"], vmin=spec["limits"][0], vmax=spec["limits"][1], origin="lower", interpolation="nearest", aspect="equal")
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(titles[column], fontsize=8)
            map_handle = handle
        support = np.asarray(plane.get("quantitative_support", plane["support"]), bool)
        reference, proposed, fit_first = (np.asarray(plane[key], float)[support] for key in ("reference", "proposed", "fit_first"))
        proposed_rmse, fit_rmse = (float(np.sqrt(np.mean((prediction - reference) ** 2))) for prediction in (proposed, fit_first))
        text = f"group={int(plane['group_index'])}; N={int(support.sum())}; Proposed RMSE={proposed_rmse:.1f}; Fit-first RMSE={fit_rmse:.1f}; ΔRMSE={proposed_rmse-fit_rmse:+.1f}"
        axes[row, 0].text(.01, .01, text, transform=axes[row, 0].transAxes, fontsize=5.4, color="black", va="bottom", ha="left", bbox={"facecolor": "white", "edgecolor": "none", "alpha": .82, "pad": 1.3})
        annotations.append(text)
        axes[row, 5].axis("off")
    figure.colorbar(map_handle, cax=axes[0, 5], label=f"{parameter} (ms)")
    figure.suptitle(title, fontsize=10)
    written = _write_figure(figure, target)
    plt.close(figure)
    metadata_record = dict(metadata)
    metadata_record.update({"purpose": "non-primary all-slice QC", "row_annotations": annotations, "outputs": written})
    (target.parent / "all_slice_metadata.json").write_text(json.dumps(metadata_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return written


def validate_fit_first_scale(run_root: str | Path, parameter: str, normalized: str | Path | None = None, ms: str | Path | None = None) -> dict[str, Any]:
    """Fail closed if a fit-first export is stale or violates the normalized-to-ms contract."""
    import nibabel as nib

    root = Path(run_root)
    result = root / "result.json"
    normalized = Path(normalized) if normalized is not None else root / "volume_normalized.nii.gz"
    ms = Path(ms) if ms is not None else root / "volume_ms.nii.gz"
    required = (result, normalized, ms, root / "model.pt")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("fit-first formal export is incomplete: " + ", ".join(missing))
    payload = json.loads(result.read_text(encoding="utf-8"))
    mean = payload.get("output_intensity_mean")
    if mean != 0:
        if mean == 700:
            raise ValueError("stale mean-700 fit-first output is forbidden")
        raise ValueError(f"fit-first output_intensity_mean must be 0, got {mean!r}")
    parameter = parameter.upper()
    scale = {"T1": 2500.0, "T2": 200.0}.get(parameter)
    if scale is None:
        raise ValueError("parameter must be T1 or T2")
    normalized_data = nib.load(str(normalized)).get_fdata(dtype=np.float32)
    ms_data = nib.load(str(ms)).get_fdata(dtype=np.float32)
    finite = np.isfinite(normalized_data) & np.isfinite(ms_data)
    if not finite.any() or not np.allclose(ms_data[finite], normalized_data[finite] * scale, rtol=1e-4, atol=1e-3):
        raise ValueError(f"fit-first {parameter} ms volume does not satisfy normalized × {scale:g}")
    return {"result_json": str(result), "output_intensity_mean": mean, "normalized": str(normalized), "ms": str(ms), "model": str(root / "model.pt"), "scale_ms": scale}


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing required {label}: {path}")
    return path


def r0_reference_mask_provenance(native_reference: str | Path, myocardium_bundle: str | Path) -> dict[str, str]:
    """Hash actual R0 reference and mask artifacts under unambiguous field names."""
    reference, masks = Path(native_reference), Path(myocardium_bundle)
    files = {
        "native_reference_manifest_path": _require_file(reference / "native_reference_manifest.json", "native reference manifest"),
        "native_reference_sax_npz_path": _require_file(reference / "sax.npz", "native reference sax npz"),
        "myocardium_manifest_path": _require_file(masks / "manifest.json", "myocardium mask manifest"),
        "myocardium_masks_npz_path": _require_file(masks / "sax_myocardium_masks.npz", "myocardium masks npz"),
    }
    result: dict[str, str] = {}
    for name, path in files.items():
        result[name] = str(path.resolve())
        result[name.removesuffix("_path") + "_sha256"] = _sha256(path)
    return result


def preflight_r0_inputs(proposed_root: str | Path, fit_first_root: str | Path, data_root: str | Path) -> dict[str, Any]:
    """Validate every formal R0 input without creating output directories."""
    proposed_root, fit_first_root, data_root = (Path(value) for value in (proposed_root, fit_first_root, data_root))
    subjects: dict[str, dict[str, Any]] = {}
    for subject in R0_SUBJECTS:
        proposed = proposed_root / subject
        proposed_t1 = _require_file(proposed / "T1_3D.nii.gz", "proposed T1")
        proposed_t2 = _require_file(proposed / "T2_3D.nii.gz", "proposed T2")
        fit = fit_first_root / subject / "nesvor"
        fit_t1, fit_t2 = fit / "t1", fit / "t2"
        reference = data_root / "reference_data" / "phase2_clean" / subject / "native_reference_2d"
        masks = data_root / "reference_data" / "phase2_clean" / subject / "myocardium_masks_v1"
        for path, label in ((reference / "sax.npz", "native reference sax npz"), (reference / "sax_T1_stack_ms.nii.gz", "native reference T1 stack"), (reference / "sax_T2_stack_ms.nii.gz", "native reference T2 stack")):
            _require_file(path, label)
        provenance = r0_reference_mask_provenance(reference, masks)
        subjects[subject] = {
            "proposed_t1": str(proposed_t1.resolve()), "proposed_t2": str(proposed_t2.resolve()),
            "fit_first_t1": validate_fit_first_scale(fit_t1, "T1"),
            "fit_first_t2": validate_fit_first_scale(fit_t2, "T2"),
            "native_reference": str(reference.resolve()), "myocardium_bundle": str(masks.resolve()),
            "reference_mask_provenance": provenance,
        }
    return {"subjects": subjects, "status": "PASS", "output_creation": "none"}


def finite_value_summary(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, float)[np.isfinite(values)]
    if not finite.size:
        raise ValueError("range audit received no finite map values")
    return {"n_finite": int(finite.size), "p0": float(np.min(finite)), "p1": float(np.quantile(finite, .01)), "p50": float(np.median(finite)), "p99": float(np.quantile(finite, .99)), "p100": float(np.max(finite))}


def metric_contract_audit() -> dict[str, Any]:
    """Record the deliberate distinction from the historical SCMR aggregator."""
    return {
        "current_phase4r_authoritative_metrics": ["RMSE", "MAE", "bias", "relative_NRMSE", "valid_MAPE", "Pearson_r", "NCC", "coverage", "reference_mean", "prediction_mean"],
        "historical_scmr_agreement_metrics": ["N", "bias", "MAE", "RMSE", "NRMSE", "Pearson_r", "NCC", "SSIM"],
        "differences": [
            "Phase-4R uses its frozen R0 common-SAX-myocardium and R1 full-valid-held-out-plane supports, not historical SCMR aggregation.",
            "Phase-4R aggregates inferential results by subject (and equal-weighted R1 phase fold), not voxel-pooled cohort inference.",
            "Phase-4R SSIM remains explicitly not implemented; the historical implementation is optional bounded-support 2-D SSIM and is not a primary endpoint.",
            "The renderer reuses only legacy visual ranges, palettes, invalid display, crop, and export conventions.",
        ],
    }


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def render_r0(
    *, subject: str, native_reference: str | Path, myocardium_bundle: str | Path,
    proposed_t1: str | Path, proposed_t2: str | Path, fit_first_t1: str | Path, fit_first_t2: str | Path,
    output: str | Path, group: int | None = None, all_slices: bool = False,
) -> Path:
    """Render R0 SAX figures from existing maps; this does not reconstruct or refit."""
    import nibabel as nib

    root = Path(output)
    if root.exists():
        raise FileExistsError(f"refusing overwrite: {root}")
    ref_root, mask_root = Path(native_reference), Path(myocardium_bundle)
    reference_mask_provenance = r0_reference_mask_provenance(ref_root, mask_root)
    reference, masks = np.load(ref_root / "sax.npz"), np.load(mask_root / "sax_myocardium_masks.npz")
    native_images = {"T1": nib.load(str(ref_root / "sax_T1_stack_ms.nii.gz")), "T2": nib.load(str(ref_root / "sax_T2_stack_ms.nii.gz"))}
    proposed_images = {"T1": nib.load(str(proposed_t1)), "T2": nib.load(str(proposed_t2))}
    fit_images = {"T1": nib.load(str(fit_first_t1)), "T2": nib.load(str(fit_first_t2))}
    fit_scale = {
        "T1": validate_fit_first_scale(Path(fit_first_t1).parent, "T1", ms=fit_first_t1),
        "T2": validate_fit_first_scale(Path(fit_first_t2).parent, "T2", ms=fit_first_t2),
    }
    root.mkdir(parents=True)
    selected: dict[str, int] = {}
    records: dict[str, list[dict[str, Any]]] = {}
    outputs: dict[str, Any] = {}
    for parameter, key in (("T1", "t1_ms"), ("T2", "t2_ms")):
        image = native_images[parameter]
        planes: list[dict[str, Any]] = []
        for index in range(reference[key].shape[0]):
            p, p_valid = reproject_volume_to_native_plane(proposed_images[parameter].get_fdata(), proposed_images[parameter].affine, image.affine, reference[key].shape[1:], index)
            f, f_valid = reproject_volume_to_native_plane(fit_images[parameter].get_fdata(), fit_images[parameter].affine, image.affine, reference[key].shape[1:], index)
            quantitative_support = paired_common_support("sax", reference[key][index], p, f, masks["myocardium_full"][index], reprojection_valid=p_valid & f_valid, myocardium_mask=masks["myocardium_full"][index])
            display_support = native_display_support(reference[key][index], p, f, p_valid, f_valid)
            planes.append({"group_index": index, "common_support_voxels": int(quantitative_support.sum()), "reference": reference[key][index], "proposed": p, "fit_first": f, "support": quantitative_support, "quantitative_support": quantitative_support, "display_support": display_support, "myocardium_mask": masks["myocardium_full"][index]})
        selected_group = select_representative_group(planes, group)
        selected[parameter] = selected_group
        chosen = next(item for item in planes if item["group_index"] == selected_group)
        panels, crop = prepare_native_comparison_panels(chosen["reference"], chosen["proposed"], chosen["fit_first"], chosen["display_support"])
        metadata = {"subject": subject, "mode": "R0", "endpoint_semantics": "SAX native-myocardium agreement/consistency; not true 3-D accuracy", "quantitative_support_definition": "SAX_myocardium AND finite(reference) AND both reprojection-valid AND finite(proposed, fit-first)", "display_support_definition": "finite non-zero native foreground AND both reprojection-valid AND finite(proposed, fit-first)", "selected_group": selected_group, "crop_bounds_rc": crop, "interpolation": "linear world-coordinate reprojection", "display_ranges_ms": {parameter: list(DISPLAY_RANGES_MS[parameter])}, "colorbars": colorbars().metadata, "source_git_sha": _git_sha()}
        outputs[parameter] = render_native_plane_comparison(parameter, panels, root / parameter / f"{parameter}_native_plane_comparison.png", title=f"{subject} R0 SAX — {parameter}", metadata=metadata, myocardium_contour=crop_myocardium_contour(chosen["myocardium_mask"], crop))
        through_plane_metadata = dict(metadata)
        through_plane_metadata["purpose"] = "through-plane continuity QC; not native-plane quantitative accuracy"
        outputs[parameter]["through_plane"] = render_through_plane_continuity(
            parameter, image.get_fdata(), image.affine,
            proposed_images[parameter].get_fdata(), proposed_images[parameter].affine,
            fit_images[parameter].get_fdata(), fit_images[parameter].affine,
            root / parameter / f"{parameter}_through_plane_continuity.png", reference_group=selected_group,
            metadata=through_plane_metadata,
        )
        records[parameter] = [{"group_index": item["group_index"], "common_support_voxels": item["common_support_voxels"]} for item in planes]
        if all_slices:
            outputs[parameter]["all_slice_montage"] = render_all_slice_montage(parameter, planes, root / parameter / "all_slices" / f"{parameter}_all_slice_montage.png", title=f"{subject} R0 SAX all-slice QC — {parameter}", metadata=metadata)
    provenance = {"subject": subject, "mode": "R0", "method_paths": {"proposed_t1": str(proposed_t1), "proposed_t2": str(proposed_t2), "fit_first_t1": str(fit_first_t1), "fit_first_t2": str(fit_first_t2)}, "fit_first_scale_contract": fit_scale, "range_unit_audit": {"proposed": {parameter: finite_value_summary(proposed_images[parameter].get_fdata()) for parameter in ("T1", "T2")}, "fit_first": {parameter: finite_value_summary(fit_images[parameter].get_fdata()) for parameter in ("T1", "T2")}, "units": "ms"}, "metric_contract_audit": metric_contract_audit(), **reference_mask_provenance, "source_git_sha": _git_sha(), "selected_groups": selected, "all_groups": records, "outputs": outputs, "colorbars": colorbars().metadata, "endpoint_semantics": "R0 primary region is common SAX myocardium; figures are publication/QC visualization only"}
    (root / "formal_figure_metadata.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return root


def select_verified_heldout_native_cache(report: Mapping[str, Any], stack: str) -> dict[str, Any]:
    """Resolve R1 native maps from the immutable held-out report provenance."""
    stack = stack.lower()
    if stack not in ("2ch", "4ch"):
        raise ValueError("R1 visualization stack must be 2ch or 4ch")
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("held-out quantitative report is missing provenance")
    sparse = provenance.get("sparse_manifest")
    if not isinstance(sparse, Mapping):
        raise ValueError("held-out quantitative report is missing sparse-manifest provenance")
    manifest_path = Path(str(sparse.get("path", "")))
    expected_sha = sparse.get("sha256")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"held-out sparse manifest is unavailable: {manifest_path}")
    if not isinstance(expected_sha, str) or _sha256(manifest_path) != expected_sha:
        raise ValueError("held-out sparse manifest SHA256 mismatch")
    cache_root = Path(str(provenance.get("native_reference_cache_root", "")))
    if not cache_root.is_dir():
        raise FileNotFoundError(f"held-out R0 native cache is unavailable: {cache_root}")
    stacks = report.get("stacks")
    stack_report = stacks.get(stack) if isinstance(stacks, Mapping) else None
    if not isinstance(stack_report, Mapping):
        raise ValueError(f"held-out quantitative report is missing {stack} group provenance")
    groups = stack_report.get("heldout_original_group_ids")
    if not isinstance(groups, list):
        raise ValueError(f"held-out quantitative report is missing {stack} held-out group IDs")
    selected = select_verified_native_cache_groups(cache_root, manifest_path, stack, [int(group) for group in groups])
    if [int(group) for group in np.asarray(selected["group_idx"])] != [int(group) for group in groups]:
        raise ValueError(f"{stack}: verified native cache groups do not match held-out report provenance")
    return selected


def render_r1_from_heldout(
    *, heldout_root: str | Path, proposed_t1: str | Path, proposed_t2: str | Path,
    fit_first_t1: str | Path, fit_first_t2: str | Path, output: str | Path,
    stack: str, group: int | None = None,
) -> Path:
    """Render an existing R1/L100 evaluation-only native-map cache without refitting."""
    import nibabel as nib

    root = Path(output)
    if root.exists():
        raise FileExistsError(f"refusing overwrite: {root}")
    heldout_root = Path(heldout_root)
    report = json.loads((heldout_root / "heldout_quantitative_metrics.json").read_text(encoding="utf-8"))
    stack = stack.lower()
    if stack not in ("2ch", "4ch"):
        raise ValueError("R1 visualization stack must be 2ch or 4ch")
    cache = select_verified_heldout_native_cache(report, stack)
    proposed_images = {"T1": nib.load(str(proposed_t1)), "T2": nib.load(str(proposed_t2))}
    fit_images = {"T1": nib.load(str(fit_first_t1)), "T2": nib.load(str(fit_first_t2))}
    fit_scale = {
        "T1": validate_fit_first_scale(Path(fit_first_t1).parent, "T1", ms=fit_first_t1),
        "T2": validate_fit_first_scale(Path(fit_first_t2).parent, "T2", ms=fit_first_t2),
    }
    root.mkdir(parents=True)
    outputs: dict[str, Any] = {}
    selected: dict[str, int] = {}
    for parameter, key in (("T1", "t1_ms"), ("T2", "t2_ms")):
        records: list[dict[str, Any]] = []
        for index, original_group in enumerate(cache["group_idx"]):
            affine_ras = np.diag([-1.0, -1.0, 1.0, 1.0]) @ np.asarray(cache["affine_lps_rc"][index], float)
            p, p_valid = reproject_volume_to_native_plane(proposed_images[parameter].get_fdata(), proposed_images[parameter].affine, affine_ras, cache[key][index].shape)
            f, f_valid = reproject_volume_to_native_plane(fit_images[parameter].get_fdata(), fit_images[parameter].affine, affine_ras, cache[key][index].shape)
            source_support = np.asarray(cache["source_support"][index], bool)
            quantitative_support = paired_common_support(stack, cache[key][index], p, f, source_support, reprojection_valid=p_valid & f_valid)
            display_support = native_display_support(cache[key][index], p, f, p_valid, f_valid) & source_support
            records.append({"group_index": int(original_group), "common_support_voxels": int(quantitative_support.sum()), "reference": cache[key][index], "proposed": p, "fit_first": f, "support": quantitative_support, "quantitative_support": quantitative_support, "display_support": display_support})
        chosen_id = select_representative_group(records, group)
        selected[parameter] = chosen_id
        chosen = next(item for item in records if item["group_index"] == chosen_id)
        panels, crop = prepare_native_comparison_panels(chosen["reference"], chosen["proposed"], chosen["fit_first"], chosen["display_support"])
        l100 = report.get("condition") == "L100"
        semantics = "full-data LAX native-plane qualitative agreement reference; not R0 primary endpoint" if l100 else "R1 held-out native-plane agreement/generalization; not true 3-D accuracy"
        metadata = {"subject": report["subject"], "mode": "L100" if l100 else "R1", "stack": stack, "condition": report["condition"], "endpoint_semantics": semantics, "quantitative_support_definition": "source foreground AND native dictionary finite AND common world reprojection valid AND finite(reference, proposed, fit-first)", "display_support_definition": "source foreground AND finite non-zero native foreground AND both reprojection-valid AND finite(proposed, fit-first)", "selected_group": chosen_id, "crop_bounds_rc": crop, "interpolation": "linear world-coordinate reprojection", "display_ranges_ms": {parameter: list(DISPLAY_RANGES_MS[parameter])}, "colorbars": colorbars().metadata, "source_git_sha": _git_sha()}
        outputs[parameter] = render_native_plane_comparison(parameter, panels, root / parameter / f"{stack}_{parameter}_native_plane_comparison.png", title=f"{report['subject']} {report['condition']} {stack.upper()} — {parameter}", metadata=metadata)
    (root / "formal_figure_metadata.json").write_text(json.dumps({"subject": report["subject"], "mode": "L100" if report.get("condition") == "L100" else "R1", "heldout_report": {"path": str((heldout_root / "heldout_quantitative_metrics.json").resolve()), "sha256": _sha256(heldout_root / "heldout_quantitative_metrics.json")}, "method_paths": {"proposed_t1": str(proposed_t1), "proposed_t2": str(proposed_t2), "fit_first_t1": str(fit_first_t1), "fit_first_t2": str(fit_first_t2)}, "fit_first_scale_contract": fit_scale, "range_unit_audit": {"proposed": {parameter: finite_value_summary(proposed_images[parameter].get_fdata()) for parameter in ("T1", "T2")}, "fit_first": {parameter: finite_value_summary(fit_images[parameter].get_fdata()) for parameter in ("T1", "T2")}, "units": "ms"}, "selected_groups": selected, "outputs": outputs, "metric_contract_audit": metric_contract_audit(), "endpoint_semantics": "L100 is a full-data LAX reference point" if report.get("condition") == "L100" else "R1 held-out 2CH/4CH evaluation", "colorbars": colorbars().metadata}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("r0", "r1", "r0-preflight"), required=True)
    parser.add_argument("--subject")
    parser.add_argument("--native-reference")
    parser.add_argument("--myocardium-bundle")
    parser.add_argument("--proposed-t1"); parser.add_argument("--proposed-t2")
    parser.add_argument("--fit-first-t1"); parser.add_argument("--fit-first-t2")
    parser.add_argument("--output"); parser.add_argument("--group", type=int); parser.add_argument("--all-slices", action="store_true"); parser.add_argument("--heldout-root"); parser.add_argument("--stack")
    parser.add_argument("--proposed-root"); parser.add_argument("--fit-first-root"); parser.add_argument("--data-root")
    args = parser.parse_args()
    if args.mode == "r0-preflight":
        if not all((args.proposed_root, args.fit_first_root, args.data_root)):
            raise ValueError("R0 preflight requires --proposed-root, --fit-first-root, and --data-root")
        print(json.dumps(preflight_r0_inputs(args.proposed_root, args.fit_first_root, args.data_root), indent=2, sort_keys=True))
    elif args.mode == "r0":
        if not all((args.subject, args.native_reference, args.myocardium_bundle, args.proposed_t1, args.proposed_t2, args.fit_first_t1, args.fit_first_t2, args.output)):
            raise ValueError("R0 requires subject, reference/mask, both method maps, and output")
        values = vars(args).copy()
        for key in ("mode", "heldout_root", "stack", "proposed_root", "fit_first_root", "data_root"):
            values.pop(key)
        print(render_r0(**values))
    else:
        if not all((args.heldout_root, args.stack, args.proposed_t1, args.proposed_t2, args.fit_first_t1, args.fit_first_t2, args.output)):
            raise ValueError("R1 requires heldout root, stack, both method maps, and output")
        values = vars(args).copy()
        for key in ("mode", "native_reference", "myocardium_bundle", "all_slices", "subject", "proposed_root", "fit_first_root", "data_root"):
            values.pop(key)
        print(render_r1_from_heldout(**values))


if __name__ == "__main__":
    main()
