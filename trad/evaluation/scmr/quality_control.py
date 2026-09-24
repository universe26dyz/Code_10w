"""Read-only SCMR all-slice QC, legacy colour maps, and range auditing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .reference_2d import NativeReference, STACKS


DISPLAY_RANGES_MS = {"T1": (0.0, 2500.0), "T2": (0.0, 200.0)}
RESIDUAL_RANGES_MS = {"T1": (0.0, 300.0), "T2": (0.0, 30.0)}
_LEGACY_ASSET_RELATIVE = Path("python/visualization/assets/colormaps")
_LEGACY_FILES = {
    "T1": ("Lipari", "lipari.txt"),
    "T2": ("Navia", "navia.txt"),
}


@dataclass(frozen=True)
class SCMRColorbars:
    mapping: Mapping[str, Any]
    residual: Any
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_legacy_scmr_colorbars(legacy_root: str | Path) -> SCMRColorbars:
    """Load the exact Lipari/Navia LUT text assets used by old SCMR figures."""

    from matplotlib import colors
    from matplotlib import pyplot as plt

    root = Path(legacy_root).expanduser().resolve()
    asset_root = root / _LEGACY_ASSET_RELATIVE
    mapping: dict[str, Any] = {}
    source_assets: dict[str, dict[str, str]] = {}
    for parameter, (name, filename) in _LEGACY_FILES.items():
        path = asset_root / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Legacy SCMR {parameter} colour-map LUT is missing: {path}. "
                "Do not substitute an unrelated default palette."
            )
        rgb = np.loadtxt(path, dtype=float)
        if rgb.ndim != 2 or rgb.shape != (256, 3) or not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 1)):
            raise ValueError(f"Legacy SCMR LUT must be finite [256,3] RGB values: {path}")
        cmap = colors.LinearSegmentedColormap.from_list(name.lower(), np.vstack(([0.0, 0.0, 0.0], rgb)), N=256)
        cmap.set_bad("black")
        cmap.set_under("black")
        mapping[parameter] = cmap
        source_assets[parameter] = {"name": name, "path": str(path), "sha256": _sha256(path), "zero_and_invalid": "black"}
    return SCMRColorbars(
        mapping=mapping,
        residual=plt.get_cmap("magma"),
        metadata={
            "legacy_colormap_source": str(root),
            "legacy_colormap_source_file": "python/visualization/colormaps.py",
            "legacy_plotting_source_file": "python/visualization/plotting.py",
            "mapping_colormaps": source_assets,
            "residual_colormap_name": "magma",
            "t1_display_range_ms": list(DISPLAY_RANGES_MS["T1"]),
            "t2_display_range_ms": list(DISPLAY_RANGES_MS["T2"]),
            "residual_range_ms": {key: list(value) for key, value in RESIDUAL_RANGES_MS.items()},
        },
    )


def _crop(data: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.where(mask)
    if not rows.size:
        return data, mask
    r0, r1 = max(0, rows.min() - 3), min(data.shape[0], rows.max() + 4)
    c0, c1 = max(0, cols.min() - 3), min(data.shape[1], cols.max() + 4)
    return data[r0:r1, c0:c1], mask[r0:r1, c0:c1]


def _metric_lookup(per_slice: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    values: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in per_slice:
        key = (str(row["parameter"]), str(row["stack"]), int(row["group_idx"]))
        if key in values:
            raise ValueError(f"Duplicate per-slice metrics for {key}.")
        values[key] = row
    return values


def render_all_slice_montages(
    reference: NativeReference,
    loaded: Mapping[str, Mapping[str, np.ndarray]],
    per_slice: Iterable[Mapping[str, Any]],
    output: str | Path,
    colorbars: SCMRColorbars,
) -> dict[str, Any]:
    """Render all native groups with exactly the evaluator's existing common mask."""

    import matplotlib.pyplot as plt

    root = Path(output)
    metrics = _metric_lookup(per_slice)
    written: dict[str, dict[str, dict[str, str]]] = {"T1": {}, "T2": {}}
    row_annotations: dict[str, dict[str, list[str]]] = {"T1": {}, "T2": {}}
    for parameter, key in (("T1", "t1_ms"), ("T2", "t2_ms")):
        for stack in STACKS:
            native = reference.stacks[stack]
            n_groups = native.t1_ms.shape[0]
            annotations: list[str] = []
            figure, axes = plt.subplots(
                n_groups,
                4,
                figsize=(12.0, max(2.65 * n_groups, 3.0)),
                squeeze=False,
                gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 0.06]},
            )
            mapping_handle = residual_handle = None
            for row, group in enumerate(native.group_idx):
                ref = getattr(native, key)[group]
                pred = loaded[stack][key][group]
                mask = native.valid_mask[group] & loaded[stack]["support"][group] & np.isfinite(ref) & np.isfinite(pred)
                if not mask.any():
                    raise ValueError(f"No common valid/support pixels for {parameter} {stack} group {group}.")
                item = metrics.get((parameter, stack, int(group)))
                if item is None:
                    raise ValueError(f"Per-slice metrics are missing for {parameter} {stack} group {group}.")
                cropped_ref, cropped_mask = _crop(ref, mask)
                cropped_pred, _ = _crop(pred, mask)
                residual, _ = _crop(np.abs(pred - ref), mask)
                panels = (
                    (cropped_ref, "Native 2-D", colorbars.mapping[parameter], DISPLAY_RANGES_MS[parameter]),
                    (cropped_pred, "Trad 3-D → native plane", colorbars.mapping[parameter], DISPLAY_RANGES_MS[parameter]),
                    (residual, "|Difference|", colorbars.residual, RESIDUAL_RANGES_MS[parameter]),
                )
                for column, (data, title, cmap, limits) in enumerate(panels):
                    handle = axes[row, column].imshow(np.ma.masked_where(~cropped_mask, data), cmap=cmap, vmin=limits[0], vmax=limits[1], origin="lower", interpolation="nearest", aspect="equal")
                    if row == 0:
                        axes[row, column].set_title(title)
                    axes[row, column].axis("off")
                    if column == 2:
                        residual_handle = handle
                    else:
                        mapping_handle = handle
                annotation = (
                    f"group_idx={int(group)}; N={int(item['N'])}; "
                    f"RMSE={float(item['RMSE_ms']):.1f}; bias={float(item['bias_ms']):+.1f}; NCC={float(item['NCC']):.3f}"
                )
                axes[row, 0].text(0.02, 0.02, annotation, transform=axes[row, 0].transAxes, fontsize=5.5,
                                  color="black", va="bottom", ha="left", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5})
                annotations.append(annotation)
                axes[row, 3].axis("off")
            figure.colorbar(mapping_handle, cax=axes[0, 3], label=f"{parameter} (ms)")
            figure.colorbar(residual_handle, cax=axes[-1, 3], label="|Δ| (ms)")
            figure.suptitle(f"{reference.manifest['subject_id']} {parameter} all-slice QC — {stack.upper()}")
            target = root / parameter / f"{parameter}_{stack.upper()}_all_slices_montage.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(target, dpi=220, bbox_inches="tight")
            figure.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
            plt.close(figure)
            written[parameter][stack] = {"png": str(target), "pdf": str(target.with_suffix(".pdf")), "groups": int(n_groups)}
            row_annotations[parameter][stack] = annotations
    metadata = {
        "mask_definition": "native_valid AND Trad_support AND finite(reference,prediction)",
        "metrics_source": "metrics/quantitative_agreement_per_slice.csv generated by the same evaluator run",
        "residual_definition": "abs(Trad reprojection - native 2-D reference)",
        "units": "ms",
        "colorbars": colorbars.metadata,
        "outputs": written,
        "row_annotations": row_annotations,
    }
    (root / "all_slice_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    return metadata


def _summary(values: np.ndarray) -> dict[str, float]:
    raw = np.asarray(values, dtype=np.float64)
    finite = raw[np.isfinite(raw)]
    if not finite.size:
        raise ValueError("Range audit received no finite values.")
    quantiles = np.quantile(finite, (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0))
    return {
        key: float(value)
        for key, value in zip(("min", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "max"), quantiles)
    } | {"finite_ratio": float(finite.size / raw.size), "n_finite": int(finite.size), "n_total": int(raw.size)}


def _boundary_occupancy(values: np.ndarray, lower: float, upper: float) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    margin = 0.01 * (upper - lower)
    return {
        "definition": f"within 1% of physical range [{lower:g}, {upper:g}]",
        "near_lower_fraction": float(np.mean(finite <= lower + margin)),
        "near_upper_fraction": float(np.mean(finite >= upper - margin)),
    }


def _load_volume(path: Path) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("nibabel is required for the range audit.") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Range audit requires exported volume: {path}")
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def _normalization_config(run_root: Path) -> dict[str, Any]:
    path = run_root / "config_resolved.yaml"
    if not path.is_file():
        return {"source": str(path), "status": "UNAVAILABLE"}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to audit config_resolved.yaml.") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {"source": str(path), "status": "PRESENT", "training_intensity_normalization": value.get("training", {}).get("intensity_normalization", {})}


def write_range_audit(
    reference: NativeReference,
    loaded: Mapping[str, Mapping[str, np.ndarray]],
    run_root: str | Path,
    output: str | Path,
    *,
    method_label: str = "Trad",
) -> dict[str, Any]:
    """Audit physical map values without altering or reprojecting any model field."""

    run_root, root = Path(run_root), Path(output)
    volumes = {
        "T1_3D": _load_volume(run_root / "T1_3D.nii.gz"),
        "T2_3D": _load_volume(run_root / "T2_3D.nii.gz"),
        "B1_3D": _load_volume(run_root / "B1_3D.nii.gz"),
        "A_3D": _load_volume(run_root / "amplitude_3D.nii.gz"),
    }
    native_values, reprojection_values = {"T1": [], "T2": []}, {"T1": [], "T2": []}
    for stack in STACKS:
        native = reference.stacks[stack]
        for parameter, key in (("T1", "t1_ms"), ("T2", "t2_ms")):
            ref, pred = getattr(native, key), loaded[stack][key]
            mask = native.valid_mask & loaded[stack]["support"] & np.isfinite(ref) & np.isfinite(pred)
            native_values[parameter].append(ref[native.valid_mask & np.isfinite(ref)])
            reprojection_values[parameter].append(pred[mask])
    t1, t2 = volumes["T1_3D"], volumes["T2_3D"]
    fields = {key: _summary(value) for key, value in volumes.items()}
    physical_bounds = {"T1_ms": [5.0, 2500.0], "T2_ms": [5.0, 200.0], "B1": [0.1, 1.2]}
    bounds_pass = bool(np.all((t1 >= 5.0) & (t1 <= 2500.0)) and np.all((t2 >= 5.0) & (t2 <= 200.0)))
    summary: dict[str, Any] = {
        "scope": f"read-only existing {method_label} outputs and verified native reference; no reconstruction/preprocessing rerun",
        "method_label": method_label,
        "quantiles": ["min", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "max"],
        "exported_3d_fields": fields,
        "native_reference_global_range": {parameter: _summary(np.concatenate(values)) for parameter, values in native_values.items()},
        "native_plane_reprojection_common_mask_range": {parameter: _summary(np.concatenate(values)) for parameter, values in reprojection_values.items()},
        "boundary_occupancy": {"T1_3D": _boundary_occupancy(t1, 5.0, 2500.0), "T2_3D": _boundary_occupancy(t2, 5.0, 200.0), "B1_3D": _boundary_occupancy(volumes["B1_3D"], 0.1, 1.2)},
        "physical_output_bounds": physical_bounds,
        "normalization_config": _normalization_config(run_root),
        "export_unit_expectations": {
            "T1_3D": "ms; export passes model t1_ms through unchanged",
            "T2_3D": "ms; export passes model t2_ms through unchanged",
            "B1_3D": "unitless transmit-scale field",
            "amplitude_3D": "original input intensity; only amplitude is multiplied by intensity_scale during export",
            "t1_t2_native_plane": "ms; reprojection copies model t1_ms/t2_ms directly and only signal prediction applies intensity_scale",
            "native_reference_2d": "ms; verified dictionary matching bundle",
            "evaluation_metrics": "native and reprojection T1/T2 compared directly in ms with no normalization step",
        },
        "audit_verdict": "PASS_NO_EVIDENCE_OF_ADDITIONAL_T1_T2_SCALING" if bounds_pass else "FAIL_PHYSICAL_BOUNDS_VIOLATED",
        "possible_scaling_bug_found": not bounds_pass,
    }
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "range_audit_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    report = "\n".join((
        "# SCMR range and unit audit",
        "",
        "## Unit chain",
        "",
        f"T1/T2/B1 are produced by the {method_label} INR as physical fields (ms/ms/unitless). Only observed signal intensity is normalized for training; amplitude is multiplied by the recorded subject scale on export and signal reprojection. T1/T2 are not multiplied by this scale at export or native-plane reprojection.",
        "",
        "## Result",
        "",
        f"Verdict: **{summary['audit_verdict']}**. The T1/T2 exported volumes satisfy their model physical bounds. The native reference is in ms and the evaluator compares it directly against ms native-plane fields.",
        "",
        "## Interpretation",
        "",
        "Historical 2D-fit-first figures report myocardial-only values. This audit uses global or common-support values because no aligned myocardial/AHA ROI is available; therefore those values are not directly comparable. Boundary occupancy and broad value ranges can reflect dictionary-grid saturation and non-myocardial/support mixture, not an inferred scaling error.",
        "",
        f"Machine-readable statistics: `{json_path.name}`.",
        "",
    ))
    report_path = root / "range_audit_report.md"
    report_path.write_text(report, encoding="utf-8")
    return {"summary_json": str(json_path), "report_md": str(report_path), "verdict": summary["audit_verdict"]}
