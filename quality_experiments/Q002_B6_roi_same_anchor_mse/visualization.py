"""Descriptive, post-training B6-style comparison visualizations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from trad.evaluation.scmr.quality_control import DISPLAY_RANGES_MS, RESIDUAL_RANGES_MS, load_legacy_scmr_colorbars


def deterministic_representatives(identities: Iterable[tuple[str, int, int, int]], *, limit: int) -> list[tuple[str, int, int, int]]:
    """Select by declared identity only, never by reference-quality or metric values."""

    return sorted(set(identities))[:int(limit)]


def visualization_display_contract() -> dict[str, object]:
    """Shared display policy recorded with every formal visualization bundle."""

    return {
        "selection_rule": "lexicographic_stack_group_anchor_identity",
        "per_method_autoscaling": False,
        "map_display_ranges": {key: list(value) for key, value in DISPLAY_RANGES_MS.items()},
        "absolute_residual_ranges": {key: list(value) for key, value in RESIDUAL_RANGES_MS.items()},
        "map_colormaps": {"T1": "Lipari", "T2": "Navia"},
        "residual_colormap": "magma",
        "orientation": "native_reference_orientation",
    }


def render_map_montages(records: Iterable[dict[str, Any]], output_dir: str | Path) -> list[Path]:
    """Render all native groups with one parameter range per method and fixed residual ranges."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    contract = visualization_display_contract()
    colorbars = load_legacy_scmr_colorbars()
    outputs: list[Path] = []
    for record in records:
        parameter = str(record["parameter"])
        reference = np.asarray(record["reference"])
        predictions = record["predictions"]
        methods = ("FrozenMLP_B6", "Q002_S640", "Q003_S640_6000", "Q003_S640_10000")
        figure, axes = plt.subplots(2, 5, figsize=(15, 6))
        map_range, residual_range = contract["map_display_ranges"][parameter], contract["absolute_residual_ranges"][parameter]
        axes[0, 0].imshow(reference, cmap=colorbars.mapping[parameter], vmin=map_range[0], vmax=map_range[1], origin="lower"); axes[0, 0].set_title("Reference")
        axes[1, 0].axis("off")
        for column, method in enumerate(methods, start=1):
            axes[0, column].imshow(predictions[method], cmap=colorbars.mapping[parameter], vmin=map_range[0], vmax=map_range[1], origin="lower"); axes[0, column].set_title(method)
            axes[1, column].imshow(np.abs(predictions[method] - reference), cmap=colorbars.residual, vmin=residual_range[0], vmax=residual_range[1], origin="lower"); axes[1, column].set_title("|residual|")
        for axis in axes.flat: axis.set_axis_off()
        figure.suptitle(f"{parameter} {record['stack']} group {record['group_idx']}")
        figure.tight_layout()
        path = output / f"{parameter}_{record['stack']}_group_{int(record['group_idx']):03d}"
        figure.savefig(path.with_suffix(".png"), dpi=160); figure.savefig(path.with_suffix(".pdf")); plt.close(figure)
        outputs.extend((path.with_suffix(".png"), path.with_suffix(".pdf")))
    return outputs


def render_fingerprint_curves(records: Iterable[dict[str, Any]], output_dir: str | Path) -> list[Path]:
    """Render deterministic pixel identities supplied by the strict common-support evaluator."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    outputs: list[Path] = []
    for record in records:
        figure, axis = plt.subplots(figsize=(7, 4))
        for label, values in record["vectors"].items(): axis.plot(range(10), values, marker="o", label=label)
        axis.set(xlabel="weight index", ylabel="signal", title=f"{record['stack']} group {record['group_idx']} anchor {record['row_col']}")
        axis.legend(fontsize=8); figure.tight_layout()
        path = output / f"fingerprint_{record['stack']}_group_{int(record['group_idx']):03d}_r{record['row_col'][0]}_c{record['row_col'][1]}"
        figure.savefig(path.with_suffix(".png"), dpi=160); figure.savefig(path.with_suffix(".pdf")); plt.close(figure)
        outputs.extend((path.with_suffix(".png"), path.with_suffix(".pdf")))
    return outputs


def write_visualization_manifest(output_dir: str | Path, manifest: dict[str, Any]) -> Path:
    path = Path(output_dir) / "visualization_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
