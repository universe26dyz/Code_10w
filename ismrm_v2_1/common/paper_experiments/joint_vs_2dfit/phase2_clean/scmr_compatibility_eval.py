"""Auxiliary, explicit-mask SCMR compatibility analysis; never a primary endpoint."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import sha256, write_immutable_json
from .evaluation import _world_sample
from .l100_mask_paired_eval import metrics_on_exact_support
from .visualization import DISPLAY_RANGES_MS, colorbars


PARAMETERS = {"t1": "T1", "t2": "T2"}


def _record(path: str | Path) -> dict[str, str]:
    item = Path(path)
    if not item.is_file():
        raise FileNotFoundError(item)
    return {"path": str(item.resolve()), "sha256": sha256(item)}


def _reference_world(shape: tuple[int, int, int], affine: np.ndarray) -> np.ndarray:
    indices = np.indices(shape, dtype=float)
    homogeneous = np.vstack((indices.reshape(3, -1), np.ones(np.prod(shape))))
    return (np.asarray(affine, float) @ homogeneous).reshape((4, *shape))


def _sample(image: Any, world: np.ndarray, *, order: int) -> tuple[np.ndarray, np.ndarray]:
    return _world_sample(np.asarray(image.get_fdata(), float), image.affine, world, order=order)


def _plot(path: Path, parameter: str, reference: np.ndarray, method_values: Mapping[str, np.ndarray], support: np.ndarray, *, title: str) -> None:
    import matplotlib.pyplot as plt

    cmap, limits = colorbars().mapping[parameter], DISPLAY_RANGES_MS[parameter]
    labels = ("Native reference", *method_values)
    values = (reference, *method_values.values())
    figure = plt.figure(figsize=(3.0 * len(values) + .6, 3.3))
    grid = figure.add_gridspec(1, len(values) + 1, width_ratios=[1.0] * len(values) + [.06], wspace=.12)
    handle = None
    for index, (label, value) in enumerate(zip(labels, values)):
        axis = figure.add_subplot(grid[0, index])
        shown = np.ma.masked_invalid(np.where(support, value, np.nan))
        handle = axis.imshow(shown[:, :, shown.shape[2] // 2].T, origin="lower", cmap=cmap, vmin=limits[0], vmax=limits[1], interpolation="none", aspect="equal")
        axis.set_title(label, fontsize=8); axis.set_xticks([]); axis.set_yticks([])
    figure.colorbar(handle, cax=figure.add_subplot(grid[0, -1]), label="ms; compatibility display scale")
    figure.suptitle(title, fontsize=10); figure.subplots_adjust(top=.83, bottom=.04, left=.02, right=.98)
    figure.savefig(path, dpi=220, bbox_inches="tight"); plt.close(figure)


def run_scmr_compatibility_analysis(config_path: str | Path, output: str | Path) -> Path:
    """Evaluate explicitly supplied existing maps within explicit legacy masks only."""
    import nibabel as nib

    config_path, output = Path(config_path), Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("subject") != "CYJ" or not isinstance(config.get("condition"), str) or not isinstance(config.get("endpoint"), str):
        raise ValueError("SCMR compatibility config requires explicit CYJ subject, condition, and endpoint")
    reference_maps, method_maps, legacy_masks = config.get("reference_maps"), config.get("method_maps"), config.get("legacy_masks")
    if not isinstance(reference_maps, Mapping) or not isinstance(method_maps, Mapping) or not method_maps or not isinstance(legacy_masks, Mapping):
        raise ValueError("SCMR compatibility config requires reference_maps, method_maps, and legacy_masks")
    if any(set(maps) != set(PARAMETERS) for maps in (reference_maps, legacy_masks, *method_maps.values())):
        raise ValueError("all compatibility map and legacy-mask sets must provide exactly t1 and t2")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        metrics: dict[str, Any] = {}
        figures: list[dict[str, Any]] = []
        inputs: dict[str, Any] = {"config": _record(config_path)}
        for parameter, display_parameter in PARAMETERS.items():
            reference_path, mask_path = Path(reference_maps[parameter]), Path(legacy_masks[parameter])
            reference_image, mask_image = nib.load(str(reference_path)), nib.load(str(mask_path))
            reference = np.asarray(reference_image.get_fdata(), float)
            if reference.ndim != 3:
                raise ValueError(f"{parameter}: compatibility reference must be 3D")
            world = _reference_world(reference.shape, reference_image.affine)
            mask_values, mask_valid = _sample(mask_image, world, order=0)
            mask = mask_valid & np.isfinite(mask_values) & (mask_values > .5)
            values: dict[str, np.ndarray] = {}
            valid: dict[str, np.ndarray] = {}
            metrics[display_parameter] = {}
            for method, maps in method_maps.items():
                path = Path(maps[parameter]); sampled, sampled_valid = _sample(nib.load(str(path)), world, order=1)
                support = mask & np.isfinite(reference) & sampled_valid & np.isfinite(sampled)
                values[method], valid[method] = sampled, support
                metrics[display_parameter][method] = metrics_on_exact_support(reference, sampled, support, int(np.count_nonzero(mask & np.isfinite(reference))))
                inputs[f"{method}_{parameter}"] = _record(path)
            filename = f"scmr_compatibility_{display_parameter}.png"
            _plot(staging / filename, display_parameter, reference, values, mask, title=f"{config['subject']} {config['condition']} {config['endpoint']} — SCMR compatibility {display_parameter}")
            figures.append({"parameter": display_parameter, "file": filename, "display_support": "explicit legacy/SCMR mask reprojected nearest-neighbor to reference grid", "map_sampling": "linear", "mask_sampling": "nearest-neighbor", "no_auto_crop": True, "physical_aspect_ratio_preserved": True, "range_ms": list(DISPLAY_RANGES_MS[display_parameter])})
            inputs[f"reference_{parameter}"] = _record(reference_path); inputs[f"legacy_mask_{parameter}"] = _record(mask_path)
        rows = [{"parameter": parameter, "method": method, **result} for parameter, by_method in metrics.items() for method, result in by_method.items()]
        with (staging / "compatibility_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row})); writer.writeheader(); writer.writerows(rows)
        write_immutable_json(staging / "compatibility_metrics.json", metrics)
        write_immutable_json(staging / "compatibility_manifest.json", {"schema": "phase4r_scmr_compatibility/v1", "scope": "auxiliary compatibility analysis only; primary_formal_results_unmodified", "subject": config["subject"], "condition": config["condition"], "endpoint": config["endpoint"], "inputs": inputs, "figures": figures, "metrics_file": "compatibility_metrics.json", "primary_aggregate_summary": "not read or modified"})
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(run_scmr_compatibility_analysis(args.config, args.output))


if __name__ == "__main__":
    main()
