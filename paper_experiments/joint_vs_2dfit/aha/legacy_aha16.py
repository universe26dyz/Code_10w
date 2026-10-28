"""Pure AHA16 rules ported from the historical CYJ evaluation implementation.

The code below intentionally retains the historical basal/mid/apical segment
numbering, anterior/inferior orientation decision, core-ROI erosion, sample
SD statistic, and single-bullseye geometry.  It contains no historical maps,
predictions, statistics, or output discovery.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


LEVELS: dict[str, tuple[int, int]] = {
    "basal": (6, 1),
    "mid": (6, 7),
    "apical": (4, 13),
}
DISPLAY_RANGES = {"t1": (0.0, 2500.0), "t2": (0.0, 200.0)}
PRIMARY_MYOCARDIUM_ROI = "myocardium_core_1px"


def build_filled_masks(outer: np.ndarray, inner: np.ndarray, spacing_rc_mm: tuple[float, float], *, erosion_mm: float = 1.0) -> dict[str, np.ndarray]:
    """Build historical filled and 1-mm-core myocardium masks from manual ROI.

    The operation is purely anatomical; it is never a prediction, training,
    reconstruction, or formal-metric mask.
    """
    from scipy.ndimage import binary_erosion, distance_transform_edt

    outer, inner = np.asarray(outer, bool), np.asarray(inner, bool)
    if outer.shape != inner.shape or outer.ndim != 2:
        raise ValueError("manual outer/inner ROI must be same-shape 2-D arrays")
    if not np.all(np.isfinite(spacing_rc_mm)) or min(spacing_rc_mm) <= 0:
        raise ValueError("manual ROI pixel spacing must be finite and positive")
    myocardium = outer & ~inner
    # The historical implementation uses a strict comparison: pixels exactly
    # one millimetre from a boundary are not part of the core.
    core_legacy = myocardium & (distance_transform_edt(myocardium, sampling=spacing_rc_mm) > erosion_mm)
    core_1px = binary_erosion(myocardium, structure=np.ones((3, 3), bool), iterations=1, border_value=0)
    return {
        "myocardium": myocardium,
        "myocardium_core": core_legacy,
        "myocardium_core_legacy": core_legacy,
        "myocardium_core_1px": np.asarray(core_1px, bool),
    }


def sectorize(mask: np.ndarray, anterior: tuple[float, float] | np.ndarray, inferior: tuple[float, float] | np.ndarray,
              n_sectors: int, first_segment: int = 1) -> np.ndarray:
    """Historical anterior-anchored AHA sector assignment, unchanged in logic."""
    mask = np.asarray(mask, bool)
    if mask.ndim != 2 or not mask.any() or n_sectors not in (4, 6):
        raise ValueError("sectorize requires a non-empty 2-D mask and 4 or 6 sectors")
    anterior, inferior = np.asarray(anterior, float), np.asarray(inferior, float)
    if anterior.shape != (2,) or inferior.shape != (2,) or not np.isfinite(anterior).all() or not np.isfinite(inferior).all():
        raise ValueError("anterior and inferior landmarks must be finite (x, y) pairs")
    y, x = np.indices(mask.shape)
    yy, xx = np.where(mask)
    center = np.array([xx.mean(), yy.mean()])
    a0 = np.arctan2(anterior[1] - center[1], anterior[0] - center[0])
    ai = np.arctan2(inferior[1] - center[1], inferior[0] - center[0])
    plus, minus = (ai - a0) % (2 * np.pi), (a0 - ai) % (2 * np.pi)
    sign = 1.0 if abs(plus - np.pi) <= abs(minus - np.pi) else -1.0
    angle = (sign * (np.arctan2(y - center[1], x - center[0]) - a0) + np.pi / n_sectors) % (2 * np.pi)
    local = np.floor(angle / (2 * np.pi / n_sectors)).astype(int)
    labels = np.zeros(mask.shape, np.uint8)
    labels[mask] = first_segment + local[mask]
    return labels


def aha16_labels(level_masks: Mapping[str, np.ndarray], landmarks: Mapping[str, tuple[tuple[float, float], tuple[float, float]]]) -> dict[str, np.ndarray]:
    """Return basal 1--6, mid 7--12, and apical 13--16 labels only."""
    if set(level_masks) != set(LEVELS) or set(landmarks) != set(LEVELS):
        raise ValueError("AHA16 requires explicit basal, mid, and apical masks and landmarks")
    return {
        level: sectorize(level_masks[level], landmarks[level][0], landmarks[level][1], n_sectors=n_segments, first_segment=first)
        for level, (n_segments, first) in LEVELS.items()
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"mean": float("nan"), "sd": float("nan")}
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)) if values.size > 1 else 0.0}


def segment_statistics(reference: np.ndarray, prediction: np.ndarray, labels: np.ndarray, support: np.ndarray, *,
                       segments: Iterable[int], parameter: str | None = None) -> dict[int, dict[str, Any]]:
    """Return historical segment summaries plus the shared metric contract.

    The anatomical labels, shared support, means/SDs, and signed differences
    remain unchanged.  ``parameter`` enables fixed-range PSNR and masked 2-D
    SSIM; callers that omit it retain the historical summary-only behavior.
    """
    reference, prediction = np.asarray(reference, float), np.asarray(prediction, float)
    labels, support = np.asarray(labels, int), np.asarray(support, bool)
    if not (reference.shape == prediction.shape == labels.shape == support.shape) or reference.ndim != 2:
        raise ValueError("AHA reference, prediction, labels, and support require one same-shape plane")
    result: dict[int, dict[str, Any]] = {}
    for segment in segments:
        mask = (labels == int(segment)) & support & np.isfinite(reference) & np.isfinite(prediction)
        ref, pred = _distribution(reference[mask]), _distribution(prediction[mask])
        record: dict[str, Any] = {
            "n": int(mask.sum()), "reference_mean": ref["mean"], "reference_sd": ref["sd"],
            "prediction_mean": pred["mean"], "prediction_sd": pred["sd"],
            "signed_difference": float(pred["mean"] - ref["mean"]),
            "absolute_difference": float(abs(pred["mean"] - ref["mean"])),
        }
        if not mask.any():
            record.update({
                "bias": None, "mae": None, "rmse": None, "nrmse": None,
                "relative_nrmse": None, "pearson_r": None, "cc": None,
                "ncc": None, "psnr": None, "psnr_data_range_ms": None,
                "psnr_status": "empty_common_support", "ssim": None,
                "ssim_status": "empty_common_support",
            })
        elif parameter is None:
            from ..phase2_clean.evaluation import _metrics
            record.update(_metrics(reference, prediction, mask,
                                  ssim_status="not_available_parameter_unspecified"))
        else:
            from ..phase2_clean.evaluation import PSNR_DATA_RANGES_MS, _metrics, masked_ssim
            key = str(parameter).lower()
            if key not in PSNR_DATA_RANGES_MS:
                raise ValueError("AHA segment parameter must be t1 or t2")
            data_range = float(PSNR_DATA_RANGES_MS[key])
            ssim_value, ssim_status = masked_ssim(reference, prediction, mask, data_range=data_range)
            record.update(_metrics(reference, prediction, mask,
                                  psnr_data_range=data_range,
                                  ssim_value=ssim_value,
                                  ssim_status=ssim_status))
        result[int(segment)] = record
    return result


def validate_aha16_statistics(rows: Iterable[Mapping[str, object]], *, minimum_segment_pixels: int = 10) -> dict[str, object]:
    """Historical SCMR AHA16 presence/support gate, adapted to paired rows."""
    if minimum_segment_pixels < 1:
        raise ValueError("minimum AHA segment pixels must be positive")
    by_parameter: dict[str, dict[int, list[int]]] = {}
    for row in rows:
        parameter, segment, count = str(row["parameter"]).lower(), int(row["segment"]), int(row["n"])
        by_parameter.setdefault(parameter, {}).setdefault(segment, []).append(count)
    failures: list[dict[str, object]] = []
    for parameter in ("t1", "t2"):
        segments = by_parameter.get(parameter, {})
        missing = sorted(set(range(1, 17)) - set(segments))
        if missing:
            failures.append({"parameter": parameter, "missing_segments": missing})
        for segment, counts in sorted(segments.items()):
            if any(count < minimum_segment_pixels for count in counts):
                failures.append({"parameter": parameter, "segment": segment, "reason": "insufficient support", "counts": counts})
    return {"status": "PASS" if not failures else "FAIL", "minimum_segment_pixels": minimum_segment_pixels,
            "failures": failures, "note": "Visual AHA orientation QC remains mandatory before final Figure 3."}


def render_single_bullseye(output_png: str | Path, values: Mapping[int, float], *, parameter: str, title: str,
                           display_range: tuple[float, float] | None = None,
                           colorbar_label: str | None = None) -> Path:
    """Save exactly one historical-style AHA16 bullseye as immutable PNG/PDF."""
    import os
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge

    parameter = parameter.lower()
    if parameter not in DISPLAY_RANGES:
        raise ValueError("parameter must be t1 or t2")
    output = Path(output_png)
    pdf = output.with_suffix(".pdf")
    if output.exists() or pdf.exists():
        raise FileExistsError(f"refusing overwrite: {output} or {pdf}")
    output.parent.mkdir(parents=True, exist_ok=True)
    vmin, vmax = DISPLAY_RANGES[parameter] if display_range is None else tuple(map(float, display_range))
    if not (np.isfinite((vmin, vmax)).all() and vmin < vmax):
        raise ValueError("bullseye display range must be finite and increasing")
    fig, axis = plt.subplots(figsize=(4.2, 4.8), constrained_layout=True)
    norm, cmap = plt.Normalize(vmin, vmax), plt.get_cmap("viridis")
    for inner, outer, segments in ((.66, 1., range(1, 7)), (.34, .66, range(7, 13)), (.08, .34, range(13, 17))):
        segments = list(segments); width = 360 / len(segments)
        for index, segment in enumerate(segments):
            value = float(values.get(segment, float("nan")))
            color = cmap(norm(value)) if np.isfinite(value) else "lightgray"
            wedge = Wedge((0, 0), outer, 90 - (index + 1) * width, 90 - index * width, width=outer - inner,
                          facecolor=color, edgecolor="white", linewidth=1)
            axis.add_patch(wedge)
            theta, radius = np.deg2rad(90 - (index + .5) * width), (inner + outer) / 2
            label = f"{segment}\n{value:.0f}" if np.isfinite(value) else str(segment)
            axis.text(radius * np.cos(theta), radius * np.sin(theta), label, ha="center", va="center", fontsize=6)
    axis.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), aspect="equal")
    axis.axis("off"); axis.set_title(title, fontsize=9)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axis,
                 label=colorbar_label or f"{parameter.upper()} (ms)", shrink=.8)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def render_scmr_figure3(output_png: str | Path, *, reference: Mapping[str, Mapping[int, float]],
                        prediction: Mapping[str, Mapping[int, float]], subject: str, case: str,
                        method: str) -> Path:
    """Render the historical SCMR Figure-3 2x2 composition for one method.

    The layout, segment geometry, limits, labels, and white boundaries follow
    ``visualization.scmr.make_figure3_bullseye``; only the paired endpoint
    names differ from its historical original/reprojection inputs.
    """
    import os
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge

    output = Path(output_png)
    pdf = output.with_suffix(".pdf")
    if output.exists() or pdf.exists():
        raise FileExistsError(f"refusing overwrite: {output} or {pdf}")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(8, 8))
    for column, parameter in enumerate(("t1", "t2")):
        vmin, vmax = DISPLAY_RANGES[parameter]
        cmap, norm = plt.get_cmap("viridis"), plt.Normalize(vmin, vmax)
        for row, (values, label) in enumerate(((reference[parameter], "Native 2D"),
                                               (prediction[parameter], method.replace("_", "-") + " endpoint"))):
            axis = axes[row, column]
            for inner, outer, segments in ((.66, 1., range(1, 7)), (.34, .66, range(7, 13)), (.08, .34, range(13, 17))):
                segment_list = list(segments); width = 360 / len(segment_list)
                for index, segment in enumerate(segment_list):
                    value = float(values.get(segment, float("nan")))
                    color = cmap(norm(value)) if np.isfinite(value) else "lightgray"
                    axis.add_patch(Wedge((0, 0), outer, 90 - (index + 1) * width, 90 - index * width,
                                         width=outer - inner, facecolor=color, edgecolor="white", linewidth=1))
                    theta, radius = np.deg2rad(90 - (index + .5) * width), (inner + outer) / 2
                    axis.text(radius * np.cos(theta), radius * np.sin(theta),
                              f"{segment}\n{value:.0f}" if np.isfinite(value) else str(segment),
                              ha="center", va="center", fontsize=6)
            axis.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), aspect="equal")
            axis.axis("off"); axis.set_title(f"{parameter.upper()} {label}")
        figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[:, column], fraction=.025, label="ms")
    figure.suptitle(f"{subject} {case} AHA16 myocardial regional means")
    figure.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output
