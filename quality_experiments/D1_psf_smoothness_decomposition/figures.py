"""D1 visualizations; cropping here is display-only and never affects metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _crop_for_display(values: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.where(support)
    if not rows.size:
        return values, support
    return values[rows.min():rows.max() + 1, cols.min():cols.max() + 1], support[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def render_comparison(
    *, parameter: str, stack: str, group: int, native: np.ndarray, central_bloch: np.ndarray, psf_bloch: np.ndarray,
    central_mlp: np.ndarray, psf_mlp: np.ndarray, support: np.ndarray, target: str | Path,
) -> None:
    """Render five maps and four residuals with explicit central/PSF labels."""

    import matplotlib.pyplot as plt
    from trad.evaluation.scmr.quality_control import DISPLAY_RANGES_MS, RESIDUAL_RANGES_MS

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    maps = [
        (native, "Native 2-D"),
        (central_bloch, "Bloch CENTRAL-PLANE"),
        (psf_bloch, "Bloch MAP-DOMAIN PSF"),
        (central_mlp, "FrozenMLP CENTRAL-PLANE"),
        (psf_mlp, "FrozenMLP MAP-DOMAIN PSF"),
    ]
    residuals = [
        (central_bloch - native, "|Bloch CENTRAL − Native|"),
        (psf_bloch - native, "|Bloch PSF − Native|"),
        (central_mlp - native, "|FrozenMLP CENTRAL − Native|"),
        (psf_mlp - native, "|FrozenMLP PSF − Native|"),
    ]
    figure, axes = plt.subplots(3, 3, figsize=(10.8, 9.2), constrained_layout=True)
    map_range, residual_range = DISPLAY_RANGES_MS[parameter], RESIDUAL_RANGES_MS[parameter]
    for axis, (values, title) in zip(axes.flat[:5], maps):
        cropped, _ = _crop_for_display(values, support)
        axis.imshow(cropped, cmap="viridis", vmin=map_range[0], vmax=map_range[1], interpolation="nearest")
        axis.set_title(title, fontsize=9); axis.axis("off")
    for axis, (values, title) in zip(axes.flat[5:], residuals):
        cropped, _ = _crop_for_display(np.abs(values), support)
        axis.imshow(cropped, cmap="magma", vmin=0, vmax=residual_range[1], interpolation="nearest")
        axis.set_title(title, fontsize=9); axis.axis("off")
    figure.suptitle(f"D1 {parameter} {stack.upper()} group {group}: display crop only; metrics use full native grid", fontsize=11)
    figure.savefig(target, dpi=180)
    plt.close(figure)
