"""Small, explicit K=8 signal-domain D2 figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def render_metric_summaries(method_weight_rows: list[dict], disagreement_weight_rows: list[dict], target_root: str | Path) -> None:
    import matplotlib.pyplot as plt

    target = Path(target_root); target.mkdir(parents=True, exist_ok=True)
    weights = range(10)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for method in ("Bloch", "FrozenMLP"):
        values = {int(row["weight_idx"]): row for row in method_weight_rows if row["method"] == method}
        axes[0].plot(weights, [values[weight]["RMSE_signal"] for weight in weights], "o-", label=method)
        axes[1].plot(weights, [values[weight]["NCC"] for weight in weights], "o-", label=method)
    axes[0].set(title="K=8 SIGNAL-DOMAIN per-weight RMSE", xlabel="weight_idx", ylabel="RMSE_signal")
    axes[1].set(title="K=8 SIGNAL-DOMAIN per-weight NCC", xlabel="weight_idx", ylabel="NCC")
    for axis in axes: axis.legend()
    figure.savefig(target / "D2_K8_signal_per_weight_rmse_ncc.png", dpi=180); plt.close(figure)
    figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    values = {int(row["weight_idx"]): row for row in disagreement_weight_rows}
    axis.plot(weights, [values[weight]["RMSE_signal"] for weight in weights], "o-", label="Bloch vs FrozenMLP")
    axis.set(title="K=8 SIGNAL-DOMAIN decoder disagreement", xlabel="weight_idx", ylabel="RMSE_signal"); axis.legend()
    figure.savefig(target / "D2_K8_signal_bloch_vs_frozenmlp_per_weight.png", dpi=180); plt.close(figure)


def render_residual_montage(artifact_paths: dict[str, dict[str, Path]], target_root: str | Path) -> None:
    import matplotlib.pyplot as plt

    target = Path(target_root); target.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for row, method in enumerate(("Bloch", "FrozenMLP")):
        for column, stack in enumerate(("sax", "2ch", "4ch")):
            with np.load(artifact_paths[method][stack], allow_pickle=False) as data:
                residual = np.asarray(data["residual"])[0]
                mask = np.asarray(data["masks"])[0].astype(bool)
            view = np.where(mask, residual, np.nan)
            axes[row, column].imshow(view, cmap="coolwarm")
            axes[row, column].set(title=f"K=8 SIGNAL-DOMAIN {method} {stack.upper()}", xticks=[], yticks=[])
    figure.savefig(target / "D2_K8_signal_representative_residual_montage.png", dpi=180); plt.close(figure)
