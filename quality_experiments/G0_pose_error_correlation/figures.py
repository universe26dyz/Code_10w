"""Unsmooothed exploratory G0 pose/error figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def correlation_grid(rows: list[dict[str, object]], statistics: list[dict[str, object]], pose_metric: str, error_metric: str, target: str | Path) -> None:
    import matplotlib.pyplot as plt

    stacks, parameters = ("sax", "2ch", "4ch"), ("T1", "T2")
    lookup = {(row["stack"], row["parameter"], row["pose_metric"], row["error_metric"]): row for row in statistics}
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for row_index, parameter in enumerate(parameters):
        for column, stack in enumerate(stacks):
            axis = axes[row_index, column]
            values = [row for row in rows if row["stack"] == stack and row["parameter"] == parameter]
            x, y = np.asarray([float(row[pose_metric]) for row in values]), np.asarray([float(row[error_metric]) for row in values])
            axis.scatter(x, y)
            for value, x_value, y_value in zip(values, x, y):
                axis.annotate(str(value["group_idx"]), (x_value, y_value), xytext=(3, 3), textcoords="offset points", fontsize=8)
            if x.size >= 2 and np.ptp(x) > 0:
                slope, intercept = np.polyfit(x, y, 1)
                line = np.linspace(x.min(), x.max(), 50)
                axis.plot(line, slope * line + intercept, "--", color="0.4", linewidth=1)
            stat = lookup[(stack, parameter, pose_metric, error_metric)]
            axis.set_title(f"{parameter} {stack.upper()}\nN={stat['N']}; rho={stat['spearman_rho']:.3g}; p={stat['spearman_p_two_sided']:.3g}; q={stat['spearman_q_bh']:.3g}")
            axis.set(xlabel=pose_metric, ylabel=error_metric)
    figure.savefig(target, dpi=180); plt.close(figure)


def group_overview(rows: list[dict[str, object]], target: str | Path) -> None:
    import matplotlib.pyplot as plt

    stacks = ("sax", "2ch", "4ch")
    figure, axes = plt.subplots(6, 3, figsize=(13, 14), sharex="col", constrained_layout=True)
    fields = ("translation_magnitude_mm", "rotation_magnitude_deg", "relative_rmse", "one_minus_correlation", "relative_rmse", "one_minus_correlation")
    parameters = (None, None, "T1", "T1", "T2", "T2")
    labels = ("translation (mm)", "rotation (deg)", "T1 relative RMSE", "T1 1-correlation", "T2 relative RMSE", "T2 1-correlation")
    for column, stack in enumerate(stacks):
        values = sorted({int(row["group_idx"]) for row in rows if row["stack"] == stack})
        for row_index, (field, parameter, label) in enumerate(zip(fields, parameters, labels)):
            axis = axes[row_index, column]
            selected = [row for row in rows if row["stack"] == stack and (parameter is None or row["parameter"] == parameter)]
            if parameter is None:
                deduplicated = {int(row["group_idx"]): row for row in selected}
                selected = [deduplicated[group] for group in values]
            else:
                selected = sorted(selected, key=lambda row: int(row["group_idx"]))
            axis.plot([int(row["group_idx"]) for row in selected], [float(row[field]) for row in selected], "o-")
            axis.set(ylabel=label, title=stack.upper() if row_index == 0 else "")
    for axis in axes[-1]:
        axis.set_xlabel("local group_idx")
    figure.savefig(target, dpi=180); plt.close(figure)
