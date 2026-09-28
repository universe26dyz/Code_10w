"""G0 figures; visual transforms are diagnostic only and never rewrite data."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def render_score_matrix(score: np.ndarray, stack: str, parameter: str, target: str | Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = axis.imshow(score, cmap="viridis", interpolation="nearest")
    figure.colorbar(image, ax=axis, label="correlation score")
    axis.set(xlabel="AUDIT candidate reconstructed group", ylabel="EXPECTED native group", title=f"G0 {parameter} {stack.upper()} score matrix")
    figure.savefig(target, dpi=180); plt.close(figure)


def render_assignment(assignment: np.ndarray, stack: str, parameter: str, target: str | Path) -> None:
    import matplotlib.pyplot as plt

    expected = np.arange(assignment.size)
    figure, axis = plt.subplots(figsize=(5, 5), constrained_layout=True)
    axis.plot(expected, expected, "k--", label="EXPECTED PAIR")
    axis.scatter(expected, assignment, label="AUDIT BEST MATCH")
    axis.set(xlabel="expected native group", ylabel="content-matched reconstructed group", title=f"G0 {parameter} {stack.upper()} assignment")
    axis.legend(); figure.savefig(target, dpi=180); plt.close(figure)


def render_worst_pair(native: np.ndarray, candidate: np.ndarray, stack: str, parameter: str, expected_group: int, matched_group: int, orientation: str, target: str | Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(10, 3.6), constrained_layout=True)
    for axis, values, title in (
        (axes[0], native, "EXPECTED PAIR: native"),
        (axes[1], candidate, f"AUDIT BEST MATCH: recon {matched_group} ({orientation})"),
        (axes[2], np.abs(candidate - native), "absolute residual"),
    ):
        axis.imshow(values, cmap="viridis" if axis is not axes[2] else "magma", interpolation="nearest")
        axis.set_title(title, fontsize=9); axis.axis("off")
    figure.suptitle(f"G0 {parameter} {stack.upper()} expected group {expected_group}")
    figure.savefig(target, dpi=180); plt.close(figure)


def render_pose_drift(rows: list[dict[str, object]], target: str | Path) -> None:
    import matplotlib.pyplot as plt

    groups = [int(row["global_group_idx"]) for row in rows]
    translations = [float(row["translation_magnitude_mm"]) for row in rows]
    rotations = [float(row["rotation_magnitude_deg"]) for row in rows]
    figure, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True, constrained_layout=True)
    axes[0].plot(groups, translations, "o-"); axes[0].set(ylabel="translation from init (mm)")
    axes[1].plot(groups, rotations, "o-"); axes[1].set(xlabel="global group", ylabel="rotation from init (deg)")
    figure.savefig(target, dpi=180); plt.close(figure)
