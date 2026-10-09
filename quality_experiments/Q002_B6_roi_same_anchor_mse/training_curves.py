"""Post-training-only convergence artifacts for the Q003-S640 10k route."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BOUNDARIES = {"stage_a_to_b": 2000, "primary_checkpoint": 6000, "final_checkpoint": 10000}
SMOOTHING_WINDOW = 100


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _series(rows: list[dict[str, str]], key: str) -> tuple[np.ndarray, np.ndarray]:
    points = [(int(row["iteration"]), float(row[key])) for row in rows if row.get(key, "") not in ("", None)]
    return np.asarray([point[0] for point in points], int), np.asarray([point[1] for point in points], float)


def _mark_stages(ax: Any) -> None:
    for iteration, label in ((2000, "A→B"), (6000, "primary / extension"), (10000, "final")):
        ax.axvline(iteration, color="black", linewidth=0.8, alpha=0.55)
        ax.text(iteration, 0.99, label, transform=ax.get_xaxis_transform(), rotation=90, va="top", ha="right", fontsize=7)


def _plot(path: Path, rows: list[dict[str, str]], columns: list[str], *, title: str, monitor: bool = False) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    for column in columns:
        x, y = _series(rows, column)
        if y.size:
            axis.plot(x, y, linewidth=0.65, alpha=0.7, label=column)
    _mark_stages(axis)
    axis.set(title=title, xlabel="optimizer iteration")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=160)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def _window_summary(rows: list[dict[str, str]], key: str) -> dict[str, float | None]:
    _, values_6000 = _series([row for row in rows if 5901 <= int(row["iteration"]) <= 6000], key)
    _, values_10000 = _series([row for row in rows if 9901 <= int(row["iteration"]) <= 10000], key)
    _, at_6000 = _series([row for row in rows if int(row["iteration"]) == 6000], key)
    _, at_10000 = _series([row for row in rows if int(row["iteration"]) == 10000], key)
    mean_6000 = float(values_6000.mean()) if values_6000.size else None
    mean_10000 = float(values_10000.mean()) if values_10000.size else None
    relative = None if mean_6000 in (None, 0.0) or mean_10000 is None else float((mean_10000 - mean_6000) / abs(mean_6000))
    return {"value_at_6000": float(at_6000[-1]) if at_6000.size else None, "value_at_10000": float(at_10000[-1]) if at_10000.size else None, "mean_5901_6000": mean_6000, "mean_9901_10000": mean_10000, "relative_change_6000_to_10000": relative}


def generate_convergence_artifacts(training_log: str | Path, monitor_log: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Render descriptive curves from closed CSV logs; this never touches training state."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    training_rows, monitor_rows = _read_rows(training_log), _read_rows(monitor_log)
    _plot(output / "data_loss_curves", training_rows, ["joint_signal_mse", "weighted_fingerprint_cosine_loss", "total_data_loss"], title="Q003-S640 data losses")
    _plot(output / "total_loss_curve", training_rows, ["total"], title="Q003-S640 total objective")
    _plot(output / "regularization_curves", training_rows, ["reg_t1", "reg_t2", "reg_b1", "amplitude_reg_t1", "amplitude_reg_t2", "transformation"], title="Q003-S640 regularization")
    _plot(output / "monitor_convergence", monitor_rows, ["monitor_joint_mse", "monitor_cosine_similarity"], title="Q003-S640 fixed monitor")
    _plot(output / "learning_rates", training_rows, ["lr_encoding", "lr_network", "lr_rigid"], title="Q003-S640 learning rates")
    summary = {key: _window_summary(training_rows, key) for key in ("joint_signal_mse", "weighted_fingerprint_cosine_loss", "total_data_loss", "total")}
    summary.update({key: _window_summary(monitor_rows, key) for key in ("monitor_joint_mse", "monitor_cosine_similarity")})
    manifest = {"source_training_log": str(Path(training_log)), "source_monitor_log": str(Path(monitor_log)), "smoothing": {"method": "none", "window": None}, "stage_boundaries": BOUNDARIES, "primary_checkpoint_iteration": 6000, "final_checkpoint_iteration": 10000, "descriptive_only": True}
    (output / "convergence_plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "convergence_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {"output_dir": output, "summary": summary, "manifest": manifest}
