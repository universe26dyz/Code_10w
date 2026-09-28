"""Strict join and exploratory association statistics for G0 pose/error data."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr


MAP_REQUIRED = {"subject_id", "stack", "parameter", "expected_group", "content_matched_group", "best_orientation", "correlation", "relative_rmse", "valid_common_support_count", "assignment_margin"}
POSE_REQUIRED = {"stack", "group_idx", "global_group_idx", "translation_magnitude_mm", "rotation_magnitude_deg", "final_center_ras_mm", "slice_normal_ras", "neighbor_center_distance_mm"}
POSE_METRICS = ("translation_magnitude_mm", "rotation_magnitude_deg")
ERROR_METRICS = ("relative_rmse", "one_minus_correlation")


def _finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite; G0 pose-error analysis does not silently drop rows.")
    return number


def _integer(value: object, label: str) -> int:
    number = int(value)
    if str(number) != str(value) and not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer.")
    return number


def join_pose_map_rows(map_rows: list[dict[str, object]], pose_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Join only stack plus expected_group equal to group_idx after official-pair checks."""

    pose_index: dict[tuple[str, int], dict[str, object]] = {}
    for row in pose_rows:
        missing = POSE_REQUIRED.difference(row)
        if missing:
            raise ValueError(f"Pose row lacks {sorted(missing)}")
        stack, group = str(row["stack"]), _integer(row["group_idx"], "group_idx")
        key = (stack, group)
        if key in pose_index:
            raise ValueError(f"duplicate pose group for stack={stack}, group_idx={group}")
        for field in ("translation_magnitude_mm", "rotation_magnitude_deg"):
            _finite(row[field], f"pose {field}")
        pose_index[key] = row
    map_keys: set[tuple[str, str, int]] = set()
    joined: list[dict[str, object]] = []
    for row in map_rows:
        missing = MAP_REQUIRED.difference(row)
        if missing:
            raise ValueError(f"Map row lacks {sorted(missing)}")
        stack, parameter = str(row["stack"]), str(row["parameter"])
        expected, matched = _integer(row["expected_group"], "expected_group"), _integer(row["content_matched_group"], "content_matched_group")
        key = (stack, parameter, expected)
        if key in map_keys:
            raise ValueError(f"duplicate map group for stack={stack}, parameter={parameter}, expected_group={expected}")
        map_keys.add(key)
        if expected != matched:
            raise ValueError("content_matched_group must equal expected_group for official same-group pose/error analysis.")
        if str(row["best_orientation"]) != "IDENTITY":
            raise ValueError("best_orientation must be IDENTITY for official same-group pose/error analysis.")
        if (stack, expected) not in pose_index:
            raise ValueError(f"missing pose for stack={stack}, expected_group={expected}")
        correlation, relative_rmse = _finite(row["correlation"], "correlation"), _finite(row["relative_rmse"], "relative_rmse")
        _finite(row["assignment_margin"], "assignment_margin")
        pose = pose_index[(stack, expected)]
        joined.append({"subject_id": str(row["subject_id"]), "stack": stack, "parameter": parameter, "group_idx": expected, "global_group_idx": _integer(pose["global_group_idx"], "global_group_idx"), "translation_magnitude_mm": _finite(pose["translation_magnitude_mm"], "translation_magnitude_mm"), "rotation_magnitude_deg": _finite(pose["rotation_magnitude_deg"], "rotation_magnitude_deg"), "correlation": correlation, "relative_rmse": relative_rmse, "one_minus_correlation": 1.0 - correlation, "valid_common_support_count": _integer(row["valid_common_support_count"], "valid_common_support_count"), "assignment_margin": _finite(row["assignment_margin"], "assignment_margin")})
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in joined:
        grouped[(str(row["stack"]), str(row["parameter"]))].append(int(row["group_idx"]))
    for (stack, parameter), groups in grouped.items():
        expected_groups = {key[1] for key in pose_index if key[0] == stack}
        if set(groups) != expected_groups:
            raise ValueError(f"stack={stack}, parameter={parameter} map group set does not exactly match pose group set.")
    for stack in {str(row["stack"]) for row in joined}:
        by_group: dict[int, list[dict[str, object]]] = defaultdict(list)
        for row in joined:
            if row["stack"] == stack:
                by_group[int(row["group_idx"])].append(row)
        for group, rows in by_group.items():
            if len(rows) == 2 and {str(row["parameter"]) for row in rows} == {"T1", "T2"}:
                for field in ("global_group_idx", "translation_magnitude_mm", "rotation_magnitude_deg"):
                    if rows[0][field] != rows[1][field]:
                        raise ValueError(f"T1/T2 pose consistency failure for stack={stack}, group={group}, field={field}")
    return joined


def primary_statistics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Compute four Pearson/Spearman association tests for one stack and parameter."""

    if len(rows) < 3:
        raise ValueError("At least three groups are required for exploratory correlation statistics.")
    stack, parameter = str(rows[0]["stack"]), str(rows[0]["parameter"])
    result = []
    for pose_metric in POSE_METRICS:
        for error_metric in ERROR_METRICS:
            x = np.asarray([_finite(row[pose_metric], pose_metric) for row in rows])
            y = np.asarray([_finite(row[error_metric], error_metric) for row in rows])
            pearson, spearman = pearsonr(x, y), spearmanr(x, y)
            result.append({"stack": stack, "parameter": parameter, "pose_metric": pose_metric, "error_metric": error_metric, "N": int(x.size), "pearson_r": float(pearson.statistic), "pearson_p_two_sided": float(pearson.pvalue), "spearman_rho": float(spearman.statistic), "spearman_p_two_sided": float(spearman.pvalue)})
    return result


def bh_fdr(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted q values in original input order."""

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("BH-FDR requires finite p values in [0, 1].")
    order, adjusted, running, count = np.argsort(values), np.empty_like(values), 1.0, len(values)
    for rank in range(count, 0, -1):
        index = order[rank - 1]
        running = min(running, values[index] * count / rank)
        adjusted[index] = running
    return adjusted.tolist()


def leave_one_out_sensitivity(rows: list[dict[str, object]], pose_metric: str, error_metric: str) -> dict[str, object]:
    """Summarize leave-one-group-out Spearman perturbations for one association."""

    if len(rows) < 4:
        raise ValueError("Leave-one-out sensitivity requires at least four groups.")
    x = np.asarray([_finite(row[pose_metric], pose_metric) for row in rows])
    y = np.asarray([_finite(row[error_metric], error_metric) for row in rows])
    full = float(spearmanr(x, y).statistic)
    loo = []
    for index, row in enumerate(rows):
        keep = np.arange(len(rows)) != index
        loo.append((float(spearmanr(x[keep], y[keep]).statistic), int(row["group_idx"]), int(row["global_group_idx"])))
    influence = [abs(rho - full) for rho, _, _ in loo]
    selected = int(np.argmax(influence))
    return {"full_rho": full, "loo_rho_min": float(min(value[0] for value in loo)), "loo_rho_max": float(max(value[0] for value in loo)), "loo_rho_median": float(np.median([value[0] for value in loo])), "most_influential_removed_group_idx": loo[selected][1], "most_influential_removed_global_group_idx": loo[selected][2]}


def read_csv_rows(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_bundle(root: str | Path) -> tuple[Path, dict[str, Any], Path, Path]:
    root = Path(root).expanduser().resolve()
    per_slice, pose, manifest_path = root / "metrics/G0_per_slice.csv", root / "metrics/G0_pose_drift.csv", root / "manifest.json"
    if root.name != "G0_geometry_correspondence_baseline_v2" or not per_slice.is_file() or not pose.is_file():
        raise FileNotFoundError("G0 v2 bundle must contain metrics/G0_per_slice.csv and metrics/G0_pose_drift.csv under G0_geometry_correspondence_baseline_v2.")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"G0 v2 bundle manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("implementation_revision") != "G0_geometry_correspondence_baseline_v2":
        raise ValueError("G0 input manifest is not the corrected baseline_v2 implementation.")
    return root, manifest, per_slice, pose
