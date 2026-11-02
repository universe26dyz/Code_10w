"""Summarize sampled Trad timing profiles without changing reconstruction."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


FIELDS = ("stage", "section", "N", "mean_ms", "median_ms", "std_ms", "p10_ms", "p90_ms", "median_fraction_of_whole_iteration")


def summarize_timing_profile(path: str | Path, *, stage_iterations: Mapping[str, int], decoder_type: str = "Bloch", run_level_ms: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Write stable stage/section statistics next to a raw profiler CSV."""

    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    required = {"iteration", "stage", "section", "milliseconds"}
    if raw and set(raw[0]) != required:
        raise ValueError("timing profile must contain iteration, stage, section, milliseconds.")
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    whole: dict[tuple[str, int], float] = {}
    for row in raw:
        key = (str(row["stage"]), int(row["iteration"]))
        value = float(row["milliseconds"])
        if not np.isfinite(value) or value < 0:
            raise ValueError("timing profile milliseconds must be finite and non-negative.")
        grouped[(key[0], str(row["section"]))].append(value)
        if row["section"] == "whole_iteration":
            whole[key] = value
    rows: list[dict[str, Any]] = []
    estimate_ms = 0.0
    for (stage, section), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        fractions = [value / whole[(stage, int(row["iteration"]))] for row, value in zip((r for r in raw if r["stage"] == stage and r["section"] == section), values) if whole.get((stage, int(row["iteration"])), 0.0) > 0]
        row = {
            "stage": stage, "section": section, "N": int(array.size),
            "mean_ms": float(array.mean()), "median_ms": float(np.median(array)),
            "std_ms": float(array.std(ddof=0)), "p10_ms": float(np.percentile(array, 10)),
            "p90_ms": float(np.percentile(array, 90)),
            "median_fraction_of_whole_iteration": float(np.median(fractions)) if fractions else float("nan"),
        }
        rows.append(row)
        if section == "whole_iteration":
            if stage not in stage_iterations:
                raise ValueError(f"stage iteration count is missing for stage {stage!r}.")
            estimate_ms += row["median_ms"] * int(stage_iterations[stage])
    run_level = {key: float(value) for key, value in (run_level_ms or {}).items()}
    required_run_level = ("checkpoint_load", "data_loading", "initialization", "optimization_total", "validation", "export", "total_runtime")
    missing_run_level = [key for key in required_run_level if key not in run_level]
    if missing_run_level:
        run_level.update({key: 0.0 for key in missing_run_level})
    result = {"schema": "reconstruction_timing_profile_summary/v2", "source": str(path), "decoder_type": decoder_type, "run_level_ms": run_level, "rows": rows, "stage_iterations": {key: int(value) for key, value in stage_iterations.items()}, "estimated_total_training_seconds": estimate_ms / 1000.0}
    csv_path, json_path = path.with_name("timing_profile_summary.csv"), path.with_name("timing_profile_summary.json")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    json_path.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-profile", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path, help="Trad YAML with training.stage_a_iterations/stage_b_iterations.")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    training = config.get("training", {}) if isinstance(config, dict) else {}
    print(json.dumps(summarize_timing_profile(args.timing_profile, stage_iterations={"A": int(training["stage_a_iterations"]), "B": int(training["stage_b_iterations"])}), indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
