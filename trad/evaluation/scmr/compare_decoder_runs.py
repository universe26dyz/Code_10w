"""Compare completed decoder-only Trad and FrozenMLP evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Required JSON does not exist: {target}")
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {target}")
    return value


def _numeric_leaves(value: Any, prefix: str = "") -> dict[str, float]:
    if isinstance(value, bool):
        return {}
    if isinstance(value, (int, float)):
        return {prefix: float(value)}
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for key, child in value.items():
        child_prefix = f"{prefix}.{key}" if prefix else str(key)
        result.update(_numeric_leaves(child, child_prefix))
    return result


def _timing_summary(path: str | Path | None, expected_decoder: str) -> dict[str, Any] | None:
    if path is None:
        return None
    summary = _load_json(path)
    if summary.get("decoder_type") != expected_decoder:
        raise ValueError(f"Timing summary {path} must declare decoder_type={expected_decoder!r}.")
    return {"path": str(Path(path).resolve()), "decoder_type": summary["decoder_type"], "run_level_ms": summary.get("run_level_ms", {})}


def compare_decoder_runs(trad_metrics_path: str | Path, mlp_metrics_path: str | Path, output_path: str | Path, *, trad_timing_summary: str | Path | None = None, mlp_timing_summary: str | Path | None = None) -> dict[str, Any]:
    """Write paired metric deltas; negative means MLP is lower than Trad."""

    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Comparison output already exists: {output}")
    trad, mlp = _load_json(trad_metrics_path), _load_json(mlp_metrics_path)
    trad_values = _numeric_leaves(trad)
    mlp_values = _numeric_leaves(mlp)
    common = sorted(set(trad_values).intersection(mlp_values))
    if not common:
        raise ValueError("Trad and MLP metrics have no matching numeric fields.")
    result = {
        "schema": "decoder_only_comparison/v1",
        "metric_definition": "value is FrozenMLP minus Bloch/Trad; no sign implies clinical superiority without metric interpretation.",
        "trad_metrics": str(Path(trad_metrics_path).resolve()),
        "mlp_metrics": str(Path(mlp_metrics_path).resolve()),
        "metric_delta_mlp_minus_trad": {key: mlp_values[key] - trad_values[key] for key in common},
        "timing": {"trad": _timing_summary(trad_timing_summary, "Bloch"), "mlp": _timing_summary(mlp_timing_summary, "FrozenMLP")},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trad-map-metrics", required=True)
    parser.add_argument("--mlp-map-metrics", required=True)
    parser.add_argument("--trad-timing-summary")
    parser.add_argument("--mlp-timing-summary")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(compare_decoder_runs(args.trad_map_metrics, args.mlp_map_metrics, args.output, trad_timing_summary=args.trad_timing_summary, mlp_timing_summary=args.mlp_timing_summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
