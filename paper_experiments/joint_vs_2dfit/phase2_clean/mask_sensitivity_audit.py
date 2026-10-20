"""Read-only quantitative sensitivity audit for SCMR and Phase4R supports."""
from __future__ import annotations

import argparse
import csv
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import sha256, write_immutable_json
from .evaluation import _metrics


MASKS = ("scmr_style_central_component", "phase4r_l100", "phase4r_l100_central_component")
REQUIRED_ARRAYS = ("reference", "prediction", *MASKS)


def _metric_or_none(reference: np.ndarray, prediction: np.ndarray, support: np.ndarray) -> dict[str, Any] | None:
    return _metrics(reference, prediction, support) if np.any(support) else None


def compare_mask_sensitivity(reference: np.ndarray, prediction: np.ndarray, scmr_style_central_component: np.ndarray, phase4r_l100: np.ndarray, phase4r_l100_central_component: np.ndarray, source_support: np.ndarray | None = None) -> dict[str, Any]:
    """Hold values fixed while exposing construction and selection-rule effects."""
    supplied = {"reference": reference, "prediction": prediction, "scmr_style_central_component": scmr_style_central_component, "phase4r_l100": phase4r_l100, "phase4r_l100_central_component": phase4r_l100_central_component}
    if source_support is not None: supplied["source_support"] = source_support
    if len({np.asarray(value).shape for value in supplied.values()}) != 1:
        raise ValueError("reference, prediction, and support arrays must share one shape")
    reference, prediction = np.asarray(reference, float), np.asarray(prediction, float)
    base = np.isfinite(reference) & np.isfinite(prediction)
    if source_support is not None: base &= np.asarray(source_support, bool)
    masks = {name: np.asarray(supplied[name], bool) for name in MASKS}
    result: dict[str, Any] = {}
    for name, mask in masks.items():
        support = base & mask
        result[name] = {"support_voxel_count_before_finite_gate": int(mask.sum()), "support_voxel_count": int(support.sum()), "coverage_fraction_of_shared_base": float(support.sum() / base.sum()) if base.any() else 0.0, "metrics": _metric_or_none(reference, prediction, support)}
    for first, second in ((MASKS[0], MASKS[1]), (MASKS[2], MASKS[1])):
        prefix = f"{first}_vs_{second}"
        for suffix, raw in (("intersection", masks[first] & masks[second]), ("first_only", masks[first] & ~masks[second]), ("second_only", masks[second] & ~masks[first])):
            support = base & raw
            result[f"{prefix}_{suffix}"] = {"support_voxel_count_before_finite_gate": int(raw.sum()), "support_voxel_count": int(support.sum()), "coverage_fraction_of_shared_base": float(support.sum() / base.sum()) if base.any() else 0.0, "metrics": _metric_or_none(reference, prediction, support)}
        union, overlap = masks[first] | masks[second], masks[first] & masks[second]
        result[f"{prefix}_overlap"] = {"first_voxels": int(masks[first].sum()), "second_voxels": int(masks[second].sum()), "intersection_voxels": int(overlap.sum()), "union_voxels": int(union.sum()), "jaccard": float(overlap.sum() / union.sum()) if union.any() else 1.0, "dice": float(2 * overlap.sum() / (masks[first].sum() + masks[second].sum())) if (masks[first].sum() + masks[second].sum()) else 1.0, "shared_finite_base_voxels": int(base.sum())}
    return result


def _load_plane(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    with np.load(path, allow_pickle=False) as payload:
        missing = [name for name in REQUIRED_ARRAYS if name not in payload.files]
        if missing: raise ValueError(f"{path} is missing required arrays: {', '.join(missing)}")
        values = {name: np.asarray(payload[name]) for name in REQUIRED_ARRAYS}
        source = np.asarray(payload["source_support"], bool) if "source_support" in payload.files else None
    return values, source


def _pooled_metrics(records: list[dict[str, np.ndarray]], domain: str) -> dict[str, Any] | None:
    reference = [record["reference"][record[domain]] for record in records if record[domain].any()]
    prediction = [record["prediction"][record[domain]] for record in records if record[domain].any()]
    if not reference: return None
    ref, pred = np.concatenate(reference), np.concatenate(prediction)
    return _metrics(ref, pred, np.ones(ref.shape, bool))


def run_mask_sensitivity_audit(input_root: str | Path, output: str | Path) -> Path:
    input_root, output = Path(input_root), Path(output)
    if not input_root.is_dir(): raise NotADirectoryError(input_root)
    if output.exists(): raise FileExistsError(f"refusing overwrite: {output}")
    files = sorted(input_root.rglob("*.npz"))
    if not files: raise FileNotFoundError(f"no .npz audit inputs under {input_root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        per_plane: list[dict[str, Any]] = []; pooled: list[dict[str, np.ndarray]] = []
        for path in files:
            values, source = _load_plane(path); result = compare_mask_sensitivity(**values, source_support=source)
            per_plane.append({"input": {"path": str(path.resolve()), "sha256": sha256(path)}, "result": result})
            base = np.isfinite(values["reference"]) & np.isfinite(values["prediction"])
            if source is not None: base &= source
            masks = {name: np.asarray(values[name], bool) for name in MASKS}
            domains = {name: base & mask for name, mask in masks.items()}
            for first, second in ((MASKS[0], MASKS[1]), (MASKS[2], MASKS[1])):
                prefix = f"{first}_vs_{second}"
                domains.update({f"{prefix}_intersection": base & masks[first] & masks[second], f"{prefix}_first_only": base & masks[first] & ~masks[second], f"{prefix}_second_only": base & masks[second] & ~masks[first]})
            pooled.append({"reference": np.asarray(values["reference"], float), "prediction": np.asarray(values["prediction"], float), **domains})
        domain_names = tuple(pooled[0].keys() - {"reference", "prediction"})
        aggregate = {"statistical_unit": "voxel-pooled native-plane values; not plane-averaged", "domains": {}}; rows: list[dict[str, Any]] = []
        for domain in sorted(domain_names):
            metrics = _pooled_metrics(pooled, domain); support_voxels = int(sum(record[domain].sum() for record in pooled))
            aggregate["domains"][domain] = {"support_voxel_count": support_voxels, "metrics": metrics}
            rows.append({"scope": "subject_voxel_pooled", "domain": domain, "support_voxel_count": support_voxels, **({} if metrics is None else metrics)})
        with (staging / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row})); writer.writeheader(); writer.writerows(rows)
        write_immutable_json(staging / "per_plane_metrics.json", {"schema": "q004_mask_sensitivity_per_plane/v2", "planes": per_plane})
        write_immutable_json(staging / "aggregate_metrics.json", aggregate)
        write_immutable_json(staging / "manifest.json", {"schema": "q004_mask_sensitivity_audit/v2", "input_root": str(input_root.resolve()), "inputs": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in files], "required_arrays": list(REQUIRED_ARRAYS), "optional_array": "source_support", "support_equations": {"shared_base": "finite(reference) AND finite(prediction) AND optional source_support", "scmr_style_central_component": "shared_base AND supplied scmr_style_central_component", "phase4r_l100": "shared_base AND supplied phase4r_l100", "phase4r_l100_central_component": "shared_base AND supplied phase4r_l100_central_component"}, "control_interpretation": "phase4r_l100_central_component isolates central-component selection from six-input SCMR strict-common construction", "no_training": True, "no_reconstruction": True, "no_export": True, "no_reprojection": True})
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True); raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--input-root", required=True); parser.add_argument("--output", required=True)
    print(run_mask_sensitivity_audit(**vars(parser.parse_args())))


if __name__ == "__main__": main()
