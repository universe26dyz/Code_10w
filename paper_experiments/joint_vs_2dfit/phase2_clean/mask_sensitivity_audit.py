"""Read-only quantitative sensitivity audit for old-quality versus Phase4R masks.

Each ``.npz`` under ``--input-root`` must contain co-registered native-plane
arrays named ``reference``, ``prediction``, ``old_quality_common``, and
``phase4r_l100``.  ``source_support`` is optional; when supplied it is a
shared eligibility gate, not a third mask being compared.  This command never
reprojects, reconstructs, exports, or alters any formal evaluation output.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import sha256, write_immutable_json
from .evaluation import _metrics


DOMAINS = ("old_quality_common", "phase4r_l100", "intersection", "old_only", "l100_only")
REQUIRED_ARRAYS = ("reference", "prediction", "old_quality_common", "phase4r_l100")


def _metric_or_none(reference: np.ndarray, prediction: np.ndarray, support: np.ndarray) -> dict[str, Any] | None:
    if not np.any(support):
        return None
    return _metrics(reference, prediction, support)


def compare_mask_sensitivity(
    reference: np.ndarray,
    prediction: np.ndarray,
    old_quality_common: np.ndarray,
    phase4r_l100: np.ndarray,
    source_support: np.ndarray | None = None,
) -> dict[str, Any]:
    """Hold values fixed and evaluate the exact effect of changing support.

    A finite, optional-source eligibility base is shared by every domain, so
    no domain receives a private prediction-validity advantage.
    """
    arrays = (reference, prediction, old_quality_common, phase4r_l100)
    if source_support is not None:
        arrays = (*arrays, source_support)
    shapes = {np.asarray(value).shape for value in arrays}
    if len(shapes) != 1:
        raise ValueError("reference, prediction, and support arrays must share one shape")
    reference, prediction = np.asarray(reference, float), np.asarray(prediction, float)
    base = np.isfinite(reference) & np.isfinite(prediction)
    if source_support is not None:
        base &= np.asarray(source_support, bool)
    old_mask = np.asarray(old_quality_common, bool)
    l100_mask = np.asarray(phase4r_l100, bool)
    raw_domains = {
        "old_quality_common": old_mask,
        "phase4r_l100": l100_mask,
        "intersection": old_mask & l100_mask,
        "old_only": old_mask & ~l100_mask,
        "l100_only": l100_mask & ~old_mask,
    }
    result: dict[str, Any] = {}
    for name, raw in raw_domains.items():
        support = base & raw
        result[name] = {
            "support_voxel_count_before_finite_gate": int(raw.sum()),
            "support_voxel_count": int(support.sum()),
            "coverage_fraction_of_shared_base": float(support.sum() / base.sum()) if base.any() else 0.0,
            "metrics": _metric_or_none(reference, prediction, support),
        }
    union = old_mask | l100_mask
    overlap = old_mask & l100_mask
    result["mask_overlap"] = {
        "old_quality_common_voxels": int(old_mask.sum()),
        "phase4r_l100_voxels": int(l100_mask.sum()),
        "intersection_voxels": int(overlap.sum()),
        "union_voxels": int(union.sum()),
        "jaccard": float(overlap.sum() / union.sum()) if union.any() else 1.0,
        "dice": float(2 * overlap.sum() / (old_mask.sum() + l100_mask.sum())) if (old_mask.sum() + l100_mask.sum()) else 1.0,
        "shared_finite_base_voxels": int(base.sum()),
    }
    return result


def _input_files(root: Path) -> list[Path]:
    files = sorted(root.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no .npz audit inputs under {root}")
    return files


def _load_plane(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    with np.load(path, allow_pickle=False) as payload:
        missing = [name for name in REQUIRED_ARRAYS if name not in payload.files]
        if missing:
            raise ValueError(f"{path} is missing required arrays: {', '.join(missing)}")
        values = {name: np.asarray(payload[name]) for name in REQUIRED_ARRAYS}
        source = np.asarray(payload["source_support"], bool) if "source_support" in payload.files else None
    return values, source


def _pooled_metrics(records: list[dict[str, np.ndarray]], domain: str) -> dict[str, Any] | None:
    reference = [record["reference"][record[domain]] for record in records if record[domain].any()]
    prediction = [record["prediction"][record[domain]] for record in records if record[domain].any()]
    if not reference:
        return None
    ref, pred = np.concatenate(reference), np.concatenate(prediction)
    return _metrics(ref, pred, np.ones(ref.shape, bool))


def run_mask_sensitivity_audit(input_root: str | Path, output: str | Path) -> Path:
    """Publish a new immutable audit directory from explicit native-plane bundles."""
    input_root, output = Path(input_root), Path(output)
    if not input_root.is_dir():
        raise NotADirectoryError(input_root)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    files = _input_files(input_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        per_plane: list[dict[str, Any]] = []
        pooled: list[dict[str, np.ndarray]] = []
        for path in files:
            values, source = _load_plane(path)
            result = compare_mask_sensitivity(**values, source_support=source)
            per_plane.append({"input": {"path": str(path.resolve()), "sha256": sha256(path)}, "result": result})
            reference, prediction = np.asarray(values["reference"], float), np.asarray(values["prediction"], float)
            base = np.isfinite(reference) & np.isfinite(prediction)
            if source is not None:
                base &= source
            old, l100 = np.asarray(values["old_quality_common"], bool), np.asarray(values["phase4r_l100"], bool)
            pooled.append({
                "reference": reference, "prediction": prediction,
                "old_quality_common": base & old,
                "phase4r_l100": base & l100,
                "intersection": base & old & l100,
                "old_only": base & old & ~l100,
                "l100_only": base & l100 & ~old,
            })
        aggregate = {"statistical_unit": "voxel-pooled native-plane values; not plane-averaged", "domains": {}}
        rows: list[dict[str, Any]] = []
        for domain in DOMAINS:
            metrics = _pooled_metrics(pooled, domain)
            support_voxels = int(sum(record[domain].sum() for record in pooled))
            aggregate["domains"][domain] = {"support_voxel_count": support_voxels, "metrics": metrics}
            rows.append({"scope": "subject_voxel_pooled", "domain": domain, "support_voxel_count": support_voxels,
                         **({} if metrics is None else metrics)})
        with (staging / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader(); writer.writerows(rows)
        write_immutable_json(staging / "per_plane_metrics.json", {"schema": "q004_mask_sensitivity_per_plane/v1", "planes": per_plane})
        write_immutable_json(staging / "aggregate_metrics.json", aggregate)
        write_immutable_json(staging / "manifest.json", {
            "schema": "q004_mask_sensitivity_audit/v1",
            "input_root": str(input_root.resolve()),
            "inputs": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in files],
            "required_arrays": list(REQUIRED_ARRAYS),
            "optional_array": "source_support",
            "support_equations": {
                "shared_base": "finite(reference) AND finite(prediction) AND optional source_support",
                "old_quality_common": "shared_base AND supplied old_quality_common",
                "phase4r_l100": "shared_base AND supplied phase4r_l100",
                "intersection": "shared_base AND old_quality_common AND phase4r_l100",
                "old_only": "shared_base AND old_quality_common AND NOT phase4r_l100",
                "l100_only": "shared_base AND phase4r_l100 AND NOT old_quality_common",
            },
            "no_training": True, "no_reconstruction": True, "no_export": True, "no_reprojection": True,
        })
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="directory of co-registered native-plane .npz bundles")
    parser.add_argument("--output", required=True, help="new audit directory; must not already exist")
    print(run_mask_sensitivity_audit(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
