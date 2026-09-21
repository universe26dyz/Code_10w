"""CPU/MATLAB helpers shared by the local PSF-K8 postprocess stage only."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import savemat

from .build_native_reference import _load_mapping
from .metrics import agreement_metrics, signal_agreement_metrics
from .reference_2d import STACKS, load_verified_reference, sha256


def write_json(value: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def load_signal(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def validate_signal_rows(data: dict[str, np.ndarray], stack: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"observed", "predicted", "residual", "group_idx", "weight_idx", "masks"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"{stack} signal archive lacks {sorted(missing)}")
    groups = np.asarray(data["group_idx"], dtype=np.int64)
    weights = np.asarray(data["weight_idx"], dtype=np.int64)
    masks = np.asarray(data["masks"], dtype=bool)
    predicted = np.asarray(data["predicted"], dtype=np.float32)
    if predicted.ndim != 3 or masks.shape != predicted.shape or groups.shape != (predicted.shape[0],) or weights.shape != groups.shape:
        raise ValueError(f"{stack} reprojection arrays are inconsistent.")
    unique = np.unique(groups)
    if not np.array_equal(unique, np.arange(unique.size)):
        raise ValueError(f"{stack} group_idx is not contiguous zero-based.")
    for group in unique:
        rows = np.flatnonzero(groups == group)
        if rows.size != 10 or not np.array_equal(weights[rows], np.arange(10)):
            raise ValueError(f"{stack} group {group} is not stored in exact weight order 0..9.")
    return groups, weights, masks


def verify_stage1(stage1_root: Path, subject_id: str) -> dict[str, Any]:
    manifest_path = stage1_root / "stage1_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Stage-1 manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    psf = manifest.get("psf", {})
    if manifest.get("status") != "PASS" or manifest.get("subject_id") != subject_id:
        raise ValueError("Stage-1 status/subject provenance is invalid.")
    if psf.get("enabled") is not True or psf.get("n_samples") != 8 or psf.get("seed") != 20260921:
        raise ValueError("Stage-1 provenance does not prove deterministic PSF K=8.")
    outputs = manifest.get("signal_outputs", {})
    for stack in STACKS:
        entry = outputs.get(stack, {})
        path = stage1_root / str(entry.get("relative_path", ""))
        if not path.is_file() or sha256(path) != entry.get("sha256"):
            raise ValueError(f"Stage-1 {stack} signal hash/provenance validation failed.")
        validate_signal_rows(load_signal(path), stack)
    sums = stage1_root / "TRANSFER_SHA256SUMS.txt"
    if not sums.is_file():
        raise FileNotFoundError(f"Stage-1 transfer checksums are missing: {sums}")
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        target = stage1_root / relative.strip()
        if not target.is_file() or sha256(target) != digest:
            raise ValueError(f"TRANSFER_SHA256SUMS mismatch: {relative}")
    return manifest


def write_synthetic_mat(signal_path: Path, target: Path, stack: str) -> None:
    data = load_signal(signal_path)
    groups, weights, masks = validate_signal_rows(data, stack)
    predicted = np.asarray(data["predicted"], dtype=np.float32)
    height, width = predicted.shape[1:]
    volume = np.zeros((height, width, 10, len(np.unique(groups))), dtype=np.float32)
    for group in np.unique(groups):
        for row in np.flatnonzero(groups == group):
            volume[:, :, int(weights[row]), group] = np.where(masks[row] & np.isfinite(predicted[row]), predicted[row], 0.0)
    savemat(target, {"Mag_synthetic": volume, "group_idx": np.arange(volume.shape[3], dtype=np.int32), "weight_idx": np.arange(10, dtype=np.int32)}, do_compression=True)


def run_matcher(source: Path, signal: Path, output: Path, matlab: str) -> None:
    matlab_dir = Path(__file__).resolve().parent / "matlab"
    quote = lambda value: str(value).replace("'", "''")
    command = f"addpath('{quote(matlab_dir)}'); build_synthetic_apparent_maps('{quote(source)}','{quote(signal)}','{quote(output)}');"
    subprocess.run([matlab, "-batch", command], check=True)
    if not output.is_file():
        raise RuntimeError(f"MATLAB did not create {output}.")


def render_signal_qc(signal_root: Path, output: Path) -> None:
    import matplotlib.pyplot as plt

    output.mkdir()
    for stack in STACKS:
        data = load_signal(signal_root / f"signal_reprojection_{stack}.npz")
        groups, weights, masks = validate_signal_rows(data, stack)
        for group in np.unique(groups):
            rows = np.flatnonzero(groups == group)
            figure, axes = plt.subplots(3, 10, figsize=(20, 6), constrained_layout=True)
            observed = np.asarray(data["observed"])[rows]
            vmax = float(np.percentile(observed[masks[rows]], 99.5))
            residual = np.abs(np.asarray(data["residual"])[rows])
            rmax = float(np.percentile(residual[masks[rows] & np.isfinite(residual)], 99.5))
            for column, row in enumerate(rows):
                for line, (key, cmap, upper) in enumerate((("observed", "gray", vmax), ("predicted", "gray", vmax), ("residual", "magma", rmax))):
                    values = np.abs(data[key][row]) if key == "residual" else data[key][row]
                    axes[line, column].imshow(np.ma.masked_where(~masks[row], values), cmap=cmap, vmin=0, vmax=max(upper, 1e-6), origin="lower")
                    axes[line, column].axis("off")
                    if line == 0: axes[line, column].set_title(f"w{int(weights[row])}")
                    if column == 0: axes[line, column].set_ylabel(key)
            figure.suptitle(f"{stack.upper()} group {int(group):02d}: Stage-1 PSF K=8")
            figure.savefig(output / f"{stack}_group_{int(group):02d}_weights_0_9.png", dpi=130)
            plt.close(figure)


def compute_metrics(reference_root: Path, preprocessed_root: Path, signal_root: Path, apparent_root: Path, mask_bundle: Path, output: Path, subject_id: str) -> dict[str, Any]:
    reference = load_verified_reference(reference_root, subject_id, preprocessed_root)
    with np.load(mask_bundle / "sax_myocardium_masks.npz", allow_pickle=False) as data:
        myocardium = {name: np.asarray(data[name], dtype=bool) for name in ("myocardium_full", "myocardium_core_legacy", "myocardium_core_1px")}
    quantitative_rows, signal_rows, myo_slice_rows, myo_summary_rows = [], [], [], []
    quantitative_pool: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {"T1": [], "T2": []}
    signal_pool = []
    for stack in STACKS:
        signal = load_signal(signal_root / f"signal_reprojection_{stack}.npz")
        groups, weights, masks = validate_signal_rows(signal, stack)
        for weight in range(10):
            rows = np.flatnonzero(weights == weight)
            signal_rows.append({"schema": "signal_metrics/v2", "subject_id": subject_id, "stack": stack, "weight_idx": weight, "region": "global_prepared_support", **signal_agreement_metrics(signal["observed"][rows], signal["predicted"][rows], masks[rows])})
        signal_pool.append((signal["observed"], signal["predicted"], masks))
        t1, t2, synthetic_valid, map_groups = _load_mapping(apparent_root / f"{stack}_apparent_mapping.mat")
        native = reference.stacks[stack]
        if t1.shape != native.t1_ms.shape or not np.array_equal(map_groups, native.group_idx):
            raise ValueError(f"{stack} synthetic maps do not match the verified native reference.")
        support = np.stack([np.all(masks[groups == group], axis=0) for group in map_groups])
        for parameter, ref, pred in (("T1", native.t1_ms, t1), ("T2", native.t2_ms, t2)):
            valid = native.valid_mask & synthetic_valid & support & np.isfinite(ref) & np.isfinite(pred)
            quantitative_rows.append({"subject_id": subject_id, "parameter": parameter, "stack": stack, "region": "global_common_support", "units": "ms", **agreement_metrics(ref, pred, valid)})
            quantitative_pool[parameter].append((ref, pred, valid))
            if stack == "sax":
                for region, roi in myocardium.items():
                    roi_valid = valid & roi
                    metric = agreement_metrics(ref, pred, roi_valid)
                    paired = roi_valid & np.isfinite(ref) & np.isfinite(pred)
                    native_mean = float(ref[paired].mean()) if paired.any() else float("nan")
                    synthetic_mean = float(pred[paired].mean()) if paired.any() else float("nan")
                    myo_summary_rows.append({"subject_id": subject_id, "parameter": parameter, "stack": stack, "region": region, "units": "ms", "native_mean_ms": native_mean, "synthetic_mean_ms": synthetic_mean, "relative_bias_percent": 100 * float(metric["bias_ms"]) / native_mean if native_mean else float("nan"), **metric})
                    for group in map_groups:
                        group_valid = roi_valid[group]
                        group_metric = agreement_metrics(ref[group], pred[group], group_valid)
                        paired_group = group_valid & np.isfinite(ref[group]) & np.isfinite(pred[group])
                        native_group = float(ref[group][paired_group].mean()) if paired_group.any() else float("nan")
                        synthetic_group = float(pred[group][paired_group].mean()) if paired_group.any() else float("nan")
                        myo_slice_rows.append({"subject_id": subject_id, "parameter": parameter, "stack": stack, "group_idx": int(group), "region": region, "units": "ms", "native_mean_ms": native_group, "synthetic_mean_ms": synthetic_group, "relative_bias_percent": 100 * float(group_metric["bias_ms"]) / native_group if native_group else float("nan"), **group_metric})
    observed = np.concatenate([x[0].ravel() for x in signal_pool]); predicted = np.concatenate([x[1].ravel() for x in signal_pool]); support = np.concatenate([x[2].ravel() for x in signal_pool])
    signal_overall = signal_agreement_metrics(observed, predicted, support)
    signal_rows.append({"schema": "signal_metrics/v2", "subject_id": subject_id, "stack": "ALL", "weight_idx": "ALL", "region": "global_prepared_support", **signal_overall})
    quantitative_overall = {}
    for parameter, entries in quantitative_pool.items():
        quantitative_overall[parameter] = agreement_metrics(np.concatenate([x[0].ravel() for x in entries]), np.concatenate([x[1].ravel() for x in entries]), np.concatenate([x[2].ravel() for x in entries]))
    write_csv(signal_rows, output / "signal_metrics.csv"); write_csv(quantitative_rows, output / "quantitative_metrics.csv")
    write_csv(myo_slice_rows, output / "quantitative_agreement_myocardium_per_slice.csv"); write_csv(myo_summary_rows, output / "quantitative_agreement_myocardium_summary.csv")
    summary = {"signal_overall": signal_overall, "quantitative_overall": quantitative_overall, "primary_myocardium_region": "myocardium_core_1px", "secondary_myocardium_regions": ["myocardium_full", "myocardium_core_legacy"]}
    write_json(summary, output / "metrics_summary.json")
    return summary
