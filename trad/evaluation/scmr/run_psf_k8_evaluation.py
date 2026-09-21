"""Evaluate an existing Trad checkpoint with deterministic native-plane PSF K=8."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.io import savemat

_TRAD_ROOT = Path(__file__).resolve().parents[2]
if str(_TRAD_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAD_ROOT))

from modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset
from modules.module_07_objective_training.trad_trainer import build_training_model, load_checkpoint
from modules.module_07_objective_training.training_space import TrainingSpace
from modules.module_08_inference_export.reprojection import export_native_plane_reprojections

from .build_native_reference import _load_mapping
from .metrics import agreement_metrics
from .reference_2d import STACKS, load_verified_reference, sha256


def _write_json(value: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _load_signal(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _validate_signal_rows(data: dict[str, np.ndarray], stack: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _write_synthetic_mat(signal_path: Path, target: Path, stack: str) -> None:
    data = _load_signal(signal_path)
    groups, weights, masks = _validate_signal_rows(data, stack)
    predicted = np.asarray(data["predicted"], dtype=np.float32)
    h, w = predicted.shape[1:]
    volume = np.zeros((h, w, 10, len(np.unique(groups))), dtype=np.float32)
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        for row in rows:
            weight = int(weights[row])
            volume[:, :, weight, group] = np.where(masks[row] & np.isfinite(predicted[row]), predicted[row], 0.0)
    savemat(target, {"Mag_synthetic": volume, "group_idx": np.arange(volume.shape[3], dtype=np.int32), "weight_idx": np.arange(10, dtype=np.int32)}, do_compression=True)


def _matlab_quote(value: str | Path) -> str:
    return str(value).replace("'", "''")


def _run_matcher(source: Path, signal: Path, output: Path, matlab: str) -> None:
    matlab_dir = Path(__file__).resolve().parent / "matlab"
    command = (
        f"addpath('{_matlab_quote(matlab_dir)}'); "
        f"build_synthetic_apparent_maps('{_matlab_quote(source)}','{_matlab_quote(signal)}','{_matlab_quote(output)}');"
    )
    subprocess.run([matlab, "-batch", command], check=True)
    if not output.is_file():
        raise RuntimeError(f"MATLAB did not create {output}.")


def _render_signal_qc(signal_root: Path, output: Path) -> None:
    import matplotlib.pyplot as plt

    output.mkdir()
    for stack in STACKS:
        data = _load_signal(signal_root / f"signal_reprojection_{stack}.npz")
        groups, weights, masks = _validate_signal_rows(data, stack)
        for group in np.unique(groups):
            rows = np.flatnonzero(groups == group)
            figure, axes = plt.subplots(3, 10, figsize=(20, 6), constrained_layout=True)
            finite_observed = data["observed"][rows][masks[rows]]
            vmax = float(np.percentile(finite_observed, 99.5)) if finite_observed.size else 1.0
            residual = np.abs(data["residual"][rows])
            residual_values = residual[masks[rows] & np.isfinite(residual)]
            rmax = float(np.percentile(residual_values, 99.5)) if residual_values.size else 1.0
            for column, row in enumerate(rows):
                for line, (key, cmap, upper) in enumerate((("observed", "gray", vmax), ("predicted", "gray", vmax), ("residual", "magma", rmax))):
                    values = np.ma.masked_where(~masks[row], np.abs(data[key][row]) if key == "residual" else data[key][row])
                    axes[line, column].imshow(values, cmap=cmap, vmin=0.0, vmax=max(upper, 1e-6), origin="lower")
                    axes[line, column].axis("off")
                    if line == 0:
                        axes[line, column].set_title(f"w{int(weights[row])}")
                    if column == 0:
                        axes[line, column].set_ylabel(key)
            figure.suptitle(f"{stack.upper()} group {int(group):02d}: PSF K=8")
            figure.savefig(output / f"{stack}_group_{int(group):02d}_weights_0_9.png", dpi=130)
            plt.close(figure)


def _metrics(
    reference_root: Path,
    preprocessed_root: Path,
    signal_root: Path,
    apparent_root: Path,
    mask_bundle: Path | None,
    output: Path,
    subject_id: str,
) -> dict[str, Any]:
    reference = load_verified_reference(reference_root, subject_id, preprocessed_root)
    quantitative_rows: list[dict[str, Any]] = []
    myocardial_slice_rows: list[dict[str, Any]] = []
    myocardial_summary_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []
    quantitative_pool: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {"T1": [], "T2": []}
    signal_pool: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    myocardium = None
    if mask_bundle is not None:
        with np.load(mask_bundle / "sax_myocardium_masks.npz", allow_pickle=False) as data:
            myocardium = {key: np.asarray(data[key], dtype=bool) for key in ("myocardium_full", "myocardium_core")}
    for stack in STACKS:
        signal = _load_signal(signal_root / f"signal_reprojection_{stack}.npz")
        groups, weights, masks = _validate_signal_rows(signal, stack)
        observed, predicted = signal["observed"], signal["predicted"]
        for weight in range(10):
            rows = np.flatnonzero(weights == weight)
            metric = agreement_metrics(observed[rows], predicted[rows], masks[rows])
            signal_rows.append({"subject_id": subject_id, "stack": stack, "weight_idx": weight, "region": "global_prepared_support", **metric})
        signal_pool.append((observed, predicted, masks))

        t1, t2, synthetic_valid, map_groups = _load_mapping(apparent_root / f"{stack}_apparent_mapping.mat")
        native = reference.stacks[stack]
        if t1.shape != native.t1_ms.shape or not np.array_equal(map_groups, native.group_idx):
            raise ValueError(f"{stack} synthetic apparent maps do not match the verified native reference.")
        support = np.stack([np.all(masks[groups == group], axis=0) for group in map_groups])
        for parameter, ref, pred in (("T1", native.t1_ms, t1), ("T2", native.t2_ms, t2)):
            valid = native.valid_mask & synthetic_valid & support & np.isfinite(ref) & np.isfinite(pred)
            metric = agreement_metrics(ref, pred, valid)
            quantitative_rows.append({"subject_id": subject_id, "parameter": parameter, "stack": stack, "region": "global_common_support", "units": "ms", **metric})
            quantitative_pool[parameter].append((ref, pred, valid))
            if stack == "sax" and myocardium is not None:
                for region, roi in myocardium.items():
                    roi_valid = valid & roi
                    roi_metric = agreement_metrics(ref, pred, roi_valid)
                    paired = roi_valid & np.isfinite(ref) & np.isfinite(pred)
                    native_mean = float(ref[paired].mean()) if paired.any() else float("nan")
                    synthetic_mean = float(pred[paired].mean()) if paired.any() else float("nan")
                    summary_row = {"subject_id": subject_id, "parameter": parameter, "stack": stack, "region": region, "units": "ms", "native_mean_ms": native_mean, "synthetic_mean_ms": synthetic_mean, "relative_bias_percent": 100.0 * float(roi_metric["bias_ms"]) / native_mean if native_mean else float("nan"), **roi_metric}
                    myocardial_summary_rows.append(summary_row)
                    for group in map_groups:
                        group_valid = roi_valid[group]
                        group_metric = agreement_metrics(ref[group], pred[group], group_valid)
                        paired_group = group_valid & np.isfinite(ref[group]) & np.isfinite(pred[group])
                        group_native = float(ref[group][paired_group].mean()) if paired_group.any() else float("nan")
                        group_synthetic = float(pred[group][paired_group].mean()) if paired_group.any() else float("nan")
                        myocardial_slice_rows.append({"subject_id": subject_id, "parameter": parameter, "stack": stack, "group_idx": int(group), "region": region, "units": "ms", "native_mean_ms": group_native, "synthetic_mean_ms": group_synthetic, "relative_bias_percent": 100.0 * float(group_metric["bias_ms"]) / group_native if group_native else float("nan"), **group_metric})
    signal_ref = np.concatenate([item[0].ravel() for item in signal_pool])
    signal_pred = np.concatenate([item[1].ravel() for item in signal_pool])
    signal_mask = np.concatenate([item[2].ravel() for item in signal_pool])
    signal_overall = agreement_metrics(signal_ref, signal_pred, signal_mask)
    signal_rows.append({"subject_id": subject_id, "stack": "ALL", "weight_idx": "ALL", "region": "global_prepared_support", **signal_overall})
    quantitative_overall = {}
    for parameter, entries in quantitative_pool.items():
        ref = np.concatenate([item[0].ravel() for item in entries])
        pred = np.concatenate([item[1].ravel() for item in entries])
        mask = np.concatenate([item[2].ravel() for item in entries])
        quantitative_overall[parameter] = agreement_metrics(ref, pred, mask)
    _write_csv(signal_rows, output / "signal_metrics.csv")
    _write_csv(quantitative_rows, output / "quantitative_metrics.csv")
    if myocardial_slice_rows:
        _write_csv(myocardial_slice_rows, output / "quantitative_agreement_myocardium_per_slice.csv")
        _write_csv(myocardial_summary_rows, output / "quantitative_agreement_myocardium_summary.csv")
    summary = {
        "signal_overall": signal_overall,
        "quantitative_overall": quantitative_overall,
        "interpretation": "Consistency with verified native 2-D dictionary maps; not ground-truth accuracy.",
        "myocardium_metrics_included": myocardium is not None,
    }
    _write_json(summary, output / "metrics_summary.json")
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device is unavailable: {device}")
    if device.type == "cpu":
        checkpoint_probe = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if "inr.encoding.params" in checkpoint_probe.get("model_state", {}):
            raise RuntimeError(
                "This checkpoint stores a tiny-cuda-nn HashGrid and cannot be evaluated by the "
                "vendored CPU HashGrid backend without an unverified model conversion; use a CUDA/tiny-cuda-nn environment."
            )
    observations = [args.prepared_root / args.subject_id / stack / "observations.npz" for stack in STACKS]
    missing = [str(path) for path in observations if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing prepared observations: " + "; ".join(missing))
    load_verified_reference(args.native_reference, args.subject_id, args.preprocessed_root)
    if args.mask_bundle is not None:
        mask_manifest = json.loads((args.mask_bundle / "manifest.json").read_text(encoding="utf-8"))
        if mask_manifest.get("status") != "PASS" or mask_manifest.get("subject_id") != args.subject_id:
            raise ValueError("Myocardium mask bundle provenance is not PASS for the requested subject.")
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty PSF K=8 evaluation: {output}")
    output.mkdir(parents=True, exist_ok=True)
    signal_root = output / "signals"; signal_root.mkdir()
    synthetic_root = output / "synthetic_signal_mat"; synthetic_root.mkdir()
    apparent_root = output / "synthetic_apparent_maps"; apparent_root.mkdir()
    metrics_root = output / "metrics"; metrics_root.mkdir()

    dataset = QuantPointDataset(observations, device=device)
    model, _, _ = build_training_model(dataset, config, _TRAD_ROOT / "configs" / "protocol_hhz_v1.yaml", device)
    checkpoint = load_checkpoint(args.checkpoint, model, device)
    space = TrainingSpace.from_state_dict(checkpoint["training_space"])
    if not torch.equal(model.rigid_psf.axisangle.detach().cpu(), checkpoint["trained_axisangle_train"].detach().cpu()):
        raise ValueError("Loaded model pose differs from checkpoint trained_axisangle_train.")

    export_native_plane_reprojections(
        model,
        space,
        observations,
        signal_root,
        output_psf={"enabled": True, "n_samples": 8},
        evaluation_seed=args.seed,
        export_parameter_maps=False,
    )
    for stack in STACKS:
        signal_mat = synthetic_root / f"{stack}_synthetic_signal.mat"
        mapping_mat = apparent_root / f"{stack}_apparent_mapping.mat"
        _write_synthetic_mat(signal_root / f"signal_reprojection_{stack}.npz", signal_mat, stack)
        _run_matcher(args.preprocessed_root / args.subject_id / stack / "preprocessed.mat", signal_mat, mapping_mat, args.matlab_executable)
    _render_signal_qc(signal_root, output / "qc_all_slices_all_weights")
    summary = _metrics(args.native_reference, args.preprocessed_root, signal_root, apparent_root, args.mask_bundle, metrics_root, args.subject_id)

    manifest = {
        "schema": "trad_psf_k8_native_reprojection/v1",
        "status": "PASS",
        "subject_id": args.subject_id,
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": sha256(args.checkpoint)},
        "config": {"path": str(args.config.resolve()), "sha256": sha256(args.config)},
        "device": str(device),
        "psf": {"enabled": True, "n_samples": 8, "seed": args.seed, "sampling_mode": "random Gaussian local PSF", "operation_order": "local PSF sample -> final Stage-B rigid pose -> INR -> Bloch signal per sample -> mean signal"},
        "final_pose_source": "model.pt model_state rigid_psf.axisangle; equality checked against trained_axisangle_train",
        "final_pose_sha256": _tensor_sha256(model.rigid_psf.axisangle),
        "dictionary_matching": {
            "matlab_wrapper": str((Path(__file__).resolve().parent / "matlab" / "build_synthetic_apparent_maps.m").resolve()),
            "matcher": str((_TRAD_ROOT / "modules" / "module_01_preprocess_matlab" / "hhz_original" / "Utility" / "function_T1T2_10HB_bssfp.m").resolve()),
            "physics_parameters_changed": False,
        },
        "prepared_observations": {stack: {"path": str(path.resolve()), "sha256": sha256(path)} for stack, path in zip(STACKS, observations)},
        "native_reference_manifest_sha256": sha256(args.native_reference / "native_reference_manifest.json"),
        "mask_bundle_manifest_sha256": sha256(args.mask_bundle / "manifest.json") if args.mask_bundle is not None else None,
        "metrics": summary,
    }
    _write_json(manifest, output / "evaluation_manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument("--preprocessed-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--native-reference", required=True, type=Path)
    parser.add_argument("--mask-bundle", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", default=20260921, type=int)
    parser.add_argument("--matlab-executable", default="matlab")
    args = parser.parse_args()
    try:
        result = run(args)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    main()
