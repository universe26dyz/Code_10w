"""Artifact export, strict paired supports, and signal-domain aggregations for D2."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from trad.evaluation.scmr.metrics import signal_agreement_metrics
from trad.modules.module_08_inference_export.reprojection import export_native_plane_reprojections


STACKS = ("sax", "2ch", "4ch")
SIGNAL_METRICS = ("bias_signal", "MAE_signal", "RMSE_signal", "NRMSE", "Pearson_r", "NCC", "SSIM")
ARTIFACT_REQUIRED = {"observed", "predicted", "residual", "group_idx", "weight_idx", "timing9_ms", "masks", "tr_ms", "vps"}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_empty_output(output: str | Path) -> Path:
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty D2 output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def export_k8_reprojections(model: Any, space: Any, prepared_inputs: Iterable[str | Path], output_dir: str | Path, *, evaluation_seed: int) -> dict[str, Path]:
    """Reuse the baseline native-plane exporter with a fixed K=8 contract only."""

    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    temporary = output / "_temporary_baseline_export"
    if temporary.exists():
        raise FileExistsError(f"Temporary D2 export path already exists: {temporary}")
    paths = export_native_plane_reprojections(
        model, space, list(prepared_inputs), temporary,
        output_psf={"enabled": True, "n_samples": 8}, evaluation_seed=evaluation_seed, export_parameter_maps=False,
    )
    result: dict[str, Path] = {}
    for stack in STACKS:
        source = paths.get(f"signal_{stack}")
        if source is None:
            raise ValueError(f"Baseline exporter did not produce the required {stack} signal reprojection.")
        target = output / f"signal_reprojection_{stack}_K8.npz"
        source.replace(target)
        result[stack] = target
    temporary.rmdir()
    return result


def strict_paired_signal_support(masks: np.ndarray, observed: np.ndarray, bloch_prediction: np.ndarray, mlp_prediction: np.ndarray) -> np.ndarray:
    """The D2 primary-comparison support, with no implicit finite-row dropping."""

    masks = np.asarray(masks, dtype=bool)
    if not (masks.shape == np.shape(observed) == np.shape(bloch_prediction) == np.shape(mlp_prediction)):
        raise ValueError("D2 strict paired signal support requires equal observed/predicted/mask shapes.")
    return masks & np.isfinite(observed) & np.isfinite(bloch_prediction) & np.isfinite(mlp_prediction)


def signal_metric_row(observed: np.ndarray, predicted: np.ndarray, mask: np.ndarray, *, min_pixels: int = 16) -> dict[str, float | int | str]:
    return signal_agreement_metrics(observed, predicted, mask, min_pixels=min_pixels)


def _load_artifact(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        missing = ARTIFACT_REQUIRED.difference(data.files)
        if missing:
            raise ValueError(f"D2 K8 artifact lacks required schema fields {sorted(missing)}: {path}")
        return {key: np.asarray(data[key]) for key in ARTIFACT_REQUIRED}


def _slice_rows(method: str, stack: str, artifact_path: str | Path, *, support: np.ndarray | None = None) -> list[dict[str, Any]]:
    data = _load_artifact(artifact_path)
    observed, predicted, masks = data["observed"], data["predicted"], data["masks"]
    if support is not None and support.shape != masks.shape:
        raise ValueError(f"D2 paired support shape differs from {method} {stack} artifact.")
    rows: list[dict[str, Any]] = []
    for index in range(observed.shape[0]):
        current_support = masks[index] if support is None else support[index]
        rows.append({
            "method": method, "stack": stack, "group_idx": int(data["group_idx"][index]), "weight_idx": int(data["weight_idx"][index]),
            "support_provenance": "method_valid_mask AND finite(observed) AND finite(method_prediction)" if support is None else "Bloch_valid AND FrozenMLP_valid AND finite(observed) AND finite(Bloch_prediction) AND finite(FrozenMLP_prediction)",
            **signal_metric_row(observed[index], predicted[index], current_support),
        })
    return rows


def method_specific_slice_rows(method: str, artifact_paths: dict[str, Path]) -> list[dict[str, Any]]:
    return [row for stack in STACKS for row in _slice_rows(method, stack, artifact_paths[stack])]


def _validate_pair(bloch_path: Path, mlp_path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    bloch, mlp = _load_artifact(bloch_path), _load_artifact(mlp_path)
    for field in ("observed", "group_idx", "weight_idx", "timing9_ms", "masks", "tr_ms", "vps"):
        if bloch[field].shape != mlp[field].shape or not np.array_equal(bloch[field], mlp[field], equal_nan=True):
            raise ValueError(f"D2 Bloch/FrozenMLP input provenance mismatch for field {field}.")
    return bloch, mlp


def paired_slice_rows(bloch_paths: dict[str, Path], mlp_paths: dict[str, Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comparison_rows: list[dict[str, Any]] = []
    disagreement_rows: list[dict[str, Any]] = []
    for stack in STACKS:
        bloch, mlp = _validate_pair(bloch_paths[stack], mlp_paths[stack])
        support = strict_paired_signal_support(bloch["masks"], bloch["observed"], bloch["predicted"], mlp["predicted"])
        for method, data in (("Bloch", bloch), ("FrozenMLP", mlp)):
            comparison_rows.extend(_slice_rows(method, stack, bloch_paths[stack] if method == "Bloch" else mlp_paths[stack], support=support))
        for index in range(support.shape[0]):
            disagreement_rows.append({
                "stack": stack, "group_idx": int(bloch["group_idx"][index]), "weight_idx": int(bloch["weight_idx"][index]),
                "support_provenance": "D2 strict paired support", **signal_metric_row(bloch["predicted"][index], mlp["predicted"][index], support[index]),
            })
    return comparison_rows, disagreement_rows


def aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result: list[dict[str, Any]] = []
    for identifier, members in sorted(groups.items()):
        weights = np.asarray([float(member["N"]) for member in members], dtype=float)
        item: dict[str, Any] = dict(zip(keys, identifier)) | {"units": "original_input_intensity", "aggregation": "N-weighted mean of per-observation metrics; SSIM is never pooled across 3-D data", "N": int(weights.sum())}
        for metric in SIGNAL_METRICS:
            values = np.asarray([float(member[metric]) for member in members], dtype=float)
            valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
            item[metric] = float(np.average(values[valid], weights=weights[valid])) if valid.any() else float("nan")
        result.append(item)
    return result


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(rows)


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
