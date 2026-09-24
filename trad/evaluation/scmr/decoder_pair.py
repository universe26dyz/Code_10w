"""Read-only loaders and metric helpers for paired decoder evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import agreement_metrics


STACKS = ("sax", "2ch", "4ch")
PARAMETERS = ("T1", "T2")
_SCHEMA = "map_domain_psf_comparison_stackwise/v1"


@dataclass(frozen=True)
class EvaluationArtifact:
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    arrays: dict[str, dict[str, dict[str, np.ndarray]]]


@dataclass(frozen=True)
class DecoderPair:
    subject_id: str
    trad: EvaluationArtifact
    mlp: EvaluationArtifact


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _load_artifact(root: str | Path, expected_decoder: str, subject_id: str) -> EvaluationArtifact:
    root = Path(root).expanduser().resolve()
    manifest_path = root / "map_domain_psf_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"PSF manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("subject_id") != subject_id:
        raise ValueError(f"PSF manifest subject_id={manifest.get('subject_id')!r}, expected {subject_id!r}.")
    if manifest.get("decoder_type") != expected_decoder:
        raise ValueError(f"PSF manifest decoder_type={manifest.get('decoder_type')!r}, expected {expected_decoder!r}.")
    arrays: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for parameter in PARAMETERS:
        path = root / f"{parameter}_native_map_domain_psf_comparison.npz"
        if not path.is_file():
            raise FileNotFoundError(f"PSF comparison is missing: {path}")
        data = _load_npz(path)
        if str(data.get("schema", "")) != _SCHEMA:
            raise ValueError(f"{path} has unsupported schema {data.get('schema')!r}.")
        if tuple(str(value) for value in data.get("stack_names", ())) != STACKS:
            raise ValueError(f"{path} must record stack_names={STACKS}.")
        arrays[parameter] = {}
        for stack in STACKS:
            keys = {"prediction": f"{stack}_predicted_ms", "native": f"{stack}_native_reference_ms", "support": f"{stack}_common_support"}
            missing = [name for name in keys.values() if name not in data]
            if missing:
                raise ValueError(f"{path} is missing {missing}.")
            prediction = np.asarray(data[keys["prediction"]], dtype=np.float32)
            native = np.asarray(data[keys["native"]], dtype=np.float32)
            support = np.asarray(data[keys["support"]], dtype=bool)
            if prediction.ndim != 3 or prediction.shape != native.shape or support.shape != native.shape:
                raise ValueError(f"{path} {parameter}/{stack} requires matching [group,row,column] prediction/native/support arrays.")
            arrays[parameter][stack] = {"prediction": prediction, "native": native, "support": support}
    return EvaluationArtifact(root=root, manifest_path=manifest_path, manifest=manifest, arrays=arrays)


def _require_shared_contract(trad: EvaluationArtifact, mlp: EvaluationArtifact) -> None:
    fields = (
        "method",
        "reconstruction_method",
        "domain",
        "n_samples",
        "seed",
        "geometry_reprojection_path",
        "geometry_source",
        "psf_implementation",
        "physical_resolution_thickness_source",
    )
    for field in fields:
        if trad.manifest.get(field) != mlp.manifest.get(field):
            raise ValueError(f"Trad/MLP PSF manifest mismatch for {field}: {trad.manifest.get(field)!r} != {mlp.manifest.get(field)!r}.")
    expected = {"method": "shared_svr", "reconstruction_method": "shared_svr", "domain": "map"}
    for field, value in expected.items():
        if trad.manifest.get(field) != value:
            raise ValueError(f"Paired decoder evaluation requires {field}={value!r}.")
    trad_native = (trad.manifest.get("native_reference") or {}).get("manifest_sha256")
    mlp_native = (mlp.manifest.get("native_reference") or {}).get("manifest_sha256")
    if not trad_native or trad_native != mlp_native:
        raise ValueError("Trad and MLP PSF manifests must have the same native-reference manifest SHA256.")


def _require_same_native_reference(trad: EvaluationArtifact, mlp: EvaluationArtifact) -> None:
    for parameter in PARAMETERS:
        for stack in STACKS:
            trad_native = trad.arrays[parameter][stack]["native"]
            mlp_native = mlp.arrays[parameter][stack]["native"]
            if trad_native.shape != mlp_native.shape:
                raise ValueError(f"Trad/MLP native-reference shape mismatch for {parameter}/{stack}: {trad_native.shape} != {mlp_native.shape}.")
            if not np.allclose(trad_native, mlp_native, rtol=1e-5, atol=1e-4, equal_nan=True):
                raise ValueError(f"Trad/MLP native-reference value mismatch for {parameter}/{stack}.")


def load_decoder_pair(subject_id: str, trad_eval: str | Path, mlp_eval: str | Path) -> DecoderPair:
    """Load existing PSF artifacts and enforce their shared scientific contract.

    Final pose values are intentionally not compared: these are independently
    optimized reconstructions.  The common geometry provenance contract is.
    """

    trad = _load_artifact(trad_eval, "Bloch", subject_id)
    mlp = _load_artifact(mlp_eval, "FrozenMLP", subject_id)
    _require_shared_contract(trad, mlp)
    _require_same_native_reference(trad, mlp)
    return DecoderPair(subject_id=subject_id, trad=trad, mlp=mlp)


def _slice_ssim_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    values = np.asarray([row["SSIM"] for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return {
        "SSIM_slice_mean": float(values.mean()) if values.size else float("nan"),
        "SSIM_slice_median": float(np.median(values)) if values.size else float("nan"),
    }


def _method_metrics(pair: DecoderPair, artifact: EvaluationArtifact, method: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    per_slice: list[dict[str, Any]] = []
    per_stack: list[dict[str, Any]] = []
    global_metrics: dict[str, Any] = {}
    for parameter in PARAMETERS:
        pooled: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        parameter_rows: list[dict[str, Any]] = []
        for stack in STACKS:
            values = artifact.arrays[parameter][stack]
            native, prediction, support = values["native"], values["prediction"], values["support"]
            stack_rows: list[dict[str, Any]] = []
            for group in range(native.shape[0]):
                metric = agreement_metrics(native[group], prediction[group], support[group])
                row = {"subject_id": pair.subject_id, "method": method, "parameter": parameter, "stack": stack, "group_idx": group,
                       "support_definition": f"{method}_existing_common_support", "comparison_definition": "consistency/agreement with verified native 2-D dictionary maps; not independent ground-truth accuracy", "units": "ms", **metric}
                per_slice.append(row)
                stack_rows.append(row)
                parameter_rows.append(row)
            metric = agreement_metrics(native, prediction, support)
            per_stack.append({"subject_id": pair.subject_id, "method": method, "parameter": parameter, "stack": stack, "group_idx": "ALL",
                              "support_definition": f"{method}_existing_common_support", "comparison_definition": "consistency/agreement with verified native 2-D dictionary maps; not independent ground-truth accuracy", "units": "ms", **metric, **_slice_ssim_summary(stack_rows)})
            pooled.append((native, prediction, support))
        native = np.concatenate([value[0].ravel() for value in pooled])
        prediction = np.concatenate([value[1].ravel() for value in pooled])
        support = np.concatenate([value[2].ravel() for value in pooled])
        global_metrics[parameter] = agreement_metrics(native, prediction, support) | _slice_ssim_summary(parameter_rows)
    return per_slice, per_stack, global_metrics


def _delta(trad: dict[str, Any], mlp: dict[str, Any]) -> dict[str, float]:
    return {f"delta_mlp_minus_trad_{key}": float(mlp[key]) - float(trad[key]) for key in ("bias_ms", "MAE_ms", "RMSE_ms", "NRMSE", "Pearson_r", "NCC", "SSIM")}


def _paired_metrics(pair: DecoderPair) -> dict[str, Any]:
    per_slice: list[dict[str, Any]] = []
    per_stack: list[dict[str, Any]] = []
    global_metrics: dict[str, Any] = {}
    for parameter in PARAMETERS:
        pooled: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        parameter_rows: dict[str, list[dict[str, Any]]] = {"Bloch": [], "FrozenMLP": []}
        for stack in STACKS:
            trad = pair.trad.arrays[parameter][stack]
            mlp = pair.mlp.arrays[parameter][stack]
            stack_rows: dict[str, list[dict[str, Any]]] = {"Bloch": [], "FrozenMLP": []}
            paired_support = trad["support"] & mlp["support"]
            for group in range(trad["native"].shape[0]):
                metrics = {"Bloch": agreement_metrics(trad["native"][group], trad["prediction"][group], paired_support[group]),
                           "FrozenMLP": agreement_metrics(trad["native"][group], mlp["prediction"][group], paired_support[group])}
                for method, metric in metrics.items():
                    row = {"subject_id": pair.subject_id, "method": method, "parameter": parameter, "stack": stack, "group_idx": group,
                           "support_definition": "paired_common_support = Bloch_common_support AND FrozenMLP_common_support",
                           "comparison_definition": "consistency/agreement with verified native 2-D dictionary maps; not independent ground-truth accuracy", "units": "ms", **metric}
                    per_slice.append(row)
                    stack_rows[method].append(row)
                    parameter_rows[method].append(row)
                per_slice.append({"subject_id": pair.subject_id, "method": "delta_mlp_minus_trad", "parameter": parameter, "stack": stack, "group_idx": group,
                                  "support_definition": "paired_common_support", "units": "ms", "N": metrics["Bloch"]["N"], **_delta(metrics["Bloch"], metrics["FrozenMLP"])})
            stack_metric = {"Bloch": agreement_metrics(trad["native"], trad["prediction"], paired_support),
                            "FrozenMLP": agreement_metrics(trad["native"], mlp["prediction"], paired_support)}
            for method, metric in stack_metric.items():
                per_stack.append({"subject_id": pair.subject_id, "method": method, "parameter": parameter, "stack": stack, "group_idx": "ALL",
                                  "support_definition": "paired_common_support = Bloch_common_support AND FrozenMLP_common_support", "units": "ms", **metric, **_slice_ssim_summary(stack_rows[method])})
            per_stack.append({"subject_id": pair.subject_id, "method": "delta_mlp_minus_trad", "parameter": parameter, "stack": stack, "group_idx": "ALL",
                              "support_definition": "paired_common_support", "units": "ms", "N": stack_metric["Bloch"]["N"], **_delta(stack_metric["Bloch"], stack_metric["FrozenMLP"])})
            pooled.append((trad["native"], trad["prediction"], mlp["prediction"], paired_support))
        native = np.concatenate([value[0].ravel() for value in pooled])
        trad_prediction = np.concatenate([value[1].ravel() for value in pooled])
        mlp_prediction = np.concatenate([value[2].ravel() for value in pooled])
        support = np.concatenate([value[3].ravel() for value in pooled])
        global_metric = {"Bloch": agreement_metrics(native, trad_prediction, support), "FrozenMLP": agreement_metrics(native, mlp_prediction, support)}
        global_metrics[parameter] = {method: metric | _slice_ssim_summary(parameter_rows[method]) for method, metric in global_metric.items()} | {"delta_mlp_minus_trad": _delta(global_metric["Bloch"], global_metric["FrozenMLP"])}
    return {"definition": "paired_common_support = Bloch existing common support AND FrozenMLP existing common support", "per_slice": per_slice, "per_stack": per_stack, "global": global_metrics}


def evaluate_pair_metrics(pair: DecoderPair) -> dict[str, Any]:
    """Evaluate existing and paired supports without mutating source artifacts."""

    trad_slice, trad_stack, trad_global = _method_metrics(pair, pair.trad, "Bloch")
    mlp_slice, mlp_stack, mlp_global = _method_metrics(pair, pair.mlp, "FrozenMLP")
    return {
        "definition": "consistency/agreement with verified native 2-D dictionary maps; not independent ground-truth accuracy",
        "method_specific_per_slice": trad_slice + mlp_slice,
        "method_specific_per_stack": trad_stack + mlp_stack,
        "method_specific_global": {parameter: {"Bloch": trad_global[parameter], "FrozenMLP": mlp_global[parameter]} for parameter in PARAMETERS},
        "paired_common_support": _paired_metrics(pair),
    }


def validate_verified_reference(pair: DecoderPair, reference: Any, reference_manifest_sha256: str) -> None:
    """Bind both comparisons to the requested verified native-reference bundle."""

    expected = (pair.trad.manifest.get("native_reference") or {}).get("manifest_sha256")
    if expected != reference_manifest_sha256:
        raise ValueError("PSF manifest native-reference SHA256 does not match the requested verified native-reference bundle.")
    for parameter, key in (("T1", "t1_ms"), ("T2", "t2_ms")):
        for stack in STACKS:
            expected_native = getattr(reference.stacks[stack], key)
            actual_native = pair.trad.arrays[parameter][stack]["native"]
            if actual_native.shape != expected_native.shape or not np.allclose(actual_native, expected_native, rtol=1e-5, atol=1e-4, equal_nan=True):
                raise ValueError(f"PSF comparison native reference does not match verified bundle for {parameter}/{stack}.")


def select_representative_groups(pair: DecoderPair) -> dict[str, int]:
    """Select one native group per stack by the largest T1/T2 paired support."""

    selected: dict[str, int] = {}
    for stack in STACKS:
        support = (
            pair.trad.arrays["T1"][stack]["support"]
            & pair.mlp.arrays["T1"][stack]["support"]
            & pair.trad.arrays["T2"][stack]["support"]
            & pair.mlp.arrays["T2"][stack]["support"]
        )
        selected[stack] = int(np.argmax(support.sum(axis=(1, 2))))
    return selected


def _paired_support(pair: DecoderPair, parameter: str, stack: str) -> np.ndarray:
    return pair.trad.arrays[parameter][stack]["support"] & pair.mlp.arrays[parameter][stack]["support"]


def _crop(data: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from .quality_control import _crop as existing_crop

    return existing_crop(data, mask)


def render_paired_figure2(pair: DecoderPair, selected: dict[str, int], output: str | Path, colorbars: Any) -> dict[str, Any]:
    """Render PSF-matched native-plane comparison using paired support only."""

    import matplotlib.pyplot as plt

    from .quality_control import DISPLAY_RANGES_MS, RESIDUAL_RANGES_MS

    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {"selected_groups": selected, "support_definition": "paired_common_support", "colorbars": colorbars.metadata}
    for parameter in PARAMETERS:
        panels = []
        for stack in STACKS:
            group = selected[stack]
            trad, mlp = pair.trad.arrays[parameter][stack], pair.mlp.arrays[parameter][stack]
            panels.append((stack, group, trad["native"][group], trad["prediction"][group], mlp["prediction"][group], _paired_support(pair, parameter, stack)[group]))
        if not all(mask.any() for *_, mask in panels):
            raise ValueError(f"No paired common-support pixels are available for Figure 2 {parameter}.")
        figure, axes = plt.subplots(3, 7, figsize=(15.4, 8.2), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 0.06, 0.06]})
        mapping_handle = residual_handle = None
        for row, (stack, group, native, trad, mlp, mask) in enumerate(panels):
            crop_native, crop_mask = _crop(native, mask)
            crop_trad, _ = _crop(trad, mask)
            crop_mlp, _ = _crop(mlp, mask)
            crop_trad_residual, _ = _crop(np.abs(trad - native), mask)
            crop_mlp_residual, _ = _crop(np.abs(mlp - native), mask)
            panels_to_render = (
                (crop_native, "Native 2-D", colorbars.mapping[parameter], DISPLAY_RANGES_MS[parameter], "map"),
                (crop_trad, "Bloch PSF-matched", colorbars.mapping[parameter], DISPLAY_RANGES_MS[parameter], "map"),
                (crop_mlp, "FrozenMLP PSF-matched", colorbars.mapping[parameter], DISPLAY_RANGES_MS[parameter], "map"),
                (crop_trad_residual, "|Bloch - Native|", colorbars.residual, RESIDUAL_RANGES_MS[parameter], "residual"),
                (crop_mlp_residual, "|FrozenMLP - Native|", colorbars.residual, RESIDUAL_RANGES_MS[parameter], "residual"),
            )
            for column, (data, title, cmap, limits, kind) in enumerate(panels_to_render):
                handle = axes[row, column].imshow(np.ma.masked_where(~crop_mask, data), cmap=cmap, vmin=limits[0], vmax=limits[1], origin="lower", interpolation="nearest", aspect="equal")
                if row == 0:
                    axes[row, column].set_title(title)
                axes[row, column].set_ylabel(f"{stack.upper()} / group {group}")
                axes[row, column].axis("off")
                if kind == "map":
                    mapping_handle = handle
                else:
                    residual_handle = handle
            axes[row, 5].axis("off")
            axes[row, 6].axis("off")
        figure.colorbar(mapping_handle, cax=axes[0, 5], label=f"{parameter} (ms)")
        figure.colorbar(residual_handle, cax=axes[0, 6], label="absolute difference (ms)")
        figure.suptitle(f"{pair.subject_id} {parameter}: PSF-matched native-plane decoder comparison")
        target = root / f"Figure2_{parameter}_native_bloch_mlp_psf.png"
        figure.savefig(target, dpi=320, bbox_inches="tight")
        figure.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)
        metadata[parameter] = {"png": str(target), "pdf": str(target.with_suffix(".pdf")), "mapping_range_ms": list(DISPLAY_RANGES_MS[parameter]), "residual_range_ms": list(RESIDUAL_RANGES_MS[parameter])}
    (root / "figure2_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    return metadata


def render_paired_all_slice_montages(pair: DecoderPair, paired_rows: list[dict[str, Any]], output: str | Path, colorbars: Any) -> dict[str, Any]:
    """Render all groups with the same paired mask shown to both decoders."""

    import matplotlib.pyplot as plt

    from .quality_control import DISPLAY_RANGES_MS, RESIDUAL_RANGES_MS

    root = Path(output)
    lookup = {(row["method"], row["parameter"], row["stack"], row["group_idx"]): row for row in paired_rows if row["method"] in {"Bloch", "FrozenMLP"}}
    written: dict[str, dict[str, dict[str, str]]] = {parameter: {} for parameter in PARAMETERS}
    for parameter in PARAMETERS:
        for stack in STACKS:
            trad, mlp = pair.trad.arrays[parameter][stack], pair.mlp.arrays[parameter][stack]
            count = trad["native"].shape[0]
            figure, axes = plt.subplots(count, 7, figsize=(15.4, max(3.0, 2.75 * count)), squeeze=False, gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 0.06, 0.06]})
            mapping_handle = residual_handle = None
            for group in range(count):
                native, trad_prediction, mlp_prediction = trad["native"][group], trad["prediction"][group], mlp["prediction"][group]
                mask = _paired_support(pair, parameter, stack)[group]
                if not mask.any():
                    raise ValueError(f"No paired common-support pixels for {parameter}/{stack}/group={group}.")
                cropped_native, cropped_mask = _crop(native, mask)
                cropped_trad, _ = _crop(trad_prediction, mask)
                cropped_mlp, _ = _crop(mlp_prediction, mask)
                cropped_trad_residual, _ = _crop(np.abs(trad_prediction - native), mask)
                cropped_mlp_residual, _ = _crop(np.abs(mlp_prediction - native), mask)
                panels = (
                    (cropped_native, "Native 2-D", colorbars.mapping[parameter], DISPLAY_RANGES_MS[parameter], "map"),
                    (cropped_trad, "Bloch PSF-matched", colorbars.mapping[parameter], DISPLAY_RANGES_MS[parameter], "map"),
                    (cropped_mlp, "FrozenMLP PSF-matched", colorbars.mapping[parameter], DISPLAY_RANGES_MS[parameter], "map"),
                    (cropped_trad_residual, "|Bloch - Native|", colorbars.residual, RESIDUAL_RANGES_MS[parameter], "residual"),
                    (cropped_mlp_residual, "|FrozenMLP - Native|", colorbars.residual, RESIDUAL_RANGES_MS[parameter], "residual"),
                )
                for column, (data, title, cmap, limits, kind) in enumerate(panels):
                    handle = axes[group, column].imshow(np.ma.masked_where(~cropped_mask, data), cmap=cmap, vmin=limits[0], vmax=limits[1], origin="lower", interpolation="nearest", aspect="equal")
                    if group == 0:
                        axes[group, column].set_title(title)
                    axes[group, column].axis("off")
                    if kind == "map":
                        mapping_handle = handle
                    else:
                        residual_handle = handle
                bloch = lookup[("Bloch", parameter, stack, group)]
                frozen = lookup[("FrozenMLP", parameter, stack, group)]
                annotation = f"group_idx={group}; paired N={int(bloch['N'])}; Bloch RMSE={float(bloch['RMSE_ms']):.1f}; MLP RMSE={float(frozen['RMSE_ms']):.1f}; ΔRMSE={float(frozen['RMSE_ms']) - float(bloch['RMSE_ms']):+.1f}; Bloch NCC={float(bloch['NCC']):.3f}; MLP NCC={float(frozen['NCC']):.3f}"
                axes[group, 0].text(0.02, 0.02, annotation, transform=axes[group, 0].transAxes, fontsize=5.2, color="black", va="bottom", ha="left", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5})
                axes[group, 5].axis("off")
                axes[group, 6].axis("off")
            figure.colorbar(mapping_handle, cax=axes[0, 5], label=f"{parameter} (ms)")
            figure.colorbar(residual_handle, cax=axes[0, 6], label="absolute difference (ms)")
            figure.suptitle(f"{pair.subject_id} {parameter} paired all-slice QC — {stack.upper()}")
            target = root / parameter / f"{parameter}_{stack.upper()}_all_slices_pair.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(target, dpi=220, bbox_inches="tight")
            figure.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
            plt.close(figure)
            written[parameter][stack] = {"png": str(target), "pdf": str(target.with_suffix(".pdf")), "groups": str(count)}
    metadata = {"support_definition": "paired_common_support", "outputs": written, "colorbars": colorbars.metadata}
    (root / "all_slice_pair_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def evaluate_myocardium_metrics(pair: DecoderPair, masks: dict[str, np.ndarray]) -> dict[str, Any]:
    """Evaluate SAX-only myocardial ROIs on method and paired supports."""

    result: dict[str, Any] = {
        "definition": "consistency/agreement with verified native 2-D dictionary maps; not independent ground-truth accuracy",
        "scope": "SAX only; myocardium ROIs are never applied to 2CH or 4CH",
        "method_specific": {},
        "paired_common_support": {},
    }
    for parameter in PARAMETERS:
        trad, mlp = pair.trad.arrays[parameter]["sax"], pair.mlp.arrays[parameter]["sax"]
        result["method_specific"][parameter] = {}
        result["paired_common_support"][parameter] = {}
        for name, roi in masks.items():
            if roi.shape != trad["native"].shape:
                raise ValueError(f"SAX myocardial ROI {name} shape {roi.shape} does not match native PSF comparison shape {trad['native'].shape}.")
            trad_metric = agreement_metrics(trad["native"], trad["prediction"], roi & trad["support"])
            mlp_metric = agreement_metrics(mlp["native"], mlp["prediction"], roi & mlp["support"])
            paired_support = roi & trad["support"] & mlp["support"]
            paired_trad = agreement_metrics(trad["native"], trad["prediction"], paired_support)
            paired_mlp = agreement_metrics(mlp["native"], mlp["prediction"], paired_support)
            result["method_specific"][parameter][name] = {"Bloch": trad_metric, "FrozenMLP": mlp_metric, "delta_mlp_minus_trad": _delta(trad_metric, mlp_metric)}
            result["paired_common_support"][parameter][name] = {"Bloch": paired_trad, "FrozenMLP": paired_mlp, "delta_mlp_minus_trad": _delta(paired_trad, paired_mlp)}
    return result


def render_paired_myocardium(pair: DecoderPair, selected: dict[str, int], masks: dict[str, np.ndarray], output: str | Path, colorbars: Any) -> dict[str, Any]:
    """Render representative SAX myocardium QC without changing source values."""

    import matplotlib.pyplot as plt

    from .quality_control import DISPLAY_RANGES_MS, RESIDUAL_RANGES_MS

    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    group = selected["sax"]
    core = masks["myocardium_core_1px"]
    metadata: dict[str, Any] = {"stack": "sax", "group_idx": group, "roi_overlay": "myocardium_core_1px", "support_definition": "paired_common_support"}
    for parameter in PARAMETERS:
        trad, mlp = pair.trad.arrays[parameter]["sax"], pair.mlp.arrays[parameter]["sax"]
        native, trad_prediction, mlp_prediction = trad["native"][group], trad["prediction"][group], mlp["prediction"][group]
        mask = _paired_support(pair, parameter, "sax")[group]
        panels = (
            (native, "Native 2-D", colorbars.mapping[parameter], DISPLAY_RANGES_MS[parameter]),
            (trad_prediction, "Bloch PSF-matched", colorbars.mapping[parameter], DISPLAY_RANGES_MS[parameter]),
            (mlp_prediction, "FrozenMLP PSF-matched", colorbars.mapping[parameter], DISPLAY_RANGES_MS[parameter]),
            (np.abs(trad_prediction - native), "|Bloch - Native|", colorbars.residual, RESIDUAL_RANGES_MS[parameter]),
            (np.abs(mlp_prediction - native), "|FrozenMLP - Native|", colorbars.residual, RESIDUAL_RANGES_MS[parameter]),
        )
        figure, axes = plt.subplots(1, 7, figsize=(15.4, 3.4), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 0.06, 0.06]})
        mapping_handle = residual_handle = None
        for column, (data, title, cmap, limits) in enumerate(panels):
            handle = axes[column].imshow(np.ma.masked_where(~mask, data), cmap=cmap, vmin=limits[0], vmax=limits[1], origin="lower", interpolation="nearest", aspect="equal")
            axes[column].contour(core[group], levels=[0.5], colors="white", linewidths=0.7)
            axes[column].set_title(title)
            axes[column].axis("off")
            if column < 3:
                mapping_handle = handle
            else:
                residual_handle = handle
        figure.colorbar(mapping_handle, cax=axes[5], label=f"{parameter} (ms)")
        figure.colorbar(residual_handle, cax=axes[6], label="absolute difference (ms)")
        figure.suptitle(f"{pair.subject_id} {parameter} SAX myocardium paired QC")
        target = root / f"{parameter}_SAX_myocardium_pair.png"
        figure.savefig(target, dpi=320, bbox_inches="tight")
        figure.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)
        metadata[parameter] = {"png": str(target), "pdf": str(target.with_suffix(".pdf"))}
    (root / "myocardium_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def render_paired_figure1(reference: Any, trad_run: str | Path, mlp_run: str | Path, output: str | Path, colorbars: Any, *, axis_name: str = "x", plane_index: int | None = None, reference_group: int = 0) -> dict[str, Any]:
    """Render same-world-plane 3-D continuity QC; no PSF data are used here."""

    import matplotlib.pyplot as plt

    from .quality_control import DISPLAY_RANGES_MS
    from .run_scmr_fig12 import _foreground_center, _plane_world, _require_nifti, _sample_plane_shaped

    nib, _ = _require_nifti()
    axis = {"x": 0, "y": 1}[axis_name]
    native = {key: nib.load(str(reference.figure1_stacks[key])) for key in ("t1", "t2")}
    if native["t1"].shape != native["t2"].shape or not np.allclose(native["t1"].affine, native["t2"].affine, atol=1e-4):
        raise ValueError("Figure 1 native T1/T2 stacks must share exact geometry.")
    if not 0 <= reference_group < native["t1"].shape[2]:
        raise IndexError(f"Figure 1 reference SAX group {reference_group} is unavailable.")
    native_t1 = native["t1"].get_fdata(dtype=np.float32)
    index = _foreground_center(native_t1, axis) if plane_index is None else plane_index
    if not 0 <= index < native_t1.shape[axis]:
        raise IndexError(f"Plane index {index} is outside native axis {axis_name}.")
    world, plane_shape, extent = _plane_world(native["t1"].affine, native["t1"].shape, axis, index)
    runs = {"Bloch": Path(trad_run), "FrozenMLP": Path(mlp_run)}
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 5, figsize=(15.5, 7), gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.055]})
    for row, (parameter, key) in enumerate((("T1", "t1"), ("T2", "t2"))):
        native_volume = native[key].get_fdata(dtype=np.float32)
        native_plane = _sample_plane_shaped(native_volume, np.linalg.inv(native[key].affine), world, plane_shape, order=0)
        mask = np.isfinite(native_plane) & (native_plane > 0)
        source = native_volume[:, :, reference_group].T
        rendered = [(source, "Native SAX + cut line", "source")]
        for method, run in runs.items():
            path = run / f"{parameter}_3D.nii.gz"
            if not path.is_file():
                raise FileNotFoundError(f"Figure 1 requires exported volume: {path}")
            image = nib.load(str(path))
            plane = _sample_plane_shaped(image.get_fdata(dtype=np.float32), np.linalg.inv(image.affine), world, plane_shape, order=1)
            rendered.append((plane, f"{method} 3-D through-plane (linear)", "plane"))
        for column, (data, title, kind) in enumerate(rendered):
            shown = np.ma.masked_where(~(np.isfinite(data) & (data > 0) if kind == "source" else mask), data)
            image = axes[row, column].imshow(shown, cmap=colorbars.mapping[parameter], vmin=DISPLAY_RANGES_MS[parameter][0], vmax=DISPLAY_RANGES_MS[parameter][1], origin="lower", interpolation="nearest", extent=None if kind == "source" else [0, extent[0], 0, extent[1]], aspect="equal")
            if kind == "source":
                (axes[row, column].axvline if axis == 0 else axes[row, column].axhline)(index, color="white", linewidth=0.8)
            axes[row, column].set_title(title if row == 0 else "")
            axes[row, column].set_ylabel(parameter)
            axes[row, column].axis("off")
        figure.colorbar(image, cax=axes[row, 4], label=f"{parameter} (ms)")
    figure.suptitle(f"{reference.manifest['subject_id']}: same-world-plane through-plane continuity QC")
    target = root / "Figure1_native_bloch_mlp_through_plane.png"
    figure.savefig(target, dpi=320, bbox_inches="tight")
    figure.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    metadata = {"purpose": "through-plane continuity QC; not PSF-matched native-plane quantitative comparison", "plane_axis": axis_name, "plane_index": index, "reference_sax_group": reference_group, "same_world_plane_check": "PASS", "native_interpolation": "nearest-neighbor", "reconstruction_interpolation": "linear", "png": str(target), "pdf": str(target.with_suffix(".pdf"))}
    (root / "figure1_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata
