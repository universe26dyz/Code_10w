"""Read-only Q001A/Q001B map and K=8 signal evaluator.

The command consumes final exports only.  It does not construct a model, train,
or migrate data.  Primary signal metrics are computed from true pooled pixels;
the historical D2 N-weighted macro is written separately as a secondary view.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from quality_experiments.D2_k8_signal_domain.core import STACKS, write_csv, write_json
from trad.evaluation.scmr.metrics import agreement_metrics
from trad.evaluation.scmr.reference_2d import load_verified_reference

from .contracts import sha256, verify_q001_input_bundle
from .evaluation import fixed_roi
from .metrics import LEGACY_MACRO, TRUE_POOLED, legacy_macro_rows, strict_common_support, true_pooled_rows
from .verify_full_fov_reference import verify_full_fov_reference
from .run_map_domain_psf import _prepared_hashes, validate_posthoc_manifest


SEED = 20260911


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _require_same_checkpoint(run: Path, route: dict[str, Any]) -> str:
    checkpoint = run / "model.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing final Q001 checkpoint: {checkpoint}")
    actual = sha256(checkpoint)
    expected = route.get("reconstruction_checkpoint_sha256")
    if actual != expected:
        raise ValueError("Q001 route manifest final checkpoint SHA256 mismatch.")
    for export, samples in (("mapping_central_no_psf", 1), ("signal_psf_K8", 8)):
        record = route.get("evaluation", {}).get(export, {})
        if record.get("checkpoint_sha256") != actual or record.get("psf_samples") != samples:
            raise ValueError(f"{export} provenance does not identify the same final checkpoint.")
    return actual


def _true_map_domain_psf_samples(root: Path) -> list[dict[str, Any]]:
    """Read only the D1-validated map-domain-PSF comparison artifacts."""
    samples: list[dict[str, Any]] = []
    for parameter in ("T1", "T2"):
        path = root / f"{parameter}_native_map_domain_psf_comparison.npz"
        with np.load(path, allow_pickle=False) as data:
            for stack in STACKS:
                prediction, reference, support = data[f"{stack}_predicted_ms"], data[f"{stack}_native_reference_ms"], data[f"{stack}_common_support"]
                for group in range(prediction.shape[0]):
                    samples.append({"endpoint": "true_map_domain_psf_K32_secondary", "export": "map_domain_psf_K32", "stack": stack, "group_idx": group, "parameter": parameter, "support_provenance": "map_domain_psf_support AND verified_native_reference_valid", "reference": reference[group], "prediction": prediction[group], "support": support[group]})
    return samples


def _validate_true_map_domain_psf(root: Path, run_root: Path, prepared_root: Path, reference_manifest: Path) -> dict[str, Any]:
    manifest_path, nested = root / "q001_map_domain_psf_manifest.json", root / "map_domain_psf_manifest.json"
    if not manifest_path.is_file() or not nested.is_file(): raise FileNotFoundError("Missing Q001 true map-domain PSF provenance manifest.")
    manifest = json.loads(manifest_path.read_text())
    validate_posthoc_manifest(manifest, run_root / "q001_route_manifest.json", run_root / "model.pt", run_root / "T1_3D.nii.gz", run_root / "T2_3D.nii.gz", run_root / "final_rigid_poses.json", _prepared_hashes(prepared_root), sha256(reference_manifest), sha256(nested))
    return manifest


def _stack_rows(run: Path, export: str, stack: str, kind: str) -> dict[str, np.ndarray]:
    prefix = "t1_t2_native_plane" if kind == "map" else "signal_reprojection"
    path = run / "evaluation" / export / f"{prefix}_{stack}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Q001 {kind} export: {path}")
    return _load(path)


def _weight_zero_maps(data: dict[str, np.ndarray], key: str, groups: int) -> tuple[np.ndarray, np.ndarray]:
    weights, group_idx = np.asarray(data["weight_idx"]), np.asarray(data["group_idx"])
    selected = [int(np.flatnonzero((group_idx == group) & (weights == 0))[0]) for group in range(groups)]
    if any(np.sum((group_idx == group) & (weights == 0)) != 1 for group in range(groups)):
        raise ValueError("Q001 map export requires exactly one weight-0 plane per native group.")
    return np.asarray(data[key])[selected], np.asarray(data["masks"], bool)[selected]


def _reference_archive(manifest: dict[str, Any], root: Path, stack: str) -> Path:
    """Use Q001 full-FOV v1 schema or the existing verified cropped-reference schema."""
    if isinstance(manifest.get("stacks"), dict):
        return root / str(manifest["stacks"][stack]["map"])
    if isinstance(manifest.get("maps"), dict):
        return root / str(manifest["maps"][stack])
    raise ValueError("Native reference manifest has neither Q001 stacks nor verified cropped maps schema.")


def fixed_baseline_map_pair(stack: str, full_prediction: np.ndarray, cropped_reference: np.ndarray, full_prediction_mask: np.ndarray, cropped_reference_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair a full-FOV prediction ROI with the complete already-cropped reference."""
    rs, cs = fixed_roi(stack)
    prediction, prediction_mask = full_prediction[rs, cs], np.asarray(full_prediction_mask, bool)[rs, cs]
    reference, reference_mask = cropped_reference, np.asarray(cropped_reference_mask, bool)
    if prediction.shape != reference.shape:
        raise ValueError(f"{stack} fixed-baseline map shape mismatch; resize/interpolation is forbidden.")
    return prediction, reference, prediction_mask & reference_mask


def _map_samples(run: Path, export: str, reference_root: Path, *, prediction_roi: bool, reference_is_already_cropped: bool, endpoint: str) -> list[dict[str, Any]]:
    manifest = json.loads((reference_root / "native_reference_manifest.json").read_text())
    samples: list[dict[str, Any]] = []
    for stack in STACKS:
        reference = _load(_reference_archive(manifest, reference_root, stack))
        prediction_data = _stack_rows(run, export, stack, "map")
        prediction_rs, prediction_cs = fixed_roi(stack) if prediction_roi else (slice(None), slice(None))
        reference_rs, reference_cs = (slice(None), slice(None)) if reference_is_already_cropped else (prediction_rs, prediction_cs)
        for parameter, key in (("T1", "t1_ms"), ("T2", "t2_ms")):
            predicted, mask = _weight_zero_maps(prediction_data, key, reference[key].shape[0])
            for group in range(reference[key].shape[0]):
                if reference_is_already_cropped:
                    pred, ref, support = fixed_baseline_map_pair(stack, predicted[group], reference[key][group], mask[group], reference["valid_mask"][group]) if prediction_roi else (predicted[group], reference[key][group], mask[group] & reference["valid_mask"][group])
                    prediction_mask, reference_mask = mask[group], np.asarray(reference["valid_mask"][group], bool)
                else:
                    ref = reference[key][group, reference_rs, reference_cs]
                    pred = predicted[group, prediction_rs, prediction_cs]
                    reference_mask = np.asarray(reference["valid_mask"][group, reference_rs, reference_cs], bool)
                    prediction_mask = mask[group, prediction_rs, prediction_cs]
                    support = reference_mask & prediction_mask
                if ref.shape != pred.shape:
                    raise ValueError(f"{stack} reference/prediction shape mismatch: {ref.shape} != {pred.shape}; resize/interpolation is forbidden.")
                support = support & np.isfinite(ref) & np.isfinite(pred)
                samples.append({"endpoint": endpoint, "export": export, "stack": stack, "group_idx": group, "parameter": parameter, "support_provenance": "native_reference_valid AND reconstruction_valid AND finite(reference,prediction)", "reference": ref, "prediction": pred, "support": support})
    return samples


def _paired_map_samples(candidate_run: Path, baseline_run: Path, reference_root: Path) -> list[dict[str, Any]]:
    """Same cropped native reference and exact common support for Q001B-vs-B6 maps."""
    manifest = json.loads((reference_root / "native_reference_manifest.json").read_text())
    samples: list[dict[str, Any]] = []
    for stack in STACKS:
        reference = _load(_reference_archive(manifest, reference_root, stack))
        candidate, baseline = _stack_rows(candidate_run, "mapping_central_no_psf", stack, "map"), _load(baseline_run / f"t1_t2_native_plane_{stack}.npz")
        for parameter, key in (("T1", "t1_ms"), ("T2", "t2_ms")):
            candidate_map, candidate_mask = _weight_zero_maps(candidate, key, reference[key].shape[0])
            baseline_map, baseline_mask = _weight_zero_maps(baseline, key, reference[key].shape[0])
            for group in range(reference[key].shape[0]):
                support = np.asarray(reference["valid_mask"][group], bool) & candidate_mask[group] & baseline_mask[group] & np.isfinite(reference[key][group]) & np.isfinite(candidate_map[group]) & np.isfinite(baseline_map[group])
                for method, prediction in (("B6_FrozenMLP", baseline_map[group]), ("Q001B", candidate_map[group])):
                    samples.append({"method": method, "endpoint": "strict_b6_q001b_cropped_map_common_support", "export": "mapping_central_no_psf", "stack": stack, "group_idx": group, "parameter": parameter, "support_provenance": "cropped_native_reference_valid AND B6_valid AND Q001B_valid AND finite(reference,B6,Q001B)", "reference": reference[key][group], "prediction": prediction, "support": support})
    return samples


def _cropped_reference_report(root: Path, preprocessed_root: Path) -> dict[str, Any]:
    """Delegate cropped-reference validation to the established SCMR verifier."""
    reference = load_verified_reference(root, "CYJ", preprocessed_root)
    return {"root": str(reference.root), "manifest_sha256": sha256(reference.manifest_path), "verified_hashes": reference.file_hashes}


def _artifact_hashes(root: Path, relative_paths: dict[str, Path]) -> dict[str, str]:
    result = {}
    for name, path in relative_paths.items():
        if not path.is_file(): raise FileNotFoundError(f"Missing required provenance artifact: {path}")
        result[name] = sha256(path)
    return result


def _signal_samples(run: Path, export: str, *, roi: bool, label: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for stack in STACKS:
        data = _stack_rows(run, export, stack, "signal")
        rs, cs = fixed_roi(stack) if roi else (slice(None), slice(None))
        for index in range(data["observed"].shape[0]):
            observed, predicted, mask = data["observed"][index, rs, cs], data["predicted"][index, rs, cs], np.asarray(data["masks"][index, rs, cs], bool)
            samples.append({"endpoint": label, "stack": stack, "group_idx": int(data["group_idx"][index]), "weight_idx": int(data["weight_idx"][index]), "support_provenance": "method_valid AND finite(observed,prediction)", "reference": observed, "prediction": predicted, "support": mask & np.isfinite(observed) & np.isfinite(predicted)})
    return samples


def _signal_matched_samples(q001_run: Path, b6_root: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for stack in STACKS:
        candidate = _stack_rows(q001_run, "signal_psf_K8", stack, "signal")
        baseline_path = b6_root / f"signal_reprojection_{stack}_K8.npz"
        if not baseline_path.is_file():
            # Formal D2 exports use this explicit artifact subdirectory.
            baseline_path = b6_root / "artifacts" / "FrozenMLP" / f"signal_reprojection_{stack}_K8.npz"
        baseline = _load(baseline_path)
        for key in ("observed", "group_idx", "weight_idx", "timing9_ms"):
            if not np.array_equal(candidate[key], baseline[key], equal_nan=True):
                raise ValueError(f"Q001B/B6 signal provenance mismatch for {stack} {key}.")
        for index in range(candidate["observed"].shape[0]):
            support = strict_common_support(candidate["observed"][index], baseline["predicted"][index], candidate["predicted"][index], baseline["masks"][index], candidate["masks"][index])
            for method, prediction in (("B6_FrozenMLP", baseline["predicted"][index]), ("Q001B", candidate["predicted"][index])):
                samples.append({"endpoint": "strict_b6_q001b_common_support", "method": method, "stack": stack, "group_idx": int(candidate["group_idx"][index]), "weight_idx": int(candidate["weight_idx"][index]), "support_provenance": "B6_valid AND Q001B_valid AND finite(observed,B6_prediction,Q001B_prediction)", "reference": candidate["observed"][index], "prediction": prediction, "support": support})
    return samples


def _write_levels(metrics: Path, stem: str, samples: list[dict[str, Any]], *, domain: str, include_method: bool = False) -> None:
    base = ("endpoint",) + (("method",) if include_method else ()) + ("stack", "weight_idx", "support_provenance")
    for label, fn in (("true_pooled", true_pooled_rows), ("legacy_macro_n_weighted", legacy_macro_rows)):
        for suffix, keys in (("per_stack_weight", base), ("per_stack", tuple(key for key in base if key != "weight_idx")), ("per_weight", tuple(key for key in base if key != "stack")), ("global", tuple(key for key in base if key not in {"stack", "weight_idx"}))):
            rows = fn(samples, keys, domain=domain)
            write_csv(metrics / f"{stem}_{label}_{suffix}.csv", rows)


def _write_map_metrics(metrics: Path, stem: str, samples: list[dict[str, Any]]) -> None:
    # Maps do not have a signal weight dimension.  Per-group maps retain exact same-index pairing.
    rows = []
    for item in samples:
        metric = agreement_metrics(item["reference"], item["prediction"], item["support"], min_pixels=16)
        rows.append({key: item[key] for key in (("method",) if "method" in item else ()) + ("endpoint", "export", "stack", "group_idx", "parameter", "support_provenance")} | {"metric_semantics": TRUE_POOLED, **metric})
    write_csv(metrics / f"{stem}_per_group_true_pooled.csv", rows)
    method_key = ("method",) if "method" in samples[0] else ()
    for label, fn in (("true_pooled", true_pooled_rows), ("legacy_macro_n_weighted", legacy_macro_rows)):
        for suffix, keys in (("per_stack", method_key + ("endpoint", "export", "stack", "parameter", "support_provenance")), ("global", method_key + ("endpoint", "export", "parameter", "support_provenance"))):
            write_csv(metrics / f"{stem}_{label}_{suffix}.csv", fn(samples, keys, domain="map"))


def run(args: argparse.Namespace) -> Path:
    run_root, output = Path(args.run_root).resolve(), Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty Q001 evaluation root: {output}")
    output.mkdir(parents=True); metrics, figures, logs = output / "metrics", output / "figures", output / "logs"
    for path in (metrics, figures, logs): path.mkdir()
    route = json.loads((run_root / "q001_route_manifest.json").read_text())
    mode = "Q001A" if str(route["experiment_id"]).startswith("Q001A_") else "Q001B" if str(route["experiment_id"]).startswith("Q001B_") else None
    if mode != args.mode: raise ValueError("Requested Q001 evaluator mode does not match route manifest.")
    checkpoint_hash = _require_same_checkpoint(run_root, route)
    verify_q001_input_bundle(args.full_input_root)
    map_samples: list[dict[str, Any]]
    if mode == "Q001A":
        if not args.full_reference_root: raise ValueError("Q001A requires --full-reference-root.")
        if not args.cropped_reference_root or not args.cropped_preprocessed_root: raise ValueError("Q001A fixed-baseline endpoint requires a verified cropped reference and preprocessed root.")
        full_report = verify_full_fov_reference(args.full_reference_root, args.full_input_root)
        cropped_report = _cropped_reference_report(Path(args.cropped_reference_root), Path(args.cropped_preprocessed_root))
        map_samples = _map_samples(run_root, "mapping_central_no_psf", Path(args.full_reference_root), prediction_roi=False, reference_is_already_cropped=False, endpoint="route_consistent_full_fov")
        _write_map_metrics(metrics, "q001a_maps", map_samples)
        cropped = _map_samples(run_root, "mapping_central_no_psf", Path(args.cropped_reference_root), prediction_roi=True, reference_is_already_cropped=True, endpoint="fixed_baseline_reference_consistency")
        _write_map_metrics(metrics, "q001a_fixed_baseline_consistency", cropped)
        signal = _signal_samples(run_root, "signal_psf_K8", roi=False, label="whole_full_fov_own_support") + _signal_samples(run_root, "signal_psf_K8", roi=True, label="fixed_baseline_crop_own_support")
        _write_levels(metrics, "q001a_signal", signal, domain="signal")
    else:
        if not args.cropped_reference_root or not args.cropped_preprocessed_root or not args.b6_d2_root or not args.b6_map_root: raise ValueError("Q001B requires a verified cropped reference, B6 map exports, and formal D2 FrozenMLP artifacts.")
        full_report = None
        cropped_report = _cropped_reference_report(Path(args.cropped_reference_root), Path(args.cropped_preprocessed_root))
        map_samples = _map_samples(run_root, "mapping_central_no_psf", Path(args.cropped_reference_root), prediction_roi=False, reference_is_already_cropped=True, endpoint="cropped_native_reference_primary")
        _write_map_metrics(metrics, "q001b_maps", map_samples)
        paired_maps = _paired_map_samples(run_root, Path(args.b6_map_root), Path(args.cropped_reference_root))
        _write_map_metrics(metrics, "q001b_vs_b6_strict_common_support_maps", paired_maps)
        if not args.map_domain_psf_root or not args.cropped_prepared_root: raise ValueError("Q001B formal map secondary requires verified map-domain output and cropped prepared geometry.")
        map_psf_manifest = _validate_true_map_domain_psf(Path(args.map_domain_psf_root), run_root, Path(args.cropped_prepared_root), Path(args.cropped_reference_root) / "native_reference_manifest.json")
        _write_map_metrics(metrics, "q001b_true_map_domain_psf_K32_secondary", _true_map_domain_psf_samples(Path(args.map_domain_psf_root)))
        strict = _signal_matched_samples(run_root, Path(args.b6_d2_root))
        _write_levels(metrics, "q001b_vs_b6_strict_common_support", strict, domain="signal", include_method=True)
    b6_maps = _artifact_hashes(Path(args.b6_map_root), {stack: Path(args.b6_map_root) / f"t1_t2_native_plane_{stack}.npz" for stack in STACKS}) if args.b6_map_root else {}
    b6_k8 = _artifact_hashes(Path(args.b6_d2_root), {stack: (Path(args.b6_d2_root) / "artifacts" / "FrozenMLP" / f"signal_reprojection_{stack}_K8.npz") for stack in STACKS}) if args.b6_d2_root else {}
    manifest = {"schema": "q001_controlled_evaluation/v1", "experiment_id": route["experiment_id"], "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(), "final_reconstruction_checkpoint": str(run_root / "model.pt"), "final_checkpoint_sha256": checkpoint_hash, "q001_input_manifest_sha256": sha256(Path(args.full_input_root) / "q001_input_manifest.json"), "full_reference": full_report, "cropped_reference": cropped_report, "map_domain_psf": map_psf_manifest if mode == "Q001B" else None, "b6_model_sha256": route.get("b6_model_sha256"), "frozenmlp_source_checkpoint_sha256": route.get("frozen_signal_simulator_sha256"), "b6_central_map_artifact_sha256": b6_maps, "b6_d2_k8_artifact_sha256": b6_k8, "k8_seed": SEED, "k32_seed": SEED, "roi_zero_based": {stack: [fixed_roi(stack)[0].start, fixed_roi(stack)[0].stop, fixed_roi(stack)[1].start, fixed_roi(stack)[1].stop] for stack in STACKS}, "metric_semantics": {"primary": TRUE_POOLED, "secondary": LEGACY_MACRO}, "g0_reuse": "Q001 evaluator retains official same-index pairing; corrected G0 correspondence/pose audit remains read-only diagnostic reuse."}
    write_json(output / "evaluation_manifest.json", manifest)
    (figures / "README.md").write_text("Figures are generated only from formal exports; no reconstruction is run by this evaluator.\n", encoding="utf-8")
    (logs / "evaluation.log").write_text("Q001 read-only evaluation completed from final exported artifacts.\n", encoding="utf-8")
    (output / "RESULT_SUMMARY.md").write_text(f"# {mode} evaluation\n\nPRIMARY: true_pooled pixel metrics. SECONDARY: legacy_macro_n_weighted metrics. No reconstruction was run.\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("Q001A", "Q001B"), required=True); parser.add_argument("--run-root", required=True); parser.add_argument("--full-input-root", required=True); parser.add_argument("--full-reference-root"); parser.add_argument("--cropped-reference-root"); parser.add_argument("--cropped-preprocessed-root"); parser.add_argument("--cropped-prepared-root"); parser.add_argument("--b6-map-root"); parser.add_argument("--b6-d2-root"); parser.add_argument("--map-domain-psf-root"); parser.add_argument("--output", required=True)
    print(run(parser.parse_args()))


if __name__ == "__main__": main()
