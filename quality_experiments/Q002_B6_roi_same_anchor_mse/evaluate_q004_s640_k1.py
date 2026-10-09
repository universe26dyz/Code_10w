"""Read-only Q004 K1-versus-Q002-S640 evaluation with derived central-K1 comparators."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from mlp.modules.module_07_objective_training.mlp_trainer import build_mlp_training_model
from reconstruction_core.orchestration import load_checkpoint
from trad.modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset
from trad.modules.module_08_inference_export.reprojection import export_native_plane_reprojections

from quality_experiments.D2_k8_signal_domain.core import write_csv, write_json
from quality_experiments.Q001_recon.metrics import true_pooled_rows
from trad.evaluation.scmr.metrics import signal_agreement_metrics
from trad.evaluation.scmr.reference_2d import load_verified_reference

from .contracts import cropped_observation_paths
from .evaluate_q002 import _canonical_weight_zero_maps, _load, _run_three_way, _sha, _signal_identity_index, _validated_route_checkpoint
from .metrics import fingerprint_cosine_summary, strict_fingerprint_three_way_common_support, strict_q004_three_way_common_support
from .visualization import deterministic_representatives, render_fingerprint_curves, render_map_montages, visualization_display_contract, write_visualization_manifest


Q004_ID = "Q004S640_K1_B6_roi_same_anchor_mse"


def _load_model(checkpoint_path: Path, dataset: QuantPointDataset, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["resolved_config"]
    protocol = config["decoder"].get("protocol_yaml", "trad/configs/protocol_hhz_v1.yaml")
    model, space, _, _ = build_mlp_training_model(dataset, config, protocol, device)
    load_checkpoint(checkpoint_path, model, device)
    return model, space


def _derive_central_k1(args: argparse.Namespace, output: Path, q004_checkpoint: Path, q002_checkpoint: Path) -> dict[str, Path]:
    inputs = cropped_observation_paths(args.cropped_prepared_root)
    device = torch.device(args.device)
    dataset = QuantPointDataset(inputs, device=args.device)
    roots = {"B6": Path(args.b6_model), "Q002_S640": q002_checkpoint, "Q004_S640_K1": q004_checkpoint}
    results: dict[str, Path] = {}
    for label, checkpoint in roots.items():
        model, space = _load_model(checkpoint, dataset, device)
        root = output / "derived_k1" / label
        export_native_plane_reprojections(model, space, inputs, root, output_psf={"enabled": True, "n_samples": 1}, export_parameter_maps=False)
        results[label] = root
    return results


def _fingerprint_record(stack: str, group: int, row_col: tuple[int, int], observed: np.ndarray, predictions: dict[str, np.ndarray]) -> dict[str, object]:
    """Describe one deterministic ten-weight fingerprint without affecting metrics."""

    metrics: dict[str, dict[str, float]] = {}
    for label, prediction in predictions.items():
        residual = prediction - observed
        denominator = max(float(np.linalg.norm(prediction)) * float(np.linalg.norm(observed)), 1.0e-8)
        metrics[label] = {"rmse": float(np.sqrt(np.mean(residual**2))), "cosine": float(np.dot(prediction, observed) / denominator)}
    return {"identity": (stack, int(group), *row_col), "stack": stack, "group_idx": int(group), "row_col": row_col, "vectors": {"Observed": observed, **predictions}, "fingerprint_metrics": metrics}


def _evaluate_derived_k1(roots: dict[str, Path], output: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    cosine_rows: list[dict[str, object]] = []
    fingerprint_candidates: list[dict[str, object]] = []
    for stack in ("sax", "2ch", "4ch"):
        archives = {label: _load(root / f"signal_reprojection_{stack}.npz") for label, root in roots.items()}
        indexes = {label: _signal_identity_index(archive, f"{label} derived K1") for label, archive in archives.items()}
        if len({frozenset(index) for index in indexes.values()}) != 1:
            raise ValueError("Derived K1 comparator identity sets differ.")
        for pair, b6_row in indexes["B6"].items():
            q2_row, q4_row = indexes["Q002_S640"][pair], indexes["Q004_S640_K1"][pair]
            observed = archives["B6"]["observed"][b6_row]
            if not np.array_equal(observed, archives["Q002_S640"]["observed"][q2_row], equal_nan=True) or not np.array_equal(observed, archives["Q004_S640_K1"]["observed"][q4_row], equal_nan=True):
                raise ValueError("Derived K1 observed provenance differs.")
            support = strict_q004_three_way_common_support(observed, archives["B6"]["predicted"][b6_row], archives["Q002_S640"]["predicted"][q2_row], archives["Q004_S640_K1"]["predicted"][q4_row], np.ones_like(archives["B6"]["masks"][b6_row], bool), archives["B6"]["masks"][b6_row], archives["Q002_S640"]["masks"][q2_row], archives["Q004_S640_K1"]["masks"][q4_row])
            for label, row in (("FrozenMLP_B6", b6_row), ("Q002_S640", q2_row), ("Q004_S640_K1", q4_row)):
                source = "B6" if label == "FrozenMLP_B6" else label
                prediction = archives[source]["predicted"][row]
                rows.append({"method": label, "stack": stack, "group_idx": pair[0], "weight_idx": pair[1], **signal_agreement_metrics(observed, prediction, support, min_pixels=16)})
                samples.append({"method": label, "stack": stack, "group_idx": pair[0], "weight_idx": pair[1], "support_provenance": "observed AND B6_K1_valid AND Q002_K1_valid AND Q004_K1_valid AND finite", "reference": observed, "prediction": prediction, "support": support})
        for group in sorted(np.unique(archives["B6"]["group_idx"]).tolist()):
            pairs = [(int(group), weight) for weight in range(10)]
            b6_rows, q2_rows, q4_rows = ([indexes[name][pair] for pair in pairs] for name in ("B6", "Q002_S640", "Q004_S640_K1"))
            support = strict_fingerprint_three_way_common_support(archives["B6"]["predicted"][b6_rows], archives["Q002_S640"]["predicted"][q2_rows], archives["Q004_S640_K1"]["predicted"][q4_rows], archives["B6"]["masks"][b6_rows], archives["Q002_S640"]["masks"][q2_rows], archives["Q004_S640_K1"]["masks"][q4_rows]) & np.isfinite(archives["B6"]["observed"][b6_rows]).all(axis=0)
            for label, source, indexes_for_method in (("FrozenMLP_B6", "B6", b6_rows), ("Q002_S640", "Q002_S640", q2_rows), ("Q004_S640_K1", "Q004_S640_K1", q4_rows)):
                cosine_rows.append({"method": label, "stack": stack, "group_idx": int(group), "role": "observed_fingerprint_fidelity_central_k1_three_way_common_support", **fingerprint_cosine_summary(archives[source]["predicted"][indexes_for_method], archives["B6"]["observed"][b6_rows], support)})
            valid = np.argwhere(support)
            if valid.size:
                row_col = tuple(int(value) for value in valid[0])
                fingerprint_candidates.append(_fingerprint_record(stack, int(group), row_col, archives["B6"]["observed"][b6_rows, row_col[0], row_col[1]], {"FrozenMLP_B6": archives["B6"]["predicted"][b6_rows, row_col[0], row_col[1]], "Q002_S640": archives["Q002_S640"]["predicted"][q2_rows, row_col[0], row_col[1]], "Q004_S640_K1": archives["Q004_S640_K1"]["predicted"][q4_rows, row_col[0], row_col[1]]}))
    metrics = output / "metrics"
    write_csv(metrics / "signal_central_K1_three_way_per_group_common_support.csv", rows)
    write_csv(metrics / "signal_central_K1_three_way_true_pooled.csv", true_pooled_rows(samples, ("method", "stack", "weight_idx", "support_provenance"), domain="signal") + true_pooled_rows(samples, ("method", "stack", "group_idx", "support_provenance"), domain="signal") + true_pooled_rows(samples, ("method", "stack", "support_provenance"), domain="signal") + true_pooled_rows(samples, ("method", "support_provenance"), domain="signal"))
    write_csv(metrics / "fingerprint_cosine_central_K1_three_way_read_only.csv", cosine_rows)
    return fingerprint_candidates


def _render_q004_visualizations(args: argparse.Namespace, output: Path, route: dict[str, object], q002_checkpoint: Path, q004_checkpoint: Path, central_k1_candidates: list[dict[str, object]]) -> None:
    """Render descriptive fixed-range figures from the same strict supports as metrics."""

    map_records: list[dict[str, object]] = []
    fingerprint_candidates: list[dict[str, object]] = []
    q002_root, q004_root = Path(args.q002_s640_run_root), Path(args.run_root)
    verified_reference = load_verified_reference(args.reference_root, "CYJ", args.preprocessed_root)
    for stack in ("sax", "2ch", "4ch"):
        b6_maps = _load(Path(args.b6_map_root) / f"t1_t2_native_plane_{stack}.npz")
        q002_maps = _load(q002_root / "evaluation/mapping_central_no_psf" / f"t1_t2_native_plane_{stack}.npz")
        q004_maps = _load(q004_root / "evaluation/mapping_central_no_psf" / f"t1_t2_native_plane_{stack}.npz")
        native = verified_reference.stacks[stack]
        for key, parameter in (("t1_ms", "T1"), ("t2_ms", "T2")):
            b6_values, b6_masks = _canonical_weight_zero_maps(b6_maps, key, native.t1_ms.shape[0])
            q002_values, q002_masks = _canonical_weight_zero_maps(q002_maps, key, native.t1_ms.shape[0])
            q004_values, q004_masks = _canonical_weight_zero_maps(q004_maps, key, native.t1_ms.shape[0])
            reference = native.t1_ms if key == "t1_ms" else native.t2_ms
            for group in range(reference.shape[0]):
                support = strict_q004_three_way_common_support(reference[group], b6_values[group], q002_values[group], q004_values[group], native.valid_mask[group], b6_masks[group], q002_masks[group], q004_masks[group])
                map_records.append({"parameter": parameter, "stack": stack, "group_idx": group, "reference": reference[group], "predictions": {"FrozenMLP_B6": b6_values[group], "Q002_S640": q002_values[group], "Q004_S640_K1": q004_values[group]}, "support": support})

        b6_path = Path(args.b6_d2_root) / "artifacts/FrozenMLP" / f"signal_reprojection_{stack}_K8.npz"
        b6 = _load(b6_path if b6_path.is_file() else Path(args.b6_d2_root) / f"signal_reprojection_{stack}_K8.npz")
        q002 = _load(q002_root / "evaluation/signal_psf_K8" / f"signal_reprojection_{stack}.npz")
        q004 = _load(q004_root / "evaluation/signal_psf_K8" / f"signal_reprojection_{stack}.npz")
        indexes = {"B6": _signal_identity_index(b6, "B6 K8"), "Q002_S640": _signal_identity_index(q002, "Q002-S640 K8"), "Q004_S640_K1": _signal_identity_index(q004, "Q004-S640-K1 K8")}
        if len({frozenset(index) for index in indexes.values()}) != 1:
            raise ValueError("Q004 visualization K8 identity sets differ.")
        for group in sorted(np.unique(b6["group_idx"]).tolist()):
            pairs = [(int(group), weight) for weight in range(10)]
            b6_rows, q002_rows, q004_rows = ([indexes[name][pair] for pair in pairs] for name in ("B6", "Q002_S640", "Q004_S640_K1"))
            support = strict_fingerprint_three_way_common_support(b6["predicted"][b6_rows], q002["predicted"][q002_rows], q004["predicted"][q004_rows], b6["masks"][b6_rows], q002["masks"][q002_rows], q004["masks"][q004_rows]) & np.isfinite(b6["observed"][b6_rows]).all(axis=0)
            valid = np.argwhere(support)
            if valid.size:
                row_col = tuple(int(value) for value in valid[0])
                fingerprint_candidates.append(_fingerprint_record(stack, int(group), row_col, b6["observed"][b6_rows, row_col[0], row_col[1]], {"FrozenMLP_B6": b6["predicted"][b6_rows, row_col[0], row_col[1]], "Q002_S640": q002["predicted"][q002_rows, row_col[0], row_col[1]], "Q004_S640_K1": q004["predicted"][q004_rows, row_col[0], row_col[1]]}))

    visualization = output / "visualization"
    visualization.mkdir()
    render_map_montages(map_records, visualization / "all_slices", methods=("FrozenMLP_B6", "Q002_S640", "Q004_S640_K1"))
    representative_ids = set(deterministic_representatives([(record["stack"], int(record["group_idx"]), 0, 0) for record in map_records], limit=6))
    render_map_montages([record for record in map_records if (record["stack"], int(record["group_idx"]), 0, 0) in representative_ids], visualization / "representative_montages", methods=("FrozenMLP_B6", "Q002_S640", "Q004_S640_K1"))
    selected_ids = set(deterministic_representatives([candidate["identity"] for candidate in fingerprint_candidates], limit=9))
    selected = [candidate for candidate in fingerprint_candidates if candidate["identity"] in selected_ids]
    render_fingerprint_curves(selected, visualization / "fingerprint_curves_K8")
    central_by_identity = {candidate["identity"]: candidate for candidate in central_k1_candidates}
    central_selected = [central_by_identity[identity] for identity in selected_ids if identity in central_by_identity]
    if not central_selected:
        central_selected = [candidate for candidate in central_k1_candidates if candidate["identity"] in set(deterministic_representatives([item["identity"] for item in central_k1_candidates], limit=9))]
    render_fingerprint_curves(central_selected, visualization / "fingerprint_curves_central_K1")
    write_visualization_manifest(visualization, {"reconstruction_git_sha": route["reconstruction_git_sha"], "evaluation_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "source_checkpoint_paths": {"q002_s640": str(q002_checkpoint), "q004_s640_k1": str(q004_checkpoint)}, "source_checkpoint_sha256": {"q002_s640": _sha(q002_checkpoint), "q004_s640_k1": _sha(q004_checkpoint)}, "b6_comparator_identity": str(args.b6_map_root), "selected_groups_anchors": [{"stack": item["stack"], "group_idx": item["group_idx"], "row_col": item["row_col"], "k8_fingerprint_metrics": item["fingerprint_metrics"], "central_k1_fingerprint_metrics": central_by_identity[item["identity"]]["fingerprint_metrics"] if item["identity"] in central_by_identity else None} for item in selected], "representative_group_rule": "lexicographic_stack_group_identity", "display": visualization_display_contract(), "common_support": "reference/observed AND B6_valid AND Q002_S640_valid AND Q004_K1_valid AND finite"})


def run_q004_s640_k1_evaluation(args: argparse.Namespace) -> Path:
    run_root, output, q002_root = Path(args.run_root), Path(args.output), Path(args.q002_s640_run_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty Q004 evaluation root: {output}")
    route, q004_checkpoint = _validated_route_checkpoint(run_root, "Q004-S640-K1")
    q002_route, q002_checkpoint = _validated_route_checkpoint(q002_root, "Q002-S640")
    if route.get("experiment_id") != Q004_ID or q002_route.get("experiment_id") != "Q002S640_B6_roi_same_anchor_mse":
        raise ValueError("Q004 evaluation route manifests do not match the requested immutable identities.")
    _run_three_way(args, run_root, output, route, q004_checkpoint, comparator_root=q002_root, comparator_experiment_id="Q002S640_B6_roi_same_anchor_mse", candidate_experiment_id=Q004_ID, comparator_label="Q002_S640", candidate_label="Q004_S640_K1", primary_comparison="Q004_S640_K1_vs_Q002_S640_K8_training", secondary_comparison="Q004_S640_K1_vs_FrozenMLP_B6", fingerprint_cosine_training_objective=False)
    roots = _derive_central_k1(args, output, q004_checkpoint, q002_checkpoint)
    central_k1_candidates = _evaluate_derived_k1(roots, output)
    _render_q004_visualizations(args, output, route, q002_checkpoint, q004_checkpoint, central_k1_candidates)
    manifest_path = output / "evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({"evaluation_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "primary_map_support": "reference_valid AND B6_valid AND Q002_S640_valid AND Q004_K1_valid AND finite", "k8_signal_support": "observed AND B6_K8_valid AND Q002_S640_K8_valid AND Q004_K1_model_reprojected_at_K8_valid AND finite", "central_k1_signal_support": "observed AND B6_K1_valid AND Q002_K1_valid AND Q004_K1_valid AND finite", "derived_k1_artifacts": {key: str(value) for key, value in roots.items()}, "derived_k1_artifacts_historical_roots_untouched": True, "comparator_checkpoint_sha256": {"q002_s640": _sha(q002_checkpoint), "q004": _sha(q004_checkpoint)}})
    write_json(manifest_path, manifest)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "q002-s640-run-root", "b6-model", "b6-map-root", "b6-d2-root", "cropped-prepared-root", "reference-root", "preprocessed-root", "output"): parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    print(run_q004_s640_k1_evaluation(build_parser().parse_args()))


if __name__ == "__main__":
    main()
