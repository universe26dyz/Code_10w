"""Strict four-way B6/Q002-S640/Q003-6000/Q003-10000 evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from quality_experiments.D2_k8_signal_domain.core import write_csv, write_json
from quality_experiments.Q001_recon.metrics import true_pooled_rows
from trad.evaluation.scmr.metrics import agreement_metrics, signal_agreement_metrics
from trad.evaluation.scmr.reference_2d import load_verified_reference

from .evaluate_q002 import _canonical_weight_zero_maps, _load, _sha, _signal_identity_index, _validated_route_checkpoint
from .metrics import fingerprint_cosine_summary, strict_fingerprint_four_way_common_support, strict_q003_four_way_common_support
from .visualization import deterministic_representatives, render_fingerprint_curves, render_map_montages, visualization_display_contract, write_visualization_manifest


STACKS = ("sax", "2ch", "4ch")
Q003_10K_ID = "Q003S640_B6_roi_same_anchor_mse_plus_cosine_10k"


def _validated_q003_checkpoints(root: Path) -> tuple[dict[str, object], Path, Path]:
    route = json.loads((root / "q002_route_manifest.json").read_text())
    final, primary = root / "model.pt", root / "checkpoints/model_iter_6000.pt"
    primary_manifest = json.loads((root / "checkpoints/model_iter_6000_manifest.json").read_text())
    if route.get("experiment_id") != Q003_10K_ID or not final.is_file() or not primary.is_file():
        raise ValueError("Q003-10k route or required checkpoints are missing.")
    if _sha(final) != route.get("final_checkpoint_10000_sha256") or _sha(primary) != route.get("primary_checkpoint_6000_sha256") or _sha(primary) != primary_manifest.get("checkpoint_sha256"):
        raise ValueError("Q003-10k checkpoint hash does not match its immutable manifest.")
    return route, primary, final


def _signal_archive(root: Path, checkpoint: int, stack: str) -> dict[str, np.ndarray]:
    return _load(root / f"evaluation/checkpoint_{checkpoint}/signal_psf_K8" / f"signal_reprojection_{stack}.npz")


def run_q003_s640_10k_four_way(args: argparse.Namespace) -> Path:
    output, run_root, q002_root = Path(args.output), Path(args.run_root), Path(args.q002_s640_run_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty Q003-10k evaluation root: {output}")
    route, primary_checkpoint, final_checkpoint = _validated_q003_checkpoints(run_root)
    q002_route, q002_checkpoint = _validated_route_checkpoint(q002_root, "Q002-S640")
    if q002_route.get("experiment_id") != "Q002S640_B6_roi_same_anchor_mse":
        raise ValueError("Q003-10k evaluator requires the immutable Q002-S640 parent route.")
    output.mkdir(parents=True); metrics = output / "metrics"; metrics.mkdir()
    map_samples: list[dict[str, object]] = []
    map_records: list[dict[str, object]] = []
    for stack in STACKS:
        b6 = _load(Path(args.b6_map_root) / f"t1_t2_native_plane_{stack}.npz")
        q002 = _load(q002_root / "evaluation/mapping_central_no_psf" / f"t1_t2_native_plane_{stack}.npz")
        q003_6000 = _load(run_root / "evaluation/checkpoint_6000/mapping_central_no_psf" / f"t1_t2_native_plane_{stack}.npz")
        q003_10000 = _load(run_root / "evaluation/checkpoint_10000/mapping_central_no_psf" / f"t1_t2_native_plane_{stack}.npz")
        native = load_verified_reference(args.reference_root, "CYJ", args.preprocessed_root).stacks[stack]
        for key, parameter in (("t1_ms", "T1"), ("t2_ms", "T2")):
            b6_maps, b6_masks = _canonical_weight_zero_maps(b6, key, native.t1_ms.shape[0])
            q002_maps, q002_masks = _canonical_weight_zero_maps(q002, key, native.t1_ms.shape[0])
            q003_6000_maps, q003_6000_masks = _canonical_weight_zero_maps(q003_6000, key, native.t1_ms.shape[0])
            q003_10000_maps, q003_10000_masks = _canonical_weight_zero_maps(q003_10000, key, native.t1_ms.shape[0])
            reference = native.t1_ms if key == "t1_ms" else native.t2_ms
            for group in range(reference.shape[0]):
                support = strict_q003_four_way_common_support(reference[group], b6_maps[group], q002_maps[group], q003_6000_maps[group], q003_10000_maps[group], native.valid_mask[group], b6_masks[group], q002_masks[group], q003_6000_masks[group], q003_10000_masks[group])
                predictions = {"FrozenMLP_B6": b6_maps[group], "Q002_S640": q002_maps[group], "Q003_S640_6000": q003_6000_maps[group], "Q003_S640_10000": q003_10000_maps[group]}
                map_records.append({"parameter": parameter, "stack": stack, "group_idx": group, "reference": reference[group], "predictions": predictions, "support": support})
                for method, prediction in predictions.items(): map_samples.append({"method": method, "stack": stack, "group_idx": group, "parameter": parameter, "reference": reference[group], "prediction": prediction, "support": support})
    map_rows = [{key: sample[key] for key in ("method", "stack", "group_idx", "parameter")} | agreement_metrics(sample["reference"], sample["prediction"], sample["support"], min_pixels=16) for sample in map_samples]
    write_csv(metrics / "central_no_psf_four_way_per_group_true_pooled.csv", map_rows)
    write_csv(metrics / "central_no_psf_four_way_true_pooled.csv", true_pooled_rows(map_samples, ("method", "stack", "parameter"), domain="map") + true_pooled_rows(map_samples, ("method", "parameter"), domain="map"))

    signal_rows: list[dict[str, object]] = []
    signal_samples: list[dict[str, object]] = []
    cosine_rows: list[dict[str, object]] = []
    fingerprint_candidates: list[dict[str, object]] = []
    for stack in STACKS:
        b6_path = Path(args.b6_d2_root) / "artifacts/FrozenMLP" / f"signal_reprojection_{stack}_K8.npz"
        b6 = _load(b6_path if b6_path.is_file() else Path(args.b6_d2_root) / f"signal_reprojection_{stack}_K8.npz")
        q002, q003_6000, q003_10000 = (_load(q002_root / "evaluation/signal_psf_K8" / f"signal_reprojection_{stack}.npz"), _signal_archive(run_root, 6000, stack), _signal_archive(run_root, 10000, stack))
        indexes = [_signal_identity_index(item, label) for item, label in ((b6, "B6 K8"), (q002, "Q002-S640 K8"), (q003_6000, "Q003-6000 K8"), (q003_10000, "Q003-10000 K8"))]
        if any(set(index) != set(indexes[0]) for index in indexes[1:]): raise ValueError("Four-way K8 identity sets differ.")
        for pair, b6_row in indexes[0].items():
            rows = [index[pair] for index in indexes]
            for key in ("observed", "timing9_ms"):
                if any(not np.array_equal(b6[key][b6_row], archive[key][row], equal_nan=True) for archive, row in zip((q002, q003_6000, q003_10000), rows[1:])): raise ValueError(f"Four-way K8 {key} provenance differs for {pair}.")
            support = strict_q003_four_way_common_support(b6["observed"][b6_row], b6["predicted"][b6_row], q002["predicted"][rows[1]], q003_6000["predicted"][rows[2]], q003_10000["predicted"][rows[3]], np.ones_like(b6["masks"][b6_row], bool), b6["masks"][b6_row], q002["masks"][rows[1]], q003_6000["masks"][rows[2]], q003_10000["masks"][rows[3]])
            for method, prediction in (("FrozenMLP_B6", b6["predicted"][b6_row]), ("Q002_S640", q002["predicted"][rows[1]]), ("Q003_S640_6000", q003_6000["predicted"][rows[2]]), ("Q003_S640_10000", q003_10000["predicted"][rows[3]])):
                signal_rows.append({"method": method, "stack": stack, "group_idx": pair[0], "weight_idx": pair[1], **signal_agreement_metrics(b6["observed"][b6_row], prediction, support, min_pixels=16)})
                signal_samples.append({"method": method, "stack": stack, "group_idx": pair[0], "weight_idx": pair[1], "support_provenance": "observed AND B6_valid AND Q002_S640_valid AND Q003_6000_valid AND Q003_10000_valid AND finite", "reference": b6["observed"][b6_row], "prediction": prediction, "support": support})
        for group in sorted(np.unique(b6["group_idx"]).tolist()):
            pairs = [(int(group), weight) for weight in range(10)]; rows = [[index[pair] for pair in pairs] for index in indexes]
            support = strict_fingerprint_four_way_common_support(b6["predicted"][rows[0]], q002["predicted"][rows[1]], q003_6000["predicted"][rows[2]], q003_10000["predicted"][rows[3]], b6["masks"][rows[0]], q002["masks"][rows[1]], q003_6000["masks"][rows[2]], q003_10000["masks"][rows[3]]) & np.isfinite(b6["observed"][rows[0]]).all(axis=0)
            for method, prediction in (("FrozenMLP_B6", b6["predicted"][rows[0]]), ("Q002_S640", q002["predicted"][rows[1]]), ("Q003_S640_6000", q003_6000["predicted"][rows[2]]), ("Q003_S640_10000", q003_10000["predicted"][rows[3]])):
                cosine_rows.append({"method": method, "stack": stack, "group_idx": int(group), "role": "observed_fingerprint_fidelity_four_way_common_support", **fingerprint_cosine_summary(prediction, b6["observed"][rows[0]], support)})
            valid = np.argwhere(support)
            if valid.size:
                row_col = tuple(int(value) for value in valid[0]); fingerprint_candidates.append({"identity": (stack, int(group), *row_col), "stack": stack, "group_idx": int(group), "row_col": row_col, "vectors": {"Observed": b6["observed"][rows[0], row_col[0], row_col[1]], "FrozenMLP_B6": b6["predicted"][rows[0], row_col[0], row_col[1]], "Q002_S640": q002["predicted"][rows[1], row_col[0], row_col[1]], "Q003_S640_6000": q003_6000["predicted"][rows[2], row_col[0], row_col[1]], "Q003_S640_10000": q003_10000["predicted"][rows[3], row_col[0], row_col[1]]}})
    write_csv(metrics / "signal_psf_K8_four_way_per_group_common_support.csv", signal_rows)
    write_csv(metrics / "signal_psf_K8_four_way_true_pooled.csv", true_pooled_rows(signal_samples, ("method", "stack", "weight_idx", "support_provenance"), domain="signal") + true_pooled_rows(signal_samples, ("method", "stack", "group_idx", "support_provenance"), domain="signal") + true_pooled_rows(signal_samples, ("method", "stack", "support_provenance"), domain="signal") + true_pooled_rows(signal_samples, ("method", "support_provenance"), domain="signal"))
    write_csv(metrics / "fingerprint_cosine_four_way_read_only.csv", cosine_rows)

    visualization = output / "visualization"; visualization.mkdir()
    render_map_montages(map_records, visualization / "all_slices")
    representative_ids = set(deterministic_representatives([(record["stack"], int(record["group_idx"]), 0, 0) for record in map_records], limit=6))
    representative_maps = [record for record in map_records if (record["stack"], int(record["group_idx"]), 0, 0) in representative_ids]
    render_map_montages(representative_maps, visualization / "representative_montages")
    selected_ids = set(deterministic_representatives([candidate["identity"] for candidate in fingerprint_candidates], limit=9))
    selected = [candidate for candidate in fingerprint_candidates if candidate["identity"] in selected_ids]
    render_fingerprint_curves(selected, visualization / "fingerprint_curves")
    visualization_manifest = {"reconstruction_git_sha": route["reconstruction_git_sha"], "evaluation_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "source_checkpoint_paths": {"q003_6000": str(primary_checkpoint), "q003_10000": str(final_checkpoint), "q002_s640": str(q002_checkpoint)}, "source_checkpoint_sha256": {"q003_6000": _sha(primary_checkpoint), "q003_10000": _sha(final_checkpoint), "q002_s640": _sha(q002_checkpoint)}, "b6_comparator_identity": str(args.b6_map_root), "selected_groups_anchors": [{"stack": item["stack"], "group_idx": item["group_idx"], "row_col": item["row_col"]} for item in selected], "representative_group_rule": "lexicographic_stack_group_identity", "display": visualization_display_contract(), "common_support": "reference/observed AND B6_valid AND Q002_S640_valid AND Q003_6000_valid AND Q003_10000_valid AND finite"}
    write_visualization_manifest(visualization, visualization_manifest)
    manifest = {"route": route["route"], "evaluation_git_sha": visualization_manifest["evaluation_git_sha"], "evaluation_git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()), "common_support": "reference_valid/observed AND B6_valid AND Q002_S640_valid AND Q003_6000_valid AND Q003_10000_valid AND finite", "primary_comparison": "Q003_S640_6000_vs_Q002_S640", "secondary_extension_comparison": "Q003_S640_10000_vs_Q003_S640_6000", "final_utility_comparisons": ["Q003_S640_6000_vs_FrozenMLP_B6", "Q003_S640_10000_vs_FrozenMLP_B6"], "historical_artifacts": "read_only"}
    write_json(output / "evaluation_manifest.json", manifest)
    (output / "RESULT_SUMMARY.md").write_text("# Q003-S640 10k four-way evaluation\n\nPrimary: Q003-6000 vs Q002-S640. Extension comparison: Q003-10000 vs Q003-6000. All metrics use one strict four-way support.\n", encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "q002-s640-run-root", "b6-map-root", "b6-d2-root", "reference-root", "preprocessed-root", "output"): parser.add_argument(f"--{name}", required=True)
    return parser


def main() -> None:
    print(run_q003_s640_10k_four_way(build_parser().parse_args()))


if __name__ == "__main__":
    main()
