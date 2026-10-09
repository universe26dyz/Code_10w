"""Read-only B6/Q002 common-support evaluator; it never constructs a model."""

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

from .evaluation import evaluation_plan
from .metrics import (
    fingerprint_cosine_summary,
    strict_fingerprint_common_support,
    strict_fingerprint_three_way_common_support,
    strict_q002_common_support,
    strict_q002_three_way_common_support,
)


STACKS = ("sax", "2ch", "4ch")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]) for name in source.files}


def _canonical_weight_zero_maps(archive: dict[str, np.ndarray], key: str, group_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Select exactly weight zero by declared group identity, never row position."""
    result, masks = [], []
    for group in range(group_count):
        index = np.flatnonzero((archive["group_idx"] == group) & (archive["weight_idx"] == 0))
        if index.size != 1: raise ValueError(f"Expected exactly one weight-zero map for group {group}.")
        result.append(archive[key][index[0]]); masks.append(archive["masks"][index[0]])
    return np.stack(result), np.stack(masks)


def _signal_identity_index(archive: dict[str, np.ndarray], label: str) -> dict[tuple[int, int], int]:
    """Validate the complete 0..9 fingerprint identity contract before pairing."""
    groups, weights = np.asarray(archive["group_idx"], np.int64), np.asarray(archive["weight_idx"], np.int64)
    if groups.shape != weights.shape:
        raise ValueError(f"{label} group_idx/weight_idx shapes differ.")
    index: dict[tuple[int, int], int] = {}
    for row, pair in enumerate(zip(groups.tolist(), weights.tolist())):
        if pair in index:
            raise ValueError(f"{label} has duplicate group-weight identity {pair}.")
        index[pair] = row
    for group in np.unique(groups):
        group_weights = {weight for current_group, weight in index if current_group == int(group)}
        if group_weights != set(range(10)):
            raise ValueError(f"{label} group {int(group)} must contain exactly weights 0..9.")
    return index


def _validated_route_checkpoint(run_root: Path, label: str) -> tuple[dict[str, object], Path]:
    route = json.loads((run_root / "q002_route_manifest.json").read_text())
    checkpoint = run_root / "model.pt"
    if not checkpoint.is_file() or _sha(checkpoint) != route["q002_checkpoint_sha256"]:
        raise ValueError(f"{label} checkpoint is missing or route-manifest hash does not match.")
    return route, checkpoint


def _run_three_way(args: argparse.Namespace, run_root: Path, output: Path, s640_route: dict[str, object], s640_checkpoint: Path) -> Path:
    """Evaluate B6, immutable Q002-64, and Q002-S640 on exactly one support."""

    q002_64_root = Path(args.q002_64_run_root)
    q002_64_route, q002_64_checkpoint = _validated_route_checkpoint(q002_64_root, "Q002-64")
    if s640_route["experiment_id"] != "Q002S640_B6_roi_same_anchor_mse" or q002_64_route["experiment_id"] != "Q002_B6_roi_same_anchor_mse":
        raise ValueError("Three-way evaluation requires immutable Q002-S640 and Q002-64 route manifests.")
    output.mkdir(parents=True); metrics = output / "metrics"; metrics.mkdir()
    map_samples: list[dict[str, object]] = []
    for stack in STACKS:
        s640 = _load(run_root / "evaluation" / "mapping_central_no_psf" / f"t1_t2_native_plane_{stack}.npz")
        q002_64 = _load(q002_64_root / "evaluation" / "mapping_central_no_psf" / f"t1_t2_native_plane_{stack}.npz")
        b6 = _load(Path(args.b6_map_root) / f"t1_t2_native_plane_{stack}.npz")
        native = load_verified_reference(args.reference_root, "CYJ", args.preprocessed_root).stacks[stack]
        for key, label in (("t1_ms", "T1"), ("t2_ms", "T2")):
            b6_maps, b6_masks = _canonical_weight_zero_maps(b6, key, native.t1_ms.shape[0])
            q64_maps, q64_masks = _canonical_weight_zero_maps(q002_64, key, native.t1_ms.shape[0])
            s640_maps, s640_masks = _canonical_weight_zero_maps(s640, key, native.t1_ms.shape[0])
            reference = native.t1_ms if key == "t1_ms" else native.t2_ms
            for group in range(reference.shape[0]):
                support = strict_q002_three_way_common_support(reference[group], b6_maps[group], q64_maps[group], s640_maps[group], b6_masks[group], q64_masks[group], s640_masks[group], native.valid_mask[group])
                for method, prediction in (("FrozenMLP_B6", b6_maps[group]), ("Q002_64", q64_maps[group]), ("Q002_S640", s640_maps[group])):
                    map_samples.append({"method": method, "stack": stack, "group_idx": group, "parameter": label, "reference": reference[group], "prediction": prediction, "support": support})
    map_rows = [{key: sample[key] for key in ("method", "stack", "group_idx", "parameter")} | agreement_metrics(sample["reference"], sample["prediction"], sample["support"], min_pixels=16) for sample in map_samples]
    write_csv(metrics / "central_no_psf_three_way_per_group_true_pooled.csv", map_rows)
    map_pooled = true_pooled_rows(map_samples, ("method", "stack", "parameter"), domain="map") + true_pooled_rows(map_samples, ("method", "parameter"), domain="map")
    write_csv(metrics / "central_no_psf_three_way_true_pooled.csv", map_pooled)

    signal_rows: list[dict[str, object]] = []
    signal_samples: list[dict[str, object]] = []
    cosine_rows: list[dict[str, object]] = []
    for stack in STACKS:
        s640 = _load(run_root / "evaluation" / "signal_psf_K8" / f"signal_reprojection_{stack}.npz")
        q002_64 = _load(q002_64_root / "evaluation" / "signal_psf_K8" / f"signal_reprojection_{stack}.npz")
        b6_path = Path(args.b6_d2_root) / "artifacts" / "FrozenMLP" / f"signal_reprojection_{stack}_K8.npz"
        b6 = _load(b6_path if b6_path.is_file() else Path(args.b6_d2_root) / f"signal_reprojection_{stack}_K8.npz")
        b6_index, q64_index, s640_index = (_signal_identity_index(b6, "B6 K8"), _signal_identity_index(q002_64, "Q002-64 K8"), _signal_identity_index(s640, "Q002-S640 K8"))
        if set(b6_index) != set(q64_index) or set(b6_index) != set(s640_index):
            raise ValueError("B6/Q002-64/Q002-S640 K8 identity sets differ.")
        for s640_row in range(s640["observed"].shape[0]):
            pair = (int(s640["group_idx"][s640_row]), int(s640["weight_idx"][s640_row]))
            b6_row, q64_row = b6_index[pair], q64_index[pair]
            for key in ("observed", "timing9_ms"):
                if not np.array_equal(s640[key][s640_row], b6[key][b6_row], equal_nan=True) or not np.array_equal(s640[key][s640_row], q002_64[key][q64_row], equal_nan=True):
                    raise ValueError(f"Three-way K8 {key} provenance differs for {pair}.")
            support = strict_q002_three_way_common_support(s640["observed"][s640_row], b6["predicted"][b6_row], q002_64["predicted"][q64_row], s640["predicted"][s640_row], b6["masks"][b6_row], q002_64["masks"][q64_row], s640["masks"][s640_row], np.ones_like(s640["masks"][s640_row], bool))
            for method, prediction in (("FrozenMLP_B6", b6["predicted"][b6_row]), ("Q002_64", q002_64["predicted"][q64_row]), ("Q002_S640", s640["predicted"][s640_row])):
                row = {"method": method, "stack": stack, "group_idx": pair[0], "weight_idx": pair[1], **signal_agreement_metrics(s640["observed"][s640_row], prediction, support, min_pixels=16)}
                signal_rows.append(row)
                signal_samples.append({"method": method, "stack": stack, "group_idx": pair[0], "weight_idx": pair[1], "support_provenance": "observed AND B6_valid AND Q002_64_valid AND Q002_S640_valid AND finite", "reference": s640["observed"][s640_row], "prediction": prediction, "support": support})
        for group in np.unique(s640["group_idx"]):
            rows = np.flatnonzero(s640["group_idx"] == group)
            if rows.size != 10 or not np.array_equal(np.sort(s640["weight_idx"][rows]), np.arange(10)):
                raise ValueError("Q002-S640 K8 archive lacks one ordered ten-weight fingerprint per group.")
            ordered = rows[np.argsort(s640["weight_idx"][rows])]
            pairs = [(int(s640["group_idx"][row]), int(s640["weight_idx"][row])) for row in ordered]
            b6_ordered = np.asarray([b6_index[pair] for pair in pairs]); q64_ordered = np.asarray([q64_index[pair] for pair in pairs])
            support = strict_fingerprint_three_way_common_support(b6["predicted"][b6_ordered], q002_64["predicted"][q64_ordered], s640["predicted"][ordered], b6["masks"][b6_ordered], q002_64["masks"][q64_ordered], s640["masks"][ordered]) & np.isfinite(s640["observed"][ordered]).all(axis=0)
            for method, prediction in (("FrozenMLP_B6", b6["predicted"][b6_ordered]), ("Q002_64", q002_64["predicted"][q64_ordered]), ("Q002_S640", s640["predicted"][ordered])):
                cosine_rows.append({"method": method, "stack": stack, "group_idx": int(group), "role": "observed_fingerprint_fidelity_three_way_common_support", **fingerprint_cosine_summary(prediction, s640["observed"][ordered], support)})
    write_csv(metrics / "signal_psf_K8_three_way_per_group_common_support.csv", signal_rows)
    signal_pooled = true_pooled_rows(signal_samples, ("method", "stack", "weight_idx", "support_provenance"), domain="signal") + true_pooled_rows(signal_samples, ("method", "stack", "group_idx", "support_provenance"), domain="signal") + true_pooled_rows(signal_samples, ("method", "stack", "support_provenance"), domain="signal") + true_pooled_rows(signal_samples, ("method", "support_provenance"), domain="signal")
    write_csv(metrics / "signal_psf_K8_three_way_true_pooled.csv", signal_pooled)
    write_csv(metrics / "fingerprint_cosine_three_way_read_only.csv", cosine_rows)
    cosine_summary = []
    for method in ("FrozenMLP_B6", "Q002_64", "Q002_S640"):
        members = [row for row in cosine_rows if row["method"] == method]
        for scope, selected in (("global_macro_groups", members), *[(f"stack:{stack}:macro_groups", [row for row in members if row["stack"] == stack]) for stack in STACKS]):
            values = np.asarray([row["mean_cosine"] for row in selected], float)
            cosine_summary.append({"method": method, "scope": scope, "macro_group_mean_cosine": float(np.nanmean(values)) if values.size else float("nan"), "macro_group_median_cosine": float(np.nanmedian(values)) if values.size else float("nan"), "support_N": int(sum(row["support_N"] for row in selected)), "near_zero_norm_count": int(sum(row["near_zero_norm_count"] for row in selected))})
    write_csv(metrics / "fingerprint_cosine_three_way_macro_group_summary.csv", cosine_summary)
    evaluation_git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {"route": s640_route["route"], "q002_64_route": q002_64_route["route"], "evaluation_git_sha": evaluation_git_sha, "evaluation_git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()), "q002_s640_reconstruction_git_sha": s640_route["reconstruction_git_sha"], "q002_64_reconstruction_git_sha": q002_64_route["reconstruction_git_sha"], "q002_s640_checkpoint_sha256": _sha(s640_checkpoint), "q002_64_checkpoint_sha256": _sha(q002_64_checkpoint), "evaluation": evaluation_plan(_sha(s640_checkpoint)), "common_support": "reference_valid AND B6_valid AND Q002_64_valid AND Q002_S640_valid AND finite", "primary_comparison": "Q002_S640_vs_Q002_64", "secondary_comparison": "Q002_S640_vs_FrozenMLP_B6", "historical_artifacts": "read_only"}
    write_json(output / "evaluation_manifest.json", manifest)
    (output / "RESULT_SUMMARY.md").write_text("# Q002-S640 three-way evaluation\n\nPrimary comparison: Q002-S640 vs Q002-64. Secondary comparison: Q002-S640 vs FrozenMLP B6. All reported method metrics use one strict three-way common support.\n", encoding="utf-8")
    return output


def run(args: argparse.Namespace) -> Path:
    run_root, output = Path(args.run_root), Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty Q002 evaluation root: {output}")
    route, checkpoint = _validated_route_checkpoint(run_root, "Q002")
    if args.q002_64_run_root is not None:
        return _run_three_way(args, run_root, output, route, checkpoint)
    output.mkdir(parents=True); metrics = output / "metrics"; metrics.mkdir()
    samples = []
    for stack in STACKS:
        candidate = _load(run_root / "evaluation" / "mapping_central_no_psf" / f"t1_t2_native_plane_{stack}.npz")
        b6 = _load(Path(args.b6_map_root) / f"t1_t2_native_plane_{stack}.npz")
        verified_reference = load_verified_reference(args.reference_root, "CYJ", args.preprocessed_root)
        native = verified_reference.stacks[stack]
        reference = {"t1_ms": native.t1_ms, "t2_ms": native.t2_ms, "valid_mask": native.valid_mask}
        for key, label in (("t1_ms", "T1"), ("t2_ms", "T2")):
            b6_maps, b6_masks = _canonical_weight_zero_maps(b6, key, reference[key].shape[0])
            candidate_maps, candidate_masks = _canonical_weight_zero_maps(candidate, key, reference[key].shape[0])
            for group in range(reference[key].shape[0]):
                support = strict_q002_common_support(reference[key][group], b6_maps[group], candidate_maps[group], b6_masks[group], candidate_masks[group], reference["valid_mask"][group])
                for method, prediction in (("FrozenMLP_B6", b6_maps[group]), ("Q002", candidate_maps[group])):
                    samples.append({"method": method, "stack": stack, "group_idx": group, "parameter": label, "reference": reference[key][group], "prediction": prediction, "support": support})
    rows = []
    for sample in samples:
        rows.append({key: sample[key] for key in ("method", "stack", "group_idx", "parameter")} | agreement_metrics(sample["reference"], sample["prediction"], sample["support"], min_pixels=16))
    write_csv(metrics / "central_no_psf_per_group_true_pooled.csv", rows)
    pooled = true_pooled_rows(samples, ("method", "stack", "parameter"), domain="map") + true_pooled_rows(samples, ("method", "parameter"), domain="map")
    write_csv(metrics / "central_no_psf_true_pooled.csv", pooled)
    signal_rows, signal_samples, cosine_rows = [], [], []
    for stack in STACKS:
        candidate = _load(run_root / "evaluation" / "signal_psf_K8" / f"signal_reprojection_{stack}.npz")
        b6_path = Path(args.b6_d2_root) / "artifacts" / "FrozenMLP" / f"signal_reprojection_{stack}_K8.npz"
        if not b6_path.is_file(): b6_path = Path(args.b6_d2_root) / f"signal_reprojection_{stack}_K8.npz"
        b6 = _load(b6_path)
        b6_index, candidate_index = _signal_identity_index(b6, "B6 K8"), _signal_identity_index(candidate, "Q002 K8")
        if set(b6_index) != set(candidate_index):
            raise ValueError("B6/Q002 K8 identity sets differ (missing or extra group-weight pair).")
        for index in range(candidate["observed"].shape[0]):
            pair = (int(candidate["group_idx"][index]), int(candidate["weight_idx"][index]))
            if pair not in b6_index: raise ValueError("B6/Q002 K8 group-weight identities differ.")
            baseline_index = b6_index[pair]
            for key in ("observed", "timing9_ms"):
                if not np.array_equal(candidate[key][index], b6[key][baseline_index], equal_nan=True): raise ValueError(f"B6/Q002 K8 {key} provenance differs.")
            support = strict_q002_common_support(candidate["observed"][index], b6["predicted"][baseline_index], candidate["predicted"][index], b6["masks"][baseline_index], candidate["masks"][index], np.ones_like(candidate["masks"][index], bool))
            for method, prediction in (("FrozenMLP_B6", b6["predicted"][baseline_index]), ("Q002", candidate["predicted"][index])):
                signal_rows.append({"method": method, "stack": stack, "group_idx": int(candidate["group_idx"][index]), "weight_idx": int(candidate["weight_idx"][index]), **signal_agreement_metrics(candidate["observed"][index], prediction, support, min_pixels=16)})
                signal_samples.append({"method": method, "stack": stack, "group_idx": int(candidate["group_idx"][index]), "weight_idx": int(candidate["weight_idx"][index]), "support_provenance": "observed AND B6_valid AND Q002_valid AND finite", "reference": candidate["observed"][index], "prediction": prediction, "support": support})
        for group in np.unique(candidate["group_idx"]):
            rows_for_group = np.flatnonzero(candidate["group_idx"] == group)
            if rows_for_group.size != 10 or not np.array_equal(np.sort(candidate["weight_idx"][rows_for_group]), np.arange(10)):
                raise ValueError("K8 signal archive lacks one ordered ten-weight fingerprint per group.")
            ordered = rows_for_group[np.argsort(candidate["weight_idx"][rows_for_group])]
            baseline_ordered = np.asarray([b6_index[(int(candidate["group_idx"][i]), int(candidate["weight_idx"][i]))] for i in ordered])
            support = strict_fingerprint_common_support(b6["predicted"][baseline_ordered], candidate["predicted"][ordered], b6["masks"][baseline_ordered], candidate["masks"][ordered]) & np.isfinite(candidate["observed"][ordered]).all(axis=0)
            for method, prediction in (("FrozenMLP_B6", b6["predicted"][baseline_ordered]), ("Q002", candidate["predicted"][ordered])):
                cosine_rows.append({"method": method, "stack": stack, "group_idx": int(group), "role": "observed_fingerprint_fidelity", **fingerprint_cosine_summary(prediction, candidate["observed"][ordered], support)})
            cosine_rows.append({"method": "B6_vs_Q002", "stack": stack, "group_idx": int(group), "role": "inter_method_similarity", **fingerprint_cosine_summary(b6["predicted"][baseline_ordered], candidate["predicted"][ordered], support)})
    write_csv(metrics / "signal_psf_K8_per_group_common_support.csv", signal_rows)
    pooled_signal = true_pooled_rows(signal_samples, ("method", "stack", "weight_idx", "support_provenance"), domain="signal") + true_pooled_rows(signal_samples, ("method", "stack", "support_provenance"), domain="signal") + true_pooled_rows(signal_samples, ("method", "support_provenance"), domain="signal")
    write_csv(metrics / "signal_psf_K8_true_pooled.csv", pooled_signal)
    write_csv(metrics / "fingerprint_cosine_read_only.csv", cosine_rows)
    cosine_summary = []
    for method in ("FrozenMLP_B6", "Q002"):
        members = [row for row in cosine_rows if row["method"] == method]
        for scope, selected in (("global", members), *[(f"stack:{stack}", [row for row in members if row["stack"] == stack]) for stack in STACKS]):
            values = np.asarray([row["mean_cosine"] for row in selected], float); cosine_summary.append({"method": method, "scope": scope, "mean_cosine": float(np.nanmean(values)) if values.size else float("nan"), "median_cosine": float(np.nanmedian(values)) if values.size else float("nan"), "support_N": int(sum(row["support_N"] for row in selected)), "near_zero_norm_count": int(sum(row["near_zero_norm_count"] for row in selected))})
    write_csv(metrics / "fingerprint_cosine_summary.csv", cosine_summary)
    evaluation_git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(); evaluation_git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    manifest = {"route": route["route"], "evaluation_git_sha": evaluation_git_sha, "evaluation_git_dirty": evaluation_git_dirty, "q002_reconstruction_git_sha": route["reconstruction_git_sha"], "q002_checkpoint_sha256": _sha(checkpoint), "b6_map_sha256": {stack: _sha(Path(args.b6_map_root) / f"t1_t2_native_plane_{stack}.npz") for stack in STACKS}, "b6_k8_root": str(args.b6_d2_root), "evaluation": evaluation_plan(_sha(checkpoint)), "common_support": "reference_valid AND B6_valid AND Q002_valid AND finite", "q001_artifacts": "read_only_not_comparable_without_audited_mapping"}
    write_json(output / "evaluation_manifest.json", manifest)
    (output / "RESULT_SUMMARY.md").write_text(f"# Q002 evaluation\n\nreconstruction Git SHA: {manifest['q002_reconstruction_git_sha']}\nevaluation Git SHA: {evaluation_git_sha}\ncheckpoint SHA256: {manifest['q002_checkpoint_sha256']}\n\nPrimary metrics use strict B6/Q002/reference common support. Q001 artifacts are read-only historical context.\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True); parser.add_argument("--b6-map-root", required=True); parser.add_argument("--b6-d2-root", required=True); parser.add_argument("--reference-root", required=True); parser.add_argument("--preprocessed-root", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--q002-64-run-root")
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
