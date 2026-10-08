"""Read-only B6/Q002 common-support evaluator; it never constructs a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from quality_experiments.D2_k8_signal_domain.core import write_csv, write_json
from quality_experiments.Q001_recon.metrics import true_pooled_rows
from trad.evaluation.scmr.metrics import agreement_metrics

from .evaluation import evaluation_plan
from .metrics import strict_fingerprint_common_support, strict_q002_common_support


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


def run(args: argparse.Namespace) -> Path:
    run_root, output = Path(args.run_root), Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty Q002 evaluation root: {output}")
    route = json.loads((run_root / "q002_route_manifest.json").read_text())
    checkpoint = run_root / "model.pt"
    if not checkpoint.is_file() or _sha(checkpoint) != route["q002_checkpoint_sha256"]:
        raise ValueError("Q002 checkpoint is missing or route-manifest hash does not match.")
    output.mkdir(parents=True); metrics = output / "metrics"; metrics.mkdir()
    samples = []
    for stack in STACKS:
        candidate = _load(run_root / "evaluation" / "mapping_central_no_psf" / f"t1_t2_native_plane_{stack}.npz")
        b6 = _load(Path(args.b6_map_root) / f"t1_t2_native_plane_{stack}.npz")
        reference = _load(Path(args.reference_root) / f"{stack}.npz")
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
    signal_rows, cosine_rows = [], []
    for stack in STACKS:
        candidate = _load(run_root / "evaluation" / "signal_psf_K8" / f"signal_reprojection_{stack}.npz")
        b6_path = Path(args.b6_d2_root) / "artifacts" / "FrozenMLP" / f"signal_reprojection_{stack}_K8.npz"
        if not b6_path.is_file(): b6_path = Path(args.b6_d2_root) / f"signal_reprojection_{stack}_K8.npz"
        b6 = _load(b6_path)
        b6_index = {(int(b6["group_idx"][i]), int(b6["weight_idx"][i])): i for i in range(b6["observed"].shape[0])}
        for index in range(candidate["observed"].shape[0]):
            pair = (int(candidate["group_idx"][index]), int(candidate["weight_idx"][index]))
            if pair not in b6_index: raise ValueError("B6/Q002 K8 group-weight identities differ.")
            baseline_index = b6_index[pair]
            for key in ("observed", "timing9_ms"):
                if not np.array_equal(candidate[key][index], b6[key][baseline_index], equal_nan=True): raise ValueError(f"B6/Q002 K8 {key} provenance differs.")
            support = strict_q002_common_support(candidate["observed"][index], b6["predicted"][baseline_index], candidate["predicted"][index], b6["masks"][baseline_index], candidate["masks"][index], np.ones_like(candidate["masks"][index], bool))
            for method, prediction in (("FrozenMLP_B6", b6["predicted"][index]), ("Q002", candidate["predicted"][index])):
                signal_rows.append({"method": method, "stack": stack, "group_idx": int(candidate["group_idx"][index]), "weight_idx": int(candidate["weight_idx"][index]), **agreement_metrics(candidate["observed"][index], prediction, support, min_pixels=16)})
        for group in np.unique(candidate["group_idx"]):
            rows_for_group = np.flatnonzero(candidate["group_idx"] == group)
            if rows_for_group.size != 10 or not np.array_equal(np.sort(candidate["weight_idx"][rows_for_group]), np.arange(10)):
                raise ValueError("K8 signal archive lacks one ordered ten-weight fingerprint per group.")
            ordered = rows_for_group[np.argsort(candidate["weight_idx"][rows_for_group])]
            baseline_ordered = np.asarray([b6_index[(int(candidate["group_idx"][i]), int(candidate["weight_idx"][i]))] for i in ordered])
            support = strict_fingerprint_common_support(b6["predicted"][baseline_ordered], candidate["predicted"][ordered], b6["masks"][baseline_ordered], candidate["masks"][ordered])
            dot = (b6["predicted"][baseline_ordered] * candidate["predicted"][ordered]).sum(axis=0); norms = np.linalg.norm(b6["predicted"][baseline_ordered], axis=0) * np.linalg.norm(candidate["predicted"][ordered], axis=0)
            cosine_rows.append({"stack": stack, "group_idx": int(group), "N": int(support.sum()), "fingerprint_cosine": float(np.mean((dot / norms)[support])) if support.any() else float("nan"), "role": "read_only_diagnostic"})
    write_csv(metrics / "signal_psf_K8_per_group_common_support.csv", signal_rows)
    write_csv(metrics / "fingerprint_cosine_read_only.csv", cosine_rows)
    manifest = {"route": route["route"], "q002_checkpoint_sha256": _sha(checkpoint), "b6_map_sha256": {stack: _sha(Path(args.b6_map_root) / f"t1_t2_native_plane_{stack}.npz") for stack in STACKS}, "b6_k8_root": str(args.b6_d2_root), "evaluation": evaluation_plan(_sha(checkpoint)), "common_support": "reference_valid AND B6_valid AND Q002_valid AND finite", "q001_artifacts": "read_only_not_comparable_without_audited_mapping"}
    write_json(output / "evaluation_manifest.json", manifest)
    (output / "RESULT_SUMMARY.md").write_text("# Q002 evaluation\n\nPrimary metrics use strict B6/Q002/reference common support. Q001 artifacts are read-only historical context.\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True); parser.add_argument("--b6-map-root", required=True); parser.add_argument("--b6-d2-root", required=True); parser.add_argument("--reference-root", required=True); parser.add_argument("--output", required=True)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
