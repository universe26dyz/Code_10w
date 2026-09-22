"""Local CPU/MATLAB Stage-2 for transferred deterministic PSF K=8 signals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .psf_k8_postprocess_common import (
    compute_metrics,
    render_signal_qc,
    run_matcher,
    verify_stage1,
    write_json,
    write_synthetic_mat,
)
from .reference_2d import STACKS, sha256
from .reference_paths import load_reference_paths


def run(args: argparse.Namespace) -> dict:
    stage1 = args.stage1_root.resolve()
    stage1_manifest = verify_stage1(stage1, args.subject_id)
    paths = load_reference_paths(args.reference_paths, native_reference_root=args.native_reference, myocardium_mask_root=args.mask_bundle)
    for warning in paths.warnings:
        print(f"WARNING: {warning}")
    if not paths.skip_metrics:
        mask_manifest = json.loads((paths.myocardium_mask_root / "manifest.json").read_text(encoding="utf-8"))
        if mask_manifest.get("schema") != "exact_native_myocardium_bundle/v2" or mask_manifest.get("status") != "PASS":
            raise ValueError("Stage 2 requires a PASS myocardium_masks_v2 bundle.")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty Stage-2 output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    synthetic_root = output / "synthetic_signal_mat"; synthetic_root.mkdir()
    apparent_root = output / "synthetic_apparent_maps"; apparent_root.mkdir()
    metrics_root = output / "metrics"; metrics_root.mkdir()
    for stack in STACKS:
        signal_path = stage1 / stage1_manifest["signal_outputs"][stack]["relative_path"]
        synthetic = synthetic_root / f"{stack}_synthetic_signal.mat"
        mapping = apparent_root / f"{stack}_apparent_mapping.mat"
        write_synthetic_mat(signal_path, synthetic, stack)
        source = args.preprocessed_root / args.subject_id / stack / "preprocessed.mat"
        if not source.is_file():
            raise FileNotFoundError(f"Current exact preprocessed MAT is missing: {source}")
        run_matcher(source, synthetic, mapping, args.matlab_executable, args.dictionary_cache_root)
    render_signal_qc(stage1 / "signals", output / "qc_all_slices_all_weights")
    metrics = compute_metrics(paths.native_reference_root, args.preprocessed_root, stage1 / "signals", apparent_root, paths.myocardium_mask_root, metrics_root, args.subject_id) if not paths.skip_metrics else None
    manifest = {
        "schema": "trad_psf_k8_stage2/v1", "status": "PASS" if not paths.skip_metrics else "PARTIAL_EVALUATION_SKIPPED", "subject_id": args.subject_id,
        "stage1": {"path": str(stage1), "manifest_sha256": sha256(stage1 / "stage1_manifest.json"), "verified": True},
        "reference_data": {"native_reference_root": str(paths.native_reference_root) if paths.native_reference_root else None, "myocardium_mask_root": str(paths.myocardium_mask_root) if paths.myocardium_mask_root else None, "warnings": list(paths.warnings)},
        "native_reference_manifest_sha256": sha256(paths.native_reference_root / "native_reference_manifest.json") if paths.native_reference_root else None,
        "mask_bundle_manifest_sha256": sha256(paths.myocardium_mask_root / "manifest.json") if paths.myocardium_mask_root else None,
        "dictionary_matching": {"matlab_wrapper": str((Path(__file__).resolve().parent / "matlab" / "build_synthetic_apparent_maps.m").resolve()), "dictionary_cache_root": str(args.dictionary_cache_root) if args.dictionary_cache_root else None, "physics_parameters_changed": False},
        "metrics": metrics,
    }
    write_json(manifest, output / "evaluation_manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True); parser.add_argument("--stage1-root", required=True, type=Path)
    parser.add_argument("--preprocessed-root", required=True, type=Path); parser.add_argument("--native-reference", type=Path)
    parser.add_argument("--mask-bundle", type=Path); parser.add_argument("--reference-paths", type=Path, default=Path(__file__).resolve().parents[3] / "config" / "reference_paths.yaml")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--matlab-executable", default="matlab")
    parser.add_argument("--dictionary-cache-root", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), indent=2, sort_keys=True, allow_nan=True))
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
