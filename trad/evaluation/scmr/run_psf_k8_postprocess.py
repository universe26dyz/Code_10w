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


def run(args: argparse.Namespace) -> dict:
    stage1 = args.stage1_root.resolve()
    stage1_manifest = verify_stage1(stage1, args.subject_id)
    mask_manifest = json.loads((args.mask_bundle / "manifest.json").read_text(encoding="utf-8"))
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
        run_matcher(source, synthetic, mapping, args.matlab_executable)
    render_signal_qc(stage1 / "signals", output / "qc_all_slices_all_weights")
    metrics = compute_metrics(args.native_reference, args.preprocessed_root, stage1 / "signals", apparent_root, args.mask_bundle, metrics_root, args.subject_id)
    manifest = {
        "schema": "trad_psf_k8_stage2/v1", "status": "PASS", "subject_id": args.subject_id,
        "stage1": {"path": str(stage1), "manifest_sha256": sha256(stage1 / "stage1_manifest.json"), "verified": True},
        "native_reference_manifest_sha256": sha256(args.native_reference / "native_reference_manifest.json"),
        "mask_bundle_manifest_sha256": sha256(args.mask_bundle / "manifest.json"),
        "dictionary_matching": {"matlab_wrapper": str((Path(__file__).resolve().parent / "matlab" / "build_synthetic_apparent_maps.m").resolve()), "physics_parameters_changed": False},
        "metrics": metrics,
    }
    write_json(manifest, output / "evaluation_manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True); parser.add_argument("--stage1-root", required=True, type=Path)
    parser.add_argument("--preprocessed-root", required=True, type=Path); parser.add_argument("--native-reference", required=True, type=Path)
    parser.add_argument("--mask-bundle", required=True, type=Path); parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--matlab-executable", default="matlab")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), indent=2, sort_keys=True, allow_nan=True))
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
