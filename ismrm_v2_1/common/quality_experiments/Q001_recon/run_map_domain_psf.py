"""Route-aware post-hoc Q001 map-domain PSF K=32 reprojection; never reconstructs."""
from __future__ import annotations

import argparse
from argparse import Namespace
import json
from pathlib import Path

from trad.evaluation.scmr.run_map_domain_psf_comparison import run as run_map_domain_psf
from trad.evaluation.scmr.map_domain_psf import sha256
import numpy as np

from .verify_full_fov_reference import verify_full_fov_reference


SEED = 20260911


def _prepared_hashes(root: Path) -> dict[str, str]:
    paths = {stack: root / "CYJ" / stack / "observations.npz" for stack in ("sax", "2ch", "4ch")}
    if not all(path.is_file() for path in paths.values()):
        paths = {stack: root / stack / "observations.npz" for stack in paths}
    return {stack: sha256(path) for stack, path in paths.items()}


def validate_posthoc_manifest(manifest: dict, route: Path, checkpoint: Path, t1: Path, t2: Path, poses: Path, prepared: dict[str, str], reference_manifest_sha256: str, nested_manifest_sha256: str) -> None:
    required = {"mode": "Q001B", "domain": "map", "n_samples": 32, "seed": SEED, "q001_route_manifest_sha256": sha256(route), "reconstruction_checkpoint_sha256": sha256(checkpoint), "t1_volume_sha256": sha256(t1), "t2_volume_sha256": sha256(t2), "final_rigid_poses_sha256": sha256(poses), "prepared_observations_sha256": prepared, "reference_manifest_sha256": reference_manifest_sha256, "map_domain_psf_manifest_sha256": nested_manifest_sha256}
    for key, expected in required.items():
        if manifest.get(key) != expected: raise ValueError(f"Q001 post-hoc provenance mismatch: {key}.")


def _full_reference_comparisons(output: Path, reference_root: Path) -> dict[str, str]:
    manifest = json.loads((reference_root / "native_reference_manifest.json").read_text())
    hashes = {}
    for parameter, reference_key in (("T1", "t1_ms"), ("T2", "t2_ms")):
        predictions, references, supports, offset = {}, {}, {}, 0
        for stack in ("sax", "2ch", "4ch"):
            with np.load(reference_root / manifest["stacks"][stack]["map"], allow_pickle=False) as ref:
                native, valid = np.asarray(ref[reference_key]), np.asarray(ref["valid_mask"], bool)
            planes = []
            for group in range(native.shape[0]):
                with np.load(output / f"{parameter}_native_map_domain_psf_{offset + group:03d}.npz", allow_pickle=False) as current:
                    planes.append(np.asarray(current["values_ms"], np.float32))
            prediction = np.stack(planes); common = valid & np.isfinite(prediction) & np.isfinite(native)
            predictions[stack], references[stack], supports[stack] = prediction, native, common; offset += native.shape[0]
        target = output / f"{parameter}_native_map_domain_psf_comparison.npz"
        np.savez_compressed(target, **{f"{stack}_{field}": value for stack in predictions for field, value in (("predicted_ms", predictions[stack]), ("native_reference_ms", references[stack]), ("common_support", supports[stack]))})
        hashes[target.name] = sha256(target)
    return hashes


def run(args: argparse.Namespace) -> Path:
    run_root, output = Path(args.run_root), Path(args.output)
    mode = args.mode
    if mode not in {"Q001A", "Q001B"}: raise ValueError("mode must be Q001A or Q001B")
    for path in (run_root / "T1_3D.nii.gz", run_root / "T2_3D.nii.gz", run_root / "final_rigid_poses.json"):
        if not path.is_file(): raise FileNotFoundError(f"Missing final Q001 export: {path}")
    reference_root, prepared_root = Path(args.reference_root), Path(args.prepared_root)
    reference_manifest = reference_root / "native_reference_manifest.json"
    if mode == "Q001A":
        verify_full_fov_reference(reference_root, args.full_input_root)
    nested = run_map_domain_psf(Namespace(method="shared_svr", subject_id="CYJ", t1_volume=str(run_root / "T1_3D.nii.gz"), t2_volume=str(run_root / "T2_3D.nii.gz"), n_samples=32, seed=SEED, output=str(output), native_reference=str(reference_root) if mode == "Q001B" else None, preprocessed_root=str(args.preprocessed_root) if mode == "Q001B" else None, mask_bundle=None, prepared_root=str(prepared_root), final_poses=str(run_root / "final_rigid_poses.json"), t1_registered_slices=None, t2_registered_slices=None, decoder_type="FrozenMLP", run_label=f"{mode}_true_map_domain_psf_K32"))
    route = run_root / "q001_route_manifest.json"; checkpoint, t1, t2, poses = run_root / "model.pt", run_root / "T1_3D.nii.gz", run_root / "T2_3D.nii.gz", run_root / "final_rigid_poses.json"
    comparison_hashes = _full_reference_comparisons(output, reference_root) if mode == "Q001A" else {}
    record = {"schema": "q001_map_domain_psf_posthoc/v1", "experiment_id": json.loads(route.read_text())["experiment_id"], "mode": mode, "git_sha": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "run_root": str(run_root.resolve()), "q001_route_manifest": str(route.resolve()), "q001_route_manifest_sha256": sha256(route), "reconstruction_checkpoint": str(checkpoint.resolve()), "reconstruction_checkpoint_sha256": sha256(checkpoint), "t1_volume": str(t1.resolve()), "t1_volume_sha256": sha256(t1), "t2_volume": str(t2.resolve()), "t2_volume_sha256": sha256(t2), "final_rigid_poses": str(poses.resolve()), "final_rigid_poses_sha256": sha256(poses), "prepared_observations_sha256": _prepared_hashes(prepared_root), "reference_root": str(reference_root.resolve()), "reference_manifest": str(reference_manifest.resolve()), "reference_manifest_sha256": sha256(reference_manifest), "domain": "map", "n_samples": 32, "seed": SEED, "implementation": "trad.evaluation.scmr.map_domain_psf", "map_domain_psf_manifest": str(nested.resolve()), "map_domain_psf_manifest_sha256": sha256(nested), "output_artifact_sha256": json.loads(nested.read_text())["output_sha256"], "comparison_artifact_sha256": comparison_hashes}
    (output / "q001_map_domain_psf_manifest.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return output / "q001_map_domain_psf_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("Q001A", "Q001B"), required=True); parser.add_argument("--run-root", required=True); parser.add_argument("--prepared-root", required=True); parser.add_argument("--reference-root", required=True); parser.add_argument("--preprocessed-root", required=True); parser.add_argument("--full-input-root"); parser.add_argument("--output", required=True)
    print(run(parser.parse_args()))


if __name__ == "__main__": main()
