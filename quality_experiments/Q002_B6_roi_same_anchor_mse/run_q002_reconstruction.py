"""Server-only B6-cropped Q002 launcher; no Q001 inputs or warm-starts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from quality_experiments.Q001_recon.config import load_b6_resolved_config
from trad.modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset
from trad.modules.module_08_inference_export.export_quantitative import export_quantitative_outputs, export_quantitative_outputs_psf128
from trad.modules.module_08_inference_export.reprojection import export_native_plane_reprojections

from .contracts import Q002_EXPERIMENT_ID, Q002S640_EXPERIMENT_ID, Q003S640_EXPERIMENT_ID, build_q002_route_config, cropped_observation_paths
from .joint_dataset import JointAnchorDataset
from .training import train_q002_reconstruction


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> Path:
    if args.device.startswith("cpu") or not torch.cuda.is_available():
        raise RuntimeError("Formal Q002 reconstruction requires CUDA.")
    inputs = cropped_observation_paths(args.cropped_prepared_root)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty Q002 output root: {output}")
    config = load_b6_resolved_config(args.b6_model)
    if str(config["decoder"].get("checkpoint")) != str(Path(args.signal_simulator)):
        raise ValueError("B6 resolved_config decoder checkpoint does not match approved FrozenMLP.")
    route = build_q002_route_config(
        config,
        cropped_prepared_root=args.cropped_prepared_root,
        experiment_id=args.experiment_id,
        anchor_batch_size=args.anchor_batch_size,
        fingerprint_cosine=route["route"].get("fingerprint_cosine"),
    )
    resolved = route["resolved_config"]; resolved["training"] = dict(resolved["training"]); resolved["training"]["device"] = args.device
    scalar = QuantPointDataset(inputs, device=args.device); joint = JointAnchorDataset(inputs, scalar)
    result = train_q002_reconstruction(
        scalar,
        joint,
        resolved,
        args.protocol,
        output,
        prepared_inputs=inputs,
        anchor_batch_size=args.anchor_batch_size,
    )
    export = resolved["export"]; export_config = {"bbox": resolved.get("bbox", {}), **export}
    raw_paths = export_quantitative_outputs(result["model"], result["training_space"], output, float(export["output_resolution_mm"]), int(export["output_batch_size"]), dataset=scalar, export_config=export_config)
    required_raw = ("t1_ms", "t2_ms", "b1", "amplitude", "poses")
    if any(name not in raw_paths or not Path(raw_paths[name]).is_file() for name in required_raw):
        raise RuntimeError("Q002 raw export must write volumes and final_rigid_poses.json before PSF128 export.")
    export_quantitative_outputs_psf128(result["model"], result["training_space"], output, dataset=scalar, export_config=export_config, batch_size=int(export["output_batch_size"]), experiment_id=args.experiment_id, subject_id="CYJ", source_checkpoint=output / "model.pt", reconstruction_code_git_commit=__import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    export_native_plane_reprojections(result["model"], result["training_space"], inputs, output / "evaluation/mapping_central_no_psf", output_psf={"enabled": False}, export_parameter_maps=True)
    export_native_plane_reprojections(result["model"], result["training_space"], inputs, output / "evaluation/signal_psf_K8", output_psf={"enabled": True, "n_samples": 8}, evaluation_seed=20260911, export_parameter_maps=False)
    git_sha = __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    common = {"experiment_id": args.experiment_id, "subject_id": "CYJ", "route": route["route"], "reconstruction_git_sha": git_sha, "git_dirty": bool(__import__("subprocess").check_output(["git", "status", "--porcelain"], text=True).strip()), "B6_model_path": str(Path(args.b6_model)), "b6_model_sha256": _sha256(Path(args.b6_model)), "approved_decoder_path": str(Path(args.signal_simulator)), "approved_decoder_sha256": _sha256(Path(args.signal_simulator)), "prepared_inputs_sha256": {str(path): _sha256(path) for path in inputs}, "joint_dataset_qc": joint.qc, "controls": result["controls"], "intensity_normalization": result["intensity_normalization"], "stack_weights": result["stack_weights"], "fixed_monitor": {"seed": result["fixed_monitor"]["seed"], "anchor_identity_sha256": result["fixed_monitor"]["anchor_identity_sha256"]}, "q002_checkpoint_sha256": _sha256(output / "model.pt"), "psf128_contract": {"resolution_mm": 1.0, "factor": 1.0, "samples": 128, "seed": 20260911}}
    if args.experiment_id == Q003S640_EXPERIMENT_ID:
        common.update({"parent_experiment": Q002S640_EXPERIMENT_ID, "scientific_change": "add_fingerprint_cosine_only", "checkpoint_sha256": common["q002_checkpoint_sha256"], "anchor_batch_size": 640, "weights_per_anchor": 10, "signal_residual_count": 6400, "training_psf_samples": 8, "data_psf_inr_location_count": 5120, "stage_a_iterations": 2000, "stage_b_iterations": 4000, "seed": 20260911, "regularization_candidate_count": 640, "regularization_effective_point_count": 256, "regularization_sampling_source": "B6 scalar cropped support", "fingerprint_cosine_enabled": True, "fingerprint_cosine_weight": 1.0, "fingerprint_cosine_epsilon": 1.0e-8})
    (output / "q002_route_manifest.json").write_text(json.dumps(common, indent=2) + "\n")
    (output / "experiment_manifest.json").write_text(json.dumps(common, indent=2) + "\n")
    (output / "RESULT_SUMMARY.md").write_text(f"# {args.experiment_id} B6 ROI same-anchor MSE\n\nCode run completed; formal evaluation must be invoked separately.\n", encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cropped-prepared-root", required=True); parser.add_argument("--b6-model", required=True); parser.add_argument("--signal-simulator", required=True); parser.add_argument("--protocol", default="trad/configs/protocol_hhz_v1.yaml"); parser.add_argument("--output", required=True); parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--experiment-id", choices=(Q002_EXPERIMENT_ID, Q002S640_EXPERIMENT_ID, Q003S640_EXPERIMENT_ID), default=Q002_EXPERIMENT_ID)
    parser.add_argument("--anchor-batch-size", type=int, default=64)
    return parser


def main() -> None:
    print(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
