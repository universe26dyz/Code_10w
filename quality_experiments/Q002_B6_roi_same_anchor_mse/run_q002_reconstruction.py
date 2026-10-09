"""Server-only B6-cropped Q002 launcher; no Q001 inputs or warm-starts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from mlp.modules.module_07_objective_training.mlp_trainer import build_mlp_training_model
from quality_experiments.Q001_recon.config import load_b6_resolved_config
from reconstruction_core.orchestration import load_checkpoint
from trad.modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset
from trad.modules.module_08_inference_export.export_quantitative import export_quantitative_outputs, export_quantitative_outputs_psf128
from trad.modules.module_08_inference_export.reprojection import export_native_plane_reprojections

from .contracts import Q002_EXPERIMENT_ID, Q002S640_EXPERIMENT_ID, Q003S640_EXPERIMENT_ID, Q003S640_10K_EXPERIMENT_ID, Q004S640_K1_EXPERIMENT_ID, build_q002_route_config, cropped_observation_paths
from .joint_dataset import JointAnchorDataset
from .training import train_q002_reconstruction
from .training_curves import generate_convergence_artifacts, generate_q004_convergence_artifacts


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
    )
    fingerprint_cosine = route["route"].get("fingerprint_cosine")
    training_psf_samples = int(route["route"]["sampling"]["shared_training_psf_samples"])
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
        fingerprint_cosine=fingerprint_cosine,
        continuation=route["route"].get("continuation"),
        training_psf_samples=training_psf_samples,
    )
    export = resolved["export"]; export_config = {"bbox": resolved.get("bbox", {}), **export}
    raw_paths = export_quantitative_outputs(result["model"], result["training_space"], output, float(export["output_resolution_mm"]), int(export["output_batch_size"]), dataset=scalar, export_config=export_config)
    required_raw = ("t1_ms", "t2_ms", "b1", "amplitude", "poses")
    if any(name not in raw_paths or not Path(raw_paths[name]).is_file() for name in required_raw):
        raise RuntimeError("Q002 raw export must write volumes and final_rigid_poses.json before PSF128 export.")
    git_sha = __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if args.experiment_id == Q003S640_10K_EXPERIMENT_ID:
        primary_checkpoint = result["primary_checkpoint"]
        if primary_checkpoint is None:
            raise RuntimeError("Q003-10k training did not produce its primary 6000-step checkpoint.")
        primary_model, primary_space, _, _ = build_mlp_training_model(scalar, result["resolved_config"], args.protocol, torch.device(args.device))
        load_checkpoint(primary_checkpoint, primary_model, torch.device(args.device))
        export_native_plane_reprojections(primary_model, primary_space, inputs, output / "evaluation/checkpoint_6000/mapping_central_no_psf", output_psf={"enabled": False}, export_parameter_maps=True)
        export_native_plane_reprojections(primary_model, primary_space, inputs, output / "evaluation/checkpoint_6000/signal_psf_K8", output_psf={"enabled": True, "n_samples": 8}, evaluation_seed=20260911, export_parameter_maps=False)
        export_native_plane_reprojections(result["model"], result["training_space"], inputs, output / "evaluation/checkpoint_10000/mapping_central_no_psf", output_psf={"enabled": False}, export_parameter_maps=True)
        export_native_plane_reprojections(result["model"], result["training_space"], inputs, output / "evaluation/checkpoint_10000/signal_psf_K8", output_psf={"enabled": True, "n_samples": 8}, evaluation_seed=20260911, export_parameter_maps=False)
        generate_convergence_artifacts(output / "training_log.csv", output / "monitor_log.csv", output / "training_curves")
    elif args.experiment_id == Q004S640_K1_EXPERIMENT_ID:
        export_native_plane_reprojections(result["model"], result["training_space"], inputs, output / "evaluation/mapping_central_no_psf", output_psf={"enabled": False}, export_parameter_maps=True)
        export_native_plane_reprojections(result["model"], result["training_space"], inputs, output / "evaluation/signal_central_K1", output_psf={"enabled": True, "n_samples": 1}, export_parameter_maps=False)
        export_native_plane_reprojections(result["model"], result["training_space"], inputs, output / "evaluation/signal_psf_K8", output_psf={"enabled": True, "n_samples": 8}, evaluation_seed=20260911, export_parameter_maps=False)
        generate_q004_convergence_artifacts(output / "training_log.csv", output / "monitor_log.csv", output / "training_curves")
    else:
        export_native_plane_reprojections(result["model"], result["training_space"], inputs, output / "evaluation/mapping_central_no_psf", output_psf={"enabled": False}, export_parameter_maps=True)
        export_native_plane_reprojections(result["model"], result["training_space"], inputs, output / "evaluation/signal_psf_K8", output_psf={"enabled": True, "n_samples": 8}, evaluation_seed=20260911, export_parameter_maps=False)
    export_quantitative_outputs_psf128(result["model"], result["training_space"], output, dataset=scalar, export_config=export_config, batch_size=int(export["output_batch_size"]), experiment_id=args.experiment_id, subject_id="CYJ", source_checkpoint=output / "model.pt", reconstruction_code_git_commit=git_sha)
    common = {"experiment_id": args.experiment_id, "subject_id": "CYJ", "route": route["route"], "reconstruction_git_sha": git_sha, "git_dirty": bool(__import__("subprocess").check_output(["git", "status", "--porcelain"], text=True).strip()), "B6_model_path": str(Path(args.b6_model)), "b6_model_sha256": _sha256(Path(args.b6_model)), "approved_decoder_path": str(Path(args.signal_simulator)), "approved_decoder_sha256": _sha256(Path(args.signal_simulator)), "prepared_inputs_sha256": {str(path): _sha256(path) for path in inputs}, "joint_dataset_qc": joint.qc, "controls": result["controls"], "intensity_normalization": result["intensity_normalization"], "stack_weights": result["stack_weights"], "fixed_monitor": {"seed": result["fixed_monitor"]["seed"], "anchor_identity_sha256": result["fixed_monitor"]["anchor_identity_sha256"]}, "q002_checkpoint_sha256": _sha256(output / "model.pt"), "psf128_contract": {"resolution_mm": 1.0, "factor": 1.0, "samples": 128, "seed": 20260911}}
    if args.experiment_id == Q003S640_EXPERIMENT_ID:
        common.update({"parent_experiment": Q002S640_EXPERIMENT_ID, "scientific_change": "add_fingerprint_cosine_only", "checkpoint_sha256": common["q002_checkpoint_sha256"], "anchor_batch_size": 640, "weights_per_anchor": 10, "signal_residual_count": 6400, "training_psf_samples": 8, "data_psf_inr_location_count": 5120, "stage_a_iterations": 2000, "stage_b_iterations": 4000, "seed": 20260911, "regularization_candidate_count": 640, "regularization_effective_point_count": 256, "regularization_sampling_source": "B6 scalar cropped support", "fingerprint_cosine_enabled": True, "fingerprint_cosine_weight": 1.0, "fingerprint_cosine_epsilon": 1.0e-8})
    if args.experiment_id == Q003S640_10K_EXPERIMENT_ID:
        primary_checkpoint = Path(result["primary_checkpoint"])
        common.update({"role": "extended_optimization_secondary", "parent_experiment": Q002S640_EXPERIMENT_ID, "scientific_change": "add_fingerprint_cosine_only_for_primary_6k", "secondary_change": "continue_same_q003_optimization_to_10k", "anchor_batch_size": 640, "weights_per_anchor": 10, "signal_residual_count": 6400, "training_psf_samples": 8, "data_psf_inr_location_count": 5120, "stage_a_iterations": 2000, "stage_b_standard_iterations": 4000, "stage_b_extension_iterations": 4000, "primary_checkpoint_iteration": 6000, "final_checkpoint_iteration": 10000, "global_iteration": result["global_iteration"], "seed": 20260911, "regularization_candidate_count": 640, "regularization_effective_point_count": 256, "regularization_sampling_source": "B6 scalar cropped support", "regularization_application_count_per_optimizer_step": 1, "fingerprint_cosine_enabled": True, "fingerprint_cosine_weight": 1.0, "fingerprint_cosine_epsilon": 1.0e-8, "primary_checkpoint_6000_path": str(primary_checkpoint.relative_to(output)), "primary_checkpoint_6000_sha256": _sha256(primary_checkpoint), "final_checkpoint_10000_path": "model.pt", "final_checkpoint_10000_sha256": common["q002_checkpoint_sha256"], "checkpoint_sha256": common["q002_checkpoint_sha256"], "extension_lr_policy": route["route"]["continuation"]["extension_lr_policy"]})
    if args.experiment_id == Q004S640_K1_EXPERIMENT_ID:
        common.update({"parent_experiment": Q002S640_EXPERIMENT_ID, "scientific_change": "training_psf_samples_8_to_central_1_only", "loss": "MSE_only", "fingerprint_cosine_enabled": False, "anchor_batch_size": 640, "weights_per_anchor": 10, "signal_residual_count": 6400, "training_psf_samples": 1, "training_psf_semantics": "central_local_coordinate_no_gaussian_draw", "data_psf_inr_location_count": 640, "stage_a_iterations": 2000, "stage_b_iterations": 4000, "seed": 20260911, "regularization_candidate_count": 640, "regularization_effective_point_count": 256, "regularization_sampling_source": "B6 scalar cropped support", "regularization_application_count_per_optimizer_step": 1, "checkpoint_path": "model.pt", "checkpoint_sha256": common["q002_checkpoint_sha256"], "k8_posthoc_signal_contract": {"n_samples": 8, "seed": 20260911}, "k1_central_signal_contract": {"n_samples": 1, "semantics": "central_local_coordinate_no_gaussian_draw"}})
    (output / "q002_route_manifest.json").write_text(json.dumps(common, indent=2) + "\n")
    (output / "experiment_manifest.json").write_text(json.dumps(common, indent=2) + "\n")
    summary = f"# {args.experiment_id} B6 ROI same-anchor MSE\n\nCode run completed; formal evaluation must be invoked separately.\n"
    if args.experiment_id == Q004S640_K1_EXPERIMENT_ID:
        summary += """
## Pre-registered interpretation logic

- A: K1 maps and central-K1 signal improve while K8 signal worsens/stays similar: PSF-realism versus central-parameter-identifiability tradeoff.
- B: maps, central-K1 signal, and K8 signal all improve: K8 training mixture was unnecessary or harmful under this data/geometry.
- C: central-K1 signal improves without T1/T2-map improvement: investigate amplitude, B1, pose, or intrinsic sequence conditioning.
- D: maps and signals worsen: K8 provides useful physical forward-model information.

These are mechanistic interpretations, not population-level claims.
"""
    (output / "RESULT_SUMMARY.md").write_text(summary, encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cropped-prepared-root", required=True); parser.add_argument("--b6-model", required=True); parser.add_argument("--signal-simulator", required=True); parser.add_argument("--protocol", default="trad/configs/protocol_hhz_v1.yaml"); parser.add_argument("--output", required=True); parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--experiment-id", choices=(Q002_EXPERIMENT_ID, Q002S640_EXPERIMENT_ID, Q003S640_EXPERIMENT_ID, Q003S640_10K_EXPERIMENT_ID, Q004S640_K1_EXPERIMENT_ID), default=Q002_EXPERIMENT_ID)
    parser.add_argument("--anchor-batch-size", type=int, default=64)
    return parser


def main() -> None:
    print(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
