"""Server-only B6-cropped Q002 launcher; no Q001 inputs or warm-starts."""

from __future__ import annotations

import argparse
import copy
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

from .contracts import Q002_EXPERIMENT_ID, Q002S640_EXPERIMENT_ID, Q002S640_K256_EXPERIMENT_ID, Q003S640_EXPERIMENT_ID, Q003S640_10K_EXPERIMENT_ID, Q004S640_K1_EXPERIMENT_ID, Q005S640_K4_EXPERIMENT_ID, build_q002_route_config, cropped_observation_paths, profiled_observation_paths
from .joint_dataset import JointAnchorDataset
from .training import train_q002_reconstruction
from .training_curves import generate_convergence_artifacts, generate_q004_convergence_artifacts
from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import APPROVED_DECODER_SHA256, B6_CONFIG_SOURCE_SHA256
from paper_experiments.joint_vs_2dfit.phase2_clean.sparse_profile_manifest import validate_sparse_profile_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sparse_profile_provenance(manifest: dict[str, object], inputs: list[Path]) -> dict[str, object]:
    """Return the immutable sparse identity carried into every Q002 endpoint."""
    return {
        "profile_id": manifest["profile_id"],
        "case_id": manifest["case_id"],
        "sparse_manifest": {
            "path": manifest["_manifest_path"],
            "sha256": manifest["_manifest_sha256"],
            "generator_git_sha": manifest["git_sha"],
            "source_prepared_root": manifest["source_prepared_root"],
            "prepared_root": manifest["prepared_root"],
        },
        "training_input_sha256": {str(path): _sha256(path) for path in inputs},
    }


FORMAL_SPARSE_EXPERIMENT_IDS = (Q002S640_EXPERIMENT_ID, Q004S640_K1_EXPERIMENT_ID, Q005S640_K4_EXPERIMENT_ID)


def validate_formal_sparse_reconstruction_dependencies(args: argparse.Namespace) -> dict[str, object] | None:
    """Validate immutable sparse inputs for the formal Q002S640, Q004S640-K1, and Q005S640-K4 routes."""
    if getattr(args, "experiment_id", None) not in FORMAL_SPARSE_EXPERIMENT_IDS:
        return None
    if not getattr(args, "sparse_manifest", None):
        raise ValueError("formal sparse execution requires an exact sparse manifest")
    expected = {
        "generator": getattr(args, "expected_sparse_generator_git_sha", None),
        "manifest": getattr(args, "expected_sparse_manifest_sha256", None),
        "B6 model": getattr(args, "expected_b6_model_sha256", None),
        "protocol": getattr(args, "expected_protocol_sha256", None),
    }
    missing = [label for label, value in expected.items() if value is None]
    if missing:
        raise ValueError("formal sparse execution requires explicit pins for " + ", ".join(missing))
    for label, value in expected.items():
        width = 40 if label == "generator" else 64
        if not isinstance(value, str) or len(value) != width or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"formal sparse execution has an invalid {label} pin")
    b6 = Path(args.b6_model)
    decoder = Path(args.signal_simulator)
    protocol = Path(args.protocol)
    if not b6.is_file() or _sha256(b6) != B6_CONFIG_SOURCE_SHA256 or expected["B6 model"] != B6_CONFIG_SOURCE_SHA256:
        raise ValueError("formal sparse execution B6 model SHA256 disagrees with the frozen contract")
    if not decoder.is_file() or _sha256(decoder) != APPROVED_DECODER_SHA256:
        raise ValueError("formal sparse execution decoder SHA256 is not the approved FrozenMLP")
    if not protocol.is_file() or _sha256(protocol) != expected["protocol"]:
        raise ValueError("formal sparse execution protocol SHA256 disagrees with the explicit admission pin")
    profile = validate_sparse_profile_manifest(
        args.sparse_manifest,
        subject=args.subject_id,
        strict=True,
        expected_generator_git_sha=expected["generator"],
        expected_manifest_sha256=expected["manifest"],
    )
    if profile.get("subject") != args.subject_id or Path(str(profile.get("prepared_root", ""))).resolve() != Path(args.cropped_prepared_root).resolve():
        raise ValueError("formal sparse execution subject or prepared root disagrees with the sparse manifest")
    if profile.get("_manifest_sha256") != expected["manifest"]:
        raise ValueError("formal sparse execution manifest SHA256 disagrees with the validated manifest")
    if Path(args.output).exists():
        raise FileExistsError("formal sparse execution requires a fresh output root")
    return profile


def validate_formal_q002s640_dependencies(args: argparse.Namespace) -> None:
    """Backward-compatible entry point for the original formal Q002S640 contract."""
    if hasattr(args, "subject_id") and hasattr(args, "cropped_prepared_root") and hasattr(args, "output"):
        validate_formal_sparse_reconstruction_dependencies(args)
        return
    if not (getattr(args, "sparse_manifest", None) and args.experiment_id == Q002S640_EXPERIMENT_ID):
        return
    expected = {"generator": getattr(args, "expected_sparse_generator_git_sha", None), "manifest": getattr(args, "expected_sparse_manifest_sha256", None), "B6 model": getattr(args, "expected_b6_model_sha256", None), "protocol": getattr(args, "expected_protocol_sha256", None)}
    missing = [label for label, value in expected.items() if value is None]
    if missing:
        raise ValueError("Q002S640 sparse execution requires explicit pins for " + ", ".join(missing))
    for label, value in expected.items():
        width = 40 if label == "generator" else 64
        if not isinstance(value, str) or len(value) != width or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"Q002S640 sparse execution has an invalid {label} pin")
    b6, decoder, protocol = Path(args.b6_model), Path(args.signal_simulator), Path(args.protocol)
    if not b6.is_file() or _sha256(b6) != B6_CONFIG_SOURCE_SHA256 or expected["B6 model"] != B6_CONFIG_SOURCE_SHA256:
        raise ValueError("Q002S640 sparse execution B6 model SHA256 disagrees with the frozen contract")
    if not decoder.is_file() or _sha256(decoder) != APPROVED_DECODER_SHA256:
        raise ValueError("Q002S640 sparse execution decoder SHA256 is not the approved FrozenMLP")
    if not protocol.is_file() or _sha256(protocol) != expected["protocol"]:
        raise ValueError("Q002S640 sparse execution protocol SHA256 disagrees with the explicit admission pin")


def run(args: argparse.Namespace) -> Path:
    endpoint_policy = getattr(args, "endpoint_policy", "legacy")
    if endpoint_policy not in ("legacy", "central_only"):
        raise ValueError("endpoint_policy must be 'legacy' or 'central_only'")
    if endpoint_policy == "central_only" and args.experiment_id != Q002S640_EXPERIMENT_ID:
        raise ValueError("central_only endpoint policy is only valid for the Q002S640/K8 route")
    formal_profile = validate_formal_sparse_reconstruction_dependencies(args)
    sparse_manifest = getattr(args, "sparse_manifest", None)
    expected_sparse_generator_sha = getattr(args, "expected_sparse_generator_git_sha", None)
    expected_sparse_manifest_sha = getattr(args, "expected_sparse_manifest_sha256", None)
    if args.device.startswith("cpu") or not torch.cuda.is_available():
        raise RuntimeError("Formal Q002 reconstruction requires CUDA.")
    sparse_profile = formal_profile or (validate_sparse_profile_manifest(sparse_manifest, subject=args.subject_id, strict=expected_sparse_generator_sha is not None, expected_generator_git_sha=expected_sparse_generator_sha, expected_manifest_sha256=expected_sparse_manifest_sha) if sparse_manifest else None)
    inputs = profiled_observation_paths(args.cropped_prepared_root, args.subject_id, sparse_manifest) if sparse_manifest else cropped_observation_paths(args.cropped_prepared_root, args.subject_id)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty Q002 output root: {output}")
    config = load_b6_resolved_config(args.b6_model)
    source_decoder_path = str(config["decoder"].get("checkpoint"))
    if _sha256(Path(args.signal_simulator)) != APPROVED_DECODER_SHA256:
        raise ValueError("runtime decoder SHA256 is not the approved FrozenMLP.")
    config = copy.deepcopy(config); config["decoder"] = dict(config["decoder"]); config["decoder"]["checkpoint"] = str(Path(args.signal_simulator))
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
        raise RuntimeError("Q002 raw export must write volumes and final_rigid_poses.json before endpoint-specific exports.")
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
    elif args.experiment_id == Q002S640_K256_EXPERIMENT_ID:
        # Formal endpoints are the raw central maps and post-hoc map-domain PSF128 below.
        export_native_plane_reprojections(result["model"], result["training_space"], inputs, output / "evaluation/mapping_central_no_psf", output_psf={"enabled": False}, export_parameter_maps=True)
    else:
        export_native_plane_reprojections(result["model"], result["training_space"], inputs, output / "evaluation/mapping_central_no_psf", output_psf={"enabled": False}, export_parameter_maps=True)
        if endpoint_policy == "legacy":
            export_native_plane_reprojections(result["model"], result["training_space"], inputs, output / "evaluation/signal_psf_K8", output_psf={"enabled": True, "n_samples": 8}, evaluation_seed=20260911, export_parameter_maps=False)
    if endpoint_policy == "legacy":
        export_quantitative_outputs_psf128(result["model"], result["training_space"], output, dataset=scalar, export_config=export_config, batch_size=int(export["output_batch_size"]), experiment_id=args.experiment_id, subject_id=args.subject_id, source_checkpoint=output / "model.pt", reconstruction_code_git_commit=git_sha)
    protocol_path = Path(args.protocol)
    protocol_provenance = {"protocol_path": str(protocol_path), "protocol_sha256": _sha256(protocol_path)} if protocol_path.is_file() else {}
    common = {"experiment_id": args.experiment_id, "subject_id": args.subject_id, "route": route["route"], "reconstruction_git_sha": git_sha, "git_dirty": bool(__import__("subprocess").check_output(["git", "status", "--porcelain"], text=True).strip()), "B6_model_path": str(Path(args.b6_model)), "b6_model_sha256": _sha256(Path(args.b6_model)), "decoder_relocation":{"b6_resolved_config_path":source_decoder_path,"runtime_decoder_path":str(Path(args.signal_simulator)),"approved_decoder_sha256":_sha256(Path(args.signal_simulator)),"scientific_config_changed":False}, "approved_decoder_path": str(Path(args.signal_simulator)), "approved_decoder_sha256": _sha256(Path(args.signal_simulator)), "dependency_sha256": {"B6_model": _sha256(Path(args.b6_model)), "frozen_decoder": _sha256(Path(args.signal_simulator)), **({"protocol": _sha256(protocol_path)} if protocol_path.is_file() else {})}, **protocol_provenance, "prepared_inputs_sha256": {str(path): _sha256(path) for path in inputs}, "joint_dataset_qc": joint.qc, "controls": result["controls"], "intensity_normalization": result["intensity_normalization"], "stack_weights": result["stack_weights"], "fixed_monitor": {"seed": result["fixed_monitor"]["seed"], "anchor_identity_sha256": result["fixed_monitor"]["anchor_identity_sha256"]}, "q002_checkpoint_sha256": _sha256(output / "model.pt"), "psf128_contract": {"resolution_mm": 1.0, "factor": 1.0, "samples": 128, "seed": 20260911}}
    if sparse_profile is not None:
        common.update(sparse_profile_provenance(sparse_profile, inputs))
    if args.experiment_id == Q003S640_EXPERIMENT_ID:
        common.update({"parent_experiment": Q002S640_EXPERIMENT_ID, "scientific_change": "add_fingerprint_cosine_only", "checkpoint_sha256": common["q002_checkpoint_sha256"], "anchor_batch_size": 640, "weights_per_anchor": 10, "signal_residual_count": 6400, "training_psf_samples": 8, "data_psf_inr_location_count": 5120, "stage_a_iterations": 2000, "stage_b_iterations": 4000, "seed": 20260911, "regularization_candidate_count": 640, "regularization_effective_point_count": 256, "regularization_sampling_source": "B6 scalar cropped support", "fingerprint_cosine_enabled": True, "fingerprint_cosine_weight": 1.0, "fingerprint_cosine_epsilon": 1.0e-8})
    if args.experiment_id == Q003S640_10K_EXPERIMENT_ID:
        primary_checkpoint = Path(result["primary_checkpoint"])
        common.update({"role": "extended_optimization_secondary", "parent_experiment": Q002S640_EXPERIMENT_ID, "scientific_change": "add_fingerprint_cosine_only_for_primary_6k", "secondary_change": "continue_same_q003_optimization_to_10k", "anchor_batch_size": 640, "weights_per_anchor": 10, "signal_residual_count": 6400, "training_psf_samples": 8, "data_psf_inr_location_count": 5120, "stage_a_iterations": 2000, "stage_b_standard_iterations": 4000, "stage_b_extension_iterations": 4000, "primary_checkpoint_iteration": 6000, "final_checkpoint_iteration": 10000, "global_iteration": result["global_iteration"], "seed": 20260911, "regularization_candidate_count": 640, "regularization_effective_point_count": 256, "regularization_sampling_source": "B6 scalar cropped support", "regularization_application_count_per_optimizer_step": 1, "fingerprint_cosine_enabled": True, "fingerprint_cosine_weight": 1.0, "fingerprint_cosine_epsilon": 1.0e-8, "primary_checkpoint_6000_path": str(primary_checkpoint.relative_to(output)), "primary_checkpoint_6000_sha256": _sha256(primary_checkpoint), "final_checkpoint_10000_path": "model.pt", "final_checkpoint_10000_sha256": common["q002_checkpoint_sha256"], "checkpoint_sha256": common["q002_checkpoint_sha256"], "extension_lr_policy": route["route"]["continuation"]["extension_lr_policy"]})
    if args.experiment_id == Q004S640_K1_EXPERIMENT_ID:
        common.update({"parent_experiment": Q002S640_EXPERIMENT_ID, "scientific_change": "training_psf_samples_8_to_central_1_only", "loss": "MSE_only", "fingerprint_cosine_enabled": False, "anchor_batch_size": 640, "weights_per_anchor": 10, "signal_residual_count": 6400, "training_psf_samples": 1, "training_psf_semantics": "central_local_coordinate_no_gaussian_draw", "data_psf_inr_location_count": 640, "stage_a_iterations": 2000, "stage_b_iterations": 4000, "seed": 20260911, "regularization_candidate_count": 640, "regularization_effective_point_count": 256, "regularization_sampling_source": "B6 scalar cropped support", "regularization_application_count_per_optimizer_step": 1, "checkpoint_path": "model.pt", "checkpoint_sha256": common["q002_checkpoint_sha256"], "k8_posthoc_signal_contract": {"n_samples": 8, "seed": 20260911}, "k1_central_signal_contract": {"n_samples": 1, "semantics": "central_local_coordinate_no_gaussian_draw"}})
    if args.experiment_id == Q005S640_K4_EXPERIMENT_ID:
        common.update({"parent_experiment": Q002S640_EXPERIMENT_ID, "scientific_change": "training_psf_samples_8_to_4_only", "loss": "MSE_only", "fingerprint_cosine_enabled": False, "anchor_batch_size": 640, "weights_per_anchor": 10, "signal_residual_count": 6400, "training_psf_samples": 4, "data_psf_inr_location_count": 2560, "psf_forward_chunk_size": 4, "stage_a_iterations": 2000, "stage_b_iterations": 4000, "seed": 20260911, "regularization_candidate_count": 640, "regularization_effective_point_count": 256, "regularization_sampling_source": "B6 scalar cropped support", "regularization_application_count_per_optimizer_step": 1, "endpoint_semantics": route["route"]["endpoint_semantics"]})
    if args.experiment_id == Q002S640_K256_EXPERIMENT_ID:
        common.update({"parent_experiment": Q002S640_EXPERIMENT_ID, "scientific_change": "training_psf_samples_8_to_256_only", "loss": "MSE_only", "fingerprint_cosine_enabled": False, "anchor_batch_size": 640, "weights_per_anchor": 10, "signal_residual_count": 6400, "training_psf_samples": 256, "training_psf_semantics": "same-anchor local PSF neighbourhood samples averaged before the 10-weight joint-vector MSE", "data_psf_inr_location_count": 163840, "psf_forward_chunk_size": 32, "stage_a_iterations": 2000, "stage_b_iterations": 4000, "seed": 20260911, "regularization_candidate_count": 640, "regularization_effective_point_count": 256, "regularization_sampling_source": "B6 scalar cropped support", "regularization_application_count_per_optimizer_step": 1, "checkpoint_path": "model.pt", "checkpoint_sha256": common["q002_checkpoint_sha256"], "fit_first_n_samples_semantics": route["route"]["fit_first_n_samples_semantics"], "endpoint_semantics": {"central": "central/no map-domain PSF post-hoc endpoint", "psf128": "post-hoc map-domain PSF128 endpoint, distinct from training K256"}})
    common.update({
        "endpoint_policy": endpoint_policy,
        "formal_endpoint": "central" if endpoint_policy == "central_only" else "central_and_posthoc_psf128",
        "training_psf_samples": int(route["route"]["sampling"]["shared_training_psf_samples"]),
        "central_output_sampling_psf_samples": 1,
        "central_output_semantics": "local central coordinate; no post-hoc map-domain PSF",
        "signal_psf_k8_generated": endpoint_policy == "legacy" and args.experiment_id != Q002S640_K256_EXPERIMENT_ID,
        "psf128_export_generated": endpoint_policy == "legacy",
    })
    if endpoint_policy == "central_only":
        common.pop("psf128_contract")
        common["psf128_export"] = {"generated": False, "formal_endpoint": False}
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
    parser.add_argument("--cropped-prepared-root", required=True); parser.add_argument("--subject-id", default="CYJ"); parser.add_argument("--sparse-manifest"); parser.add_argument("--expected-sparse-generator-git-sha"); parser.add_argument("--expected-sparse-manifest-sha256"); parser.add_argument("--b6-model", required=True); parser.add_argument("--expected-b6-model-sha256"); parser.add_argument("--signal-simulator", required=True); parser.add_argument("--protocol", default="trad/configs/protocol_hhz_v1.yaml"); parser.add_argument("--expected-protocol-sha256"); parser.add_argument("--output", required=True); parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--experiment-id", choices=(Q002_EXPERIMENT_ID, Q002S640_EXPERIMENT_ID, Q002S640_K256_EXPERIMENT_ID, Q003S640_EXPERIMENT_ID, Q003S640_10K_EXPERIMENT_ID, Q004S640_K1_EXPERIMENT_ID, Q005S640_K4_EXPERIMENT_ID), default=Q002_EXPERIMENT_ID)
    parser.add_argument("--anchor-batch-size", type=int, default=64)
    parser.add_argument("--endpoint-policy", choices=("legacy", "central_only"), default="legacy")
    return parser


def main() -> None:
    print(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
