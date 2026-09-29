"""Explicit server-only Q001 FrozenMLP reconstruction launcher.

This command has no default paths and never loads B6 reconstruction state.
It loads only B6 ``resolved_config`` and constructs a fresh model/training run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from mlp.modules.module_07_objective_training.mlp_trainer import train_mlp_reconstruction
from trad.modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset
from trad.modules.module_08_inference_export.export_quantitative import export_quantitative_outputs
from trad.modules.module_08_inference_export.reprojection import export_native_plane_reprojections

from .config import build_route_config, load_b6_resolved_config
from .contracts import verify_q001_input_bundle
from .evaluation import evaluation_plan
from .routes import build_route_plan, registration_inputs_for_training


STACKS = ("sax", "2ch", "4ch")


def _paths(root: str | Path) -> list[Path]:
    root = Path(root)
    paths = [root / "CYJ" / stack / "observations.npz" for stack in STACKS]
    if not all(path.is_file() for path in paths): raise FileNotFoundError("Expected CYJ/{sax,2ch,4ch}/observations.npz.")
    return paths


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_provenance(b6_model: str | Path, signal_simulator: str | Path) -> dict[str, str]:
    return {"b6_model": str(Path(b6_model)), "b6_model_sha256": _sha256(Path(b6_model)), "frozen_signal_simulator": str(Path(signal_simulator)), "frozen_signal_simulator_sha256": _sha256(Path(signal_simulator))}


def run(args: argparse.Namespace) -> Path:
    if args.device.startswith("cpu"): raise RuntimeError("Formal Q001 reconstruction is server-GPU only; CPU execution is forbidden.")
    if not torch.cuda.is_available(): raise RuntimeError("Q001 reconstruction requires CUDA.")
    full_manifest = verify_q001_input_bundle(args.full_input_root)
    full_paths, cropped_paths = _paths(Path(args.full_input_root) / "full_fov_prepared"), _paths(args.cropped_prepared_root)
    mode = "Q001A" if args.experiment_id.startswith("Q001A_") else "Q001B" if args.experiment_id.startswith("Q001B_") else ""
    plan = build_route_plan(mode, [str(path) for path in full_paths], [str(path) for path in cropped_paths])
    # Explicit identity ordering prevents a future caller from attaching a full
    # registration stack to the wrong cropped optimization stack.
    registration_inputs = registration_inputs_for_training(plan.training_inputs, [str(path) for path in full_paths])
    b6_config = load_b6_resolved_config(args.b6_model)
    if str(b6_config.get("decoder", {}).get("checkpoint")) != str(Path(args.signal_simulator)):
        raise ValueError("B6 resolved_config decoder checkpoint does not match the approved FrozenMLP signal simulator.")
    route_record = build_route_config(b6_config, args.experiment_id, full_prepared_root=str(Path(args.full_input_root) / "full_fov_prepared"), cropped_prepared_root=str(Path(args.cropped_prepared_root)) if mode == "Q001B" else None)
    config = route_record["resolved_config"]
    config["training"] = dict(config["training"]); config["training"]["device"] = args.device
    config["stack_initialization"] = dict(config["stack_initialization"]); config["stack_initialization"]["enabled"] = True
    output = Path(args.output)
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"Refusing non-empty Q001 output root: {output}")
    dataset = QuantPointDataset(plan.training_inputs, device=torch.device(args.device))
    result = train_mlp_reconstruction(dataset, config, args.protocol, output, prepared_inputs=registration_inputs, subject_id="CYJ", command=" ".join(__import__("sys").argv))
    checkpoint = output / "model.pt"; checkpoint_hash = _sha256(checkpoint)
    export_cfg = config["export"]
    export_quantitative_outputs(result["model"], result["training_space"], output, float(export_cfg["output_resolution_mm"]), int(export_cfg["output_batch_size"]), dataset=dataset, export_config={"bbox": config.get("bbox", {}), **export_cfg})
    export_native_plane_reprojections(result["model"], result["training_space"], plan.training_inputs, output / "evaluation/mapping_central_no_psf", output_psf={"enabled": False}, export_parameter_maps=True)
    export_native_plane_reprojections(result["model"], result["training_space"], plan.training_inputs, output / "evaluation/mapping_map_psf_K32", output_psf={"enabled": True, "n_samples": 32}, evaluation_seed=20260911, export_parameter_maps=True)
    export_native_plane_reprojections(result["model"], result["training_space"], plan.training_inputs, output / "evaluation/signal_psf_K8", output_psf={"enabled": True, "n_samples": 8}, evaluation_seed=20260911, export_parameter_maps=False)
    (output / "q001_route_manifest.json").write_text(json.dumps({"experiment_id": args.experiment_id, "route": route_record["route"], "training_inputs": list(plan.training_inputs), "stack_initialization_inputs": list(registration_inputs), "stack_registration_calls": plan.stack_registration_calls, "full_input_manifest_sha256": _sha256(Path(args.full_input_root) / "q001_input_manifest.json"), "full_input_status": full_manifest["status"], **checkpoint_provenance(args.b6_model, args.signal_simulator), "reconstruction_checkpoint_sha256": checkpoint_hash, "evaluation": evaluation_plan(checkpoint_hash, mode), "b6_reconstruction_warm_start": False}, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True, choices=("Q001A_full_fov_full_recon", "Q001B_full_fov_reg_cropped_recon")); parser.add_argument("--full-input-root", required=True); parser.add_argument("--cropped-prepared-root", required=True); parser.add_argument("--b6-model", required=True); parser.add_argument("--signal-simulator", required=True); parser.add_argument("--protocol", default="trad/configs/protocol_hhz_v1.yaml"); parser.add_argument("--output", required=True); parser.add_argument("--device", default="cuda:0")
    print(run(parser.parse_args()))


if __name__ == "__main__": main()
