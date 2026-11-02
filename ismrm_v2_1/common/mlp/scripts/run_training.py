"""Explicit CLI for one MLP-decoder quantitative reconstruction."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from mlp.modules.module_07_objective_training.mlp_trainer import train_mlp_reconstruction
from reconstruction_core.orchestration import finalize_timing_profile
from trad.modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset
from trad.modules.module_08_inference_export.export_quantitative import export_quantitative_outputs
from trad.modules.module_08_inference_export.reprojection import export_native_plane_reprojections
from trad.modules.module_09_qc_benchmark.qc import validate_smoke_outputs
from trad.scripts.run_training import load_training_config, reject_blocked_experiment


def controlled_config_without_decoder(config: dict) -> dict:
    """Remove decoder and output-identifying metadata for scientific parity checks."""

    controlled = copy.deepcopy(config)
    controlled.pop("decoder", None)
    controlled.pop("experiment", None)
    return controlled


def load_controlled_config(config_path: str | Path, mlp_checkpoint: str | Path | None) -> dict:
    """Resolve the B6-derived template; only an explicit checkpoint unlocks it."""

    config = load_training_config(config_path)
    experiment = config.get("experiment")
    if isinstance(experiment, dict) and experiment.get("blocked") is True:
        if mlp_checkpoint is None:
            reject_blocked_experiment(config)
        config = copy.deepcopy(config)
        config["experiment"]["blocked"] = False
    if mlp_checkpoint is not None:
        config = copy.deepcopy(config)
        config["decoder"] = dict(config.get("decoder", {}))
        config["decoder"]["checkpoint"] = str(mlp_checkpoint)
    return config


def run_reconstruction(config_path: str | Path, protocol_path: str | Path, observations: list[str | Path], output_dir: str | Path, mlp_checkpoint: str | Path | None = None, subject_id: str | None = None) -> dict[str, object]:
    run_started = time.perf_counter()
    config = load_controlled_config(config_path, mlp_checkpoint)
    if not isinstance(config, dict) or not isinstance(config.get("training"), dict) or not isinstance(config.get("decoder"), dict):
        raise ValueError("--config must contain explicit training and decoder mappings.")
    if "device" not in config["training"] or "checkpoint" not in config["decoder"] or "allow_functional_fixture_checkpoint" not in config["decoder"]:
        raise ValueError("training.device and decoder checkpoint/functional-fixture allowance must be explicit.")
    data_started = time.perf_counter()
    dataset = QuantPointDataset(observations, device=torch.device(config["training"]["device"]))
    data_loading_ms = (time.perf_counter() - data_started) * 1000.0
    result = train_mlp_reconstruction(dataset, config, protocol_path, output_dir, prepared_inputs=observations, subject_id=subject_id or Path(observations[0]).parents[1].name, command=" ".join(__import__("sys").argv))
    export_cfg = config.get("export")
    if not isinstance(export_cfg, dict) or "output_resolution_mm" not in export_cfg or "output_batch_size" not in export_cfg:
        raise ValueError("export.output_resolution_mm and export.output_batch_size must be explicit.")
    export_started = time.perf_counter()
    paths = export_quantitative_outputs(result["model"], result["training_space"], result["output_dir"], float(export_cfg["output_resolution_mm"]), int(export_cfg["output_batch_size"]), dataset=dataset, export_config={"bbox": config.get("bbox", {}), **export_cfg})
    paths.update(export_native_plane_reprojections(result["model"], result["training_space"], observations, result["output_dir"], output_psf=config.get("psf", {}).get("export", {})))
    export_ms = (time.perf_counter() - export_started) * 1000.0
    validation_started = time.perf_counter()
    qc = validate_smoke_outputs(result["output_dir"])
    run_level_ms = dict(result["run_level_ms"])
    run_level_ms.update({"data_loading": data_loading_ms, "export": export_ms, "validation": (time.perf_counter() - validation_started) * 1000.0, "total_runtime": (time.perf_counter() - run_started) * 1000.0})
    summary = finalize_timing_profile(result["output_dir"] / "timing_profile.csv", stage_iterations={"A": int(config["training"]["stage_a_iterations"]), "B": int(config["training"]["stage_b_iterations"])}, decoder_type=str(result["decoder_metadata"]["decoder_type"]), run_level_ms=run_level_ms)
    return {"outputs": {key: str(value) for key, value in paths.items()}, "qc": qc, "timing_profile_summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--protocol", required=True)
    parser.add_argument("--observations", required=True, action="append"); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mlp-checkpoint", required=True)
    args = parser.parse_args()
    print(json.dumps(run_reconstruction(args.config, args.protocol, args.observations, args.output_dir, mlp_checkpoint=args.mlp_checkpoint), sort_keys=True))


if __name__ == "__main__":
    main()
