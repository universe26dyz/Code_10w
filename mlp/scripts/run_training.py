"""Explicit CLI for one MLP-decoder quantitative reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset
from modules.module_07_objective_training.mlp_trainer import train_mlp_reconstruction
from modules.module_08_inference_export.export_quantitative import export_quantitative_outputs
from modules.module_09_qc_benchmark.qc import validate_smoke_outputs


def run_reconstruction(config_path: str | Path, protocol_path: str | Path, observations: list[str | Path], output_dir: str | Path, mlp_checkpoint: str | Path | None = None) -> dict[str, object]:
    with Path(config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("training"), dict) or not isinstance(config.get("decoder"), dict):
        raise ValueError("--config must contain explicit training and decoder mappings.")
    if "device" not in config["training"] or "checkpoint" not in config["decoder"] or "allow_functional_fixture_checkpoint" not in config["decoder"]:
        raise ValueError("training.device and decoder checkpoint/functional-fixture allowance must be explicit.")
    if mlp_checkpoint is not None:
        config = dict(config)
        config["decoder"] = dict(config["decoder"])
        config["decoder"]["checkpoint"] = str(mlp_checkpoint)
    dataset = QuantPointDataset(observations, device=torch.device(config["training"]["device"]))
    result = train_mlp_reconstruction(dataset, config, protocol_path, output_dir)
    export_cfg = config.get("export")
    if not isinstance(export_cfg, dict) or "output_resolution_mm" not in export_cfg or "output_batch_size" not in export_cfg:
        raise ValueError("export.output_resolution_mm and export.output_batch_size must be explicit.")
    paths = export_quantitative_outputs(result["model"], result["training_space"], result["output_dir"], float(export_cfg["output_resolution_mm"]), int(export_cfg["output_batch_size"]))
    qc = validate_smoke_outputs(result["output_dir"])
    return {"outputs": {key: str(value) for key, value in paths.items()}, "qc": qc}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--protocol", required=True)
    parser.add_argument("--observations", required=True, action="append"); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_reconstruction(args.config, args.protocol, args.observations, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
