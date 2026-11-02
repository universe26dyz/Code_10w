"""Checkpoint-only Trad export using explicit root-derived CLI paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset
from modules.module_07_objective_training.trad_trainer import build_training_model, load_checkpoint
from modules.module_07_objective_training.training_space import TrainingSpace
from modules.module_08_inference_export.export_quantitative import export_quantitative_outputs
from modules.module_08_inference_export.reprojection import export_native_plane_reprojections


def export_checkpoint(config_path: str | Path, protocol_path: str | Path, checkpoint_path: str | Path, observations: list[str | Path], output_dir: str | Path) -> dict[str, str]:
    with Path(config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("training"), dict):
        raise ValueError("--config must contain a training mapping.")
    device = torch.device(config["training"]["device"])
    dataset = QuantPointDataset(observations, device=device)
    model, _, _ = build_training_model(dataset, config, protocol_path, device)
    checkpoint = load_checkpoint(checkpoint_path, model, device)
    space = TrainingSpace.from_state_dict(checkpoint["training_space"])
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    export_cfg = config.get("export", {})
    paths = export_quantitative_outputs(model, space, output, float(export_cfg["output_resolution_mm"]), int(export_cfg["output_batch_size"]), dataset=dataset, export_config={"bbox": config.get("bbox", {}), **export_cfg})
    paths.update(export_native_plane_reprojections(model, space, observations, output, output_psf=config.get("psf", {}).get("export", {})))
    return {key: str(value) for key, value in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--protocol", required=True)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--observations", required=True, action="append")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.config, args.protocol, args.checkpoint, args.observations, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
