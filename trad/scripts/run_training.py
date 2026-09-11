"""Explicit CLI for one Trad v1 reconstruction; all tuning lives in YAML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset
from modules.module_07_objective_training.trad_trainer import train_trad
from modules.module_08_inference_export.export_quantitative import export_quantitative_outputs
from modules.module_09_qc_benchmark.qc import validate_smoke_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--observations", required=True, action="append")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("training"), dict):
        raise ValueError("--config must be a mapping containing training.")
    if "device" not in config["training"]:
        raise ValueError("training.device must be explicit.")
    dataset = QuantPointDataset(args.observations, device=torch.device(config["training"]["device"]))
    result = train_trad(dataset, config, args.protocol, args.output_dir)
    export_cfg = config.get("export")
    if not isinstance(export_cfg, dict) or "output_resolution_mm" not in export_cfg or "output_batch_size" not in export_cfg:
        raise ValueError("export.output_resolution_mm and export.output_batch_size must be explicit.")
    paths = export_quantitative_outputs(result["model"], result["training_space"], result["output_dir"], float(export_cfg["output_resolution_mm"]), int(export_cfg["output_batch_size"]))
    qc = validate_smoke_outputs(result["output_dir"])
    print(json.dumps({"outputs": {key: str(value) for key, value in paths.items()}, "qc": qc}, sort_keys=True))


if __name__ == "__main__":
    main()
