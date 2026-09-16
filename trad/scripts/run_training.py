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
from modules.module_08_inference_export.reprojection import export_native_plane_reprojections
from modules.module_09_qc_benchmark.qc import validate_smoke_outputs


def run_reconstruction(config_path: str | Path, protocol_path: str | Path, observations: list[str | Path], output_dir: str | Path, subject_id: str | None = None) -> dict[str, object]:
    with Path(config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("training"), dict):
        raise ValueError("--config must be a mapping containing training.")
    if "device" not in config["training"]:
        raise ValueError("training.device must be explicit.")
    # The CLI is the reconstruction pipeline: HB1 registration is on unless a
    # caller explicitly disables it for a controlled tiny/unit invocation.
    config.setdefault("stack_initialization", {}).setdefault("enabled", True)
    dataset = QuantPointDataset(observations, device=torch.device(config["training"]["device"]))
    result = train_trad(dataset, config, protocol_path, output_dir, prepared_inputs=observations, subject_id=subject_id or Path(observations[0]).parents[1].name, command=" ".join(__import__("sys").argv))
    export_cfg = config.get("export")
    if not isinstance(export_cfg, dict) or "output_resolution_mm" not in export_cfg or "output_batch_size" not in export_cfg:
        raise ValueError("export.output_resolution_mm and export.output_batch_size must be explicit.")
    export_settings = {"bbox": config.get("bbox", {}), **export_cfg}
    paths = export_quantitative_outputs(result["model"], result["training_space"], result["output_dir"], float(export_cfg["output_resolution_mm"]), int(export_cfg["output_batch_size"]), dataset=dataset, export_config=export_settings)
    paths.update(export_native_plane_reprojections(result["model"], result["training_space"], observations, result["output_dir"], output_psf=config.get("psf", {}).get("export", {})))
    qc = validate_smoke_outputs(result["output_dir"])
    return {"outputs": {key: str(value) for key, value in paths.items()}, "qc": qc}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--observations", required=True, action="append")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_reconstruction(args.config, args.protocol, args.observations, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
