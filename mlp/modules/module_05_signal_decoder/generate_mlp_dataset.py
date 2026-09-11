"""CLI wrapper for one-time HHZ-teacher HDF5 synthetic dataset generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .synthetic_dataset import generate_mlp_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--timing-pool", required=True); parser.add_argument("--protocol", required=True); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as handle: config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("dataset"), dict): raise ValueError("Config must contain dataset mapping.")
    dataset = config["dataset"]
    for key in ("train", "valid", "test", "seed", "teacher_chunk_size", "teacher_device"):
        if key not in dataset: raise ValueError(f"dataset.{key} must be explicit.")
    print(generate_mlp_dataset(args.timing_pool, args.protocol, args.output_dir, dataset, int(dataset["seed"]), int(dataset["teacher_chunk_size"]), str(dataset["teacher_device"])))


if __name__ == "__main__": main()
