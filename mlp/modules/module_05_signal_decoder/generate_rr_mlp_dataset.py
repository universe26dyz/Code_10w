"""CLI for the mDM-inspired coherent-RR synthetic dataset generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .synthetic_dataset import generate_rr_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("dataset"), dict):
        raise ValueError("RR synthetic config must contain a dataset mapping.")
    print(json.dumps(generate_rr_synthetic_dataset(args.output_dir, config["dataset"]), indent=2))


if __name__ == "__main__":
    main()
