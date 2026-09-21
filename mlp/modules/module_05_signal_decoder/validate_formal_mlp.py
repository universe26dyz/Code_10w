"""Held-out formal-candidate signal/gradient validation; never approves a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .mlp_model import MdmSignalMLP
from .provenance import sha256_file
from .test_mlp import fidelity_metrics
from .train_mlp import load_h5_split, validate_dataset_provenance


def validate_formal_mlp(checkpoint_path: str | Path, dataset_dir: str | Path, timing_pool: str | Path | None, protocol: str | Path, output_path: str | Path, device_name: str, batch_size: int, gradient_samples: int) -> dict[str, object]:
    output_path = Path(output_path)
    if output_path.exists(): raise FileExistsError(f"Formal validation output already exists: {output_path}")
    metadata, pool, teacher_protocol = validate_dataset_provenance(dataset_dir, timing_pool, protocol)
    checkpoint = torch.load(checkpoint_path, map_location=device_name, weights_only=False)
    if bool(checkpoint.get("functional_fixture")) or not bool(checkpoint.get("formal_candidate")) or checkpoint.get("validation_status") != "unvalidated":
        raise ValueError("Formal validation requires one unvalidated non-functional formal candidate checkpoint.")
    if metadata.get("schema") != "mlp_rr_synthetic/v1" and checkpoint.get("timing_pool_sha256") != metadata.get("timing_pool_sha256"):
        raise ValueError("Checkpoint and dataset timing-pool SHA256 differ.")
    model = MdmSignalMLP().to(device_name); model.load_state_dict(checkpoint["state_dict"]); model.eval()
    metrics = fidelity_metrics(model, load_h5_split(Path(dataset_dir) / "test.h5", "test"), pool, str(protocol), batch_size, gradient_samples, protocol=teacher_protocol)
    report = {"checkpoint_path": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path), "subject_split": {key: metadata[key] for key in ("train_subject_ids", "valid_subject_ids", "test_subject_ids")}, "test_timing_domain_relative_to_train": metadata["test_timing_relation_to_train"], "train_timing9_min_ms": metadata["train_timing9_min_ms"], "train_timing9_max_ms": metadata["train_timing9_max_ms"], "pool_timing9_min_ms": metadata["pool_timing9_min_ms"], "pool_timing9_max_ms": metadata["pool_timing9_max_ms"], **metrics, "validation_status": "awaiting_manual_review"}
    output_path.parent.mkdir(parents=True, exist_ok=True); output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "dataset-dir", "protocol", "output", "device"): parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--timing-pool")
    parser.add_argument("--batch-size", type=int, required=True); parser.add_argument("--gradient-samples", type=int, required=True)
    args = parser.parse_args(); print(json.dumps(validate_formal_mlp(args.checkpoint, args.dataset_dir, args.timing_pool, args.protocol, args.output, args.device, args.batch_size, args.gradient_samples), indent=2))


if __name__ == "__main__": main()
