"""Resolve and validate the decoder referenced by a current/B6 provenance bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _common import json_safe, sha256, write_json


def _load_mapping(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        import torch

        value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint metadata must be a mapping: {path}")
    return value


def _approved(metadata: dict[str, Any]) -> bool:
    return metadata.get("validation_status") == "approved_by_manual_review" and bool(metadata.get("approved_by_manual_review", metadata.get("validation_status") == "approved_by_manual_review"))


def audit_checkpoint(checkpoint: str | Path, *, require_approved: bool = True, provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    path = Path(checkpoint).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Decoder checkpoint does not exist: {path}")
    metadata = _load_mapping(path)
    accepted = _approved(metadata)
    if require_approved and not accepted:
        raise ValueError("Decoder is not approved_by_manual_review; refusing paper manifest.")
    fields = ("validation_status", "approved_by_manual_review", "architecture", "protocol_hhz_v1", "parameter_ranges", "tr_ms", "vps", "timing9_min_ms", "timing9_max_ms", "train_timing9_min_ms", "train_timing9_max_ms")
    report = {"audit_type": "decoder_checkpoint", "absolute_path": str(path), "sha256": sha256(path), "approved_for_paper": accepted}
    report.update({field: metadata.get(field) for field in fields})
    report["approved_by_manual_review"] = accepted
    if report["tr_ms"] is None and isinstance(report["protocol_hhz_v1"], dict):
        report["tr_ms"] = report["protocol_hhz_v1"].get("tr_ms")
        report["vps"] = report["protocol_hhz_v1"].get("vps")
    if provenance:
        report["resolved_from_provenance"] = provenance
    return json_safe(report)


def resolve_from_b6(model_path: str | Path) -> tuple[Path, dict[str, Any]]:
    model = Path(model_path).resolve()
    record = _load_mapping(model)
    config = record.get("resolved_config")
    if not isinstance(config, dict) or not isinstance(config.get("decoder"), dict):
        raise ValueError("Reference B6 model lacks resolved_config.decoder provenance.")
    declared = Path(config["decoder"].get("checkpoint", ""))
    bundle_root = model.parents[2] if len(model.parents) > 2 else model.parent
    candidates = [declared, bundle_root / "inputs" / "mlp" / "signal_simulator_approved.pth", bundle_root / "inputs" / "mlp" / "signal_simulator_best.pth"]
    source_sha = record.get("source_mlp_checkpoint_sha256")
    for candidate in candidates:
        if candidate.is_file() and (not source_sha or sha256(candidate) == source_sha):
            return candidate.resolve(), {"reference_b6_model": str(model), "declared_decoder_path": str(declared), "source_mlp_checkpoint_sha256": source_sha}
    raise FileNotFoundError("Could not resolve the B6 decoder locally; provide --checkpoint explicitly.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--reference-b6-model")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.reference_b6_model:
        checkpoint, provenance = resolve_from_b6(args.reference_b6_model)
    else:
        checkpoint, provenance = Path(args.checkpoint), None
    write_json(args.output, audit_checkpoint(checkpoint, provenance=provenance))
    print(Path(args.output))


if __name__ == "__main__":
    main()
