"""Create a successor readiness artifact only from a passing server preflight JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contracts import sha256

REQUIRED_GATES = (
    "actual_observation_hash_gate",
    "references_ready",
    "approved_decoder_gate",
    "b6_config_source_gate",
    "environment_gate",
    "phase1_gate",
)


def finalize_server_readiness(preflight_json: str | Path, output: str | Path) -> Path:
    source, destination = Path(preflight_json), Path(output)
    if destination.exists():
        raise FileExistsError(f"refusing overwrite: {destination}")
    payload = json.loads(source.read_text())
    failed = [key for key in REQUIRED_GATES if payload.get(key) is not True]
    if failed:
        raise ValueError(f"server preflight did not pass: {', '.join(failed)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({
        "schema": "phase2_server_readiness/v1",
        "source_preflight": {"path": str(source.resolve()), "sha256": sha256(source)},
        "server_gates": {key: True for key in REQUIRED_GATES},
        "formal_reconstruction_ready": True,
    }, indent=2, sort_keys=True) + "\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(finalize_server_readiness(args.preflight_json, args.output))


if __name__ == "__main__":
    main()
