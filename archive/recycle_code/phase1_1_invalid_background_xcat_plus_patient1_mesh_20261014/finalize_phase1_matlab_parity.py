"""Non-destructively finalize the Phase-1 gate from an external MATLAB reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parity_passes(result: dict) -> bool:
    return (
        result.get("status") == "pass"
        and float(result.get("max_abs_raw_error", float("inf"))) <= 1e-5
        and float(result.get("min_normalized_cosine", float("-inf"))) >= 0.999999
        and result.get("all_finite") is True
    )


def build_final_gate(prior_gate: dict, validator_result: dict, provenance: dict) -> dict:
    """Build the immutable successor gate without changing its predecessor."""

    if not parity_passes(validator_result):
        raise ValueError("validated MATLAB parity does not meet the Phase-1 acceptance criteria")
    if prior_gate.get("renderer_convergence_status") != "pass":
        raise ValueError("renderer convergence must pass before final readiness can be set")
    return {
        "phase": "1.2_matlab_parity_finalization",
        "renderer_convergence_status": prior_gate["renderer_convergence_status"],
        "accepted_quadrature": prior_gate["accepted_quadrature"],
        "matlab_parity_status": "pass",
        "matlab_validator_result": validator_result,
        "matlab_reference_provenance": provenance,
        "formal_synthetic_reconstruction_ready": True,
        "blockers": [],
        "frozen_contract": prior_gate["frozen_contract"],
        "inherited_relevant_sha256": prior_gate["relevant_sha256"],
    }


def finalize(output_root: str | Path, parity_json: str | Path, output_json: str | Path | None = None) -> dict:
    root = Path(output_root)
    prior_path = root / "phase1_final_gate.json"
    parity_path = Path(parity_json)
    reference_path = root / "matlab_hhz_reference.mat"
    config_path = root / "matlab_hhz_reference_config.json"
    exporter_path = Path(__file__).resolve().parents[3] / "trad/modules/module_05_signal_decoder/export_matlab_signal_reference.m"
    output_path = Path(output_json) if output_json else root / "phase1_final_gate_matlab_parity.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    for path in (prior_path, parity_path, reference_path, config_path, exporter_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    prior = json.loads(prior_path.read_text())
    parity = json.loads(parity_path.read_text())
    expected_reference = str(reference_path.resolve())
    if parity.get("reference_mat") != expected_reference:
        raise ValueError("validator result does not name this output root's MATLAB reference")
    provenance = {
        "authority": "external MATLAB execution of original-HHZ export_matlab_signal_reference.m",
        "reference_mat": expected_reference,
        "reference_mat_sha256": sha256(reference_path),
        "reference_config": str(config_path.resolve()),
        "reference_config_sha256": sha256(config_path),
        "original_exporter": str(exporter_path.resolve()),
        "original_exporter_sha256": sha256(exporter_path),
        "validator_result_json": str(parity_path.resolve()),
        "validator_result_sha256": sha256(parity_path),
        "validator": "validate_hhz_parity.py --reference-config ... --reference-mat ...",
        "prior_final_gate": str(prior_path.resolve()),
        "prior_final_gate_sha256": sha256(prior_path),
    }
    gate = build_final_gate(prior, parity, provenance)
    output_path.write_text(json.dumps(gate, indent=2) + "\n")
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a non-destructive Phase-1 MATLAB-parity final gate.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--parity-json", required=True)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    print(json.dumps(finalize(args.output_root, args.parity_json, args.output_json), indent=2))


if __name__ == "__main__":
    main()
