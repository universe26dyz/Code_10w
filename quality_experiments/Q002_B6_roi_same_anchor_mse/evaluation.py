"""Immutable Q002 evaluation metadata."""

from __future__ import annotations


def evaluation_plan(checkpoint_sha256: str) -> dict[str, dict[str, object]]:
    return {
        "central_no_map_psf": {"psf_samples": 1, "primary": True, "checkpoint_sha256": checkpoint_sha256},
        "signal_psf_K8": {"psf_samples": 8, "seed": 20260911, "checkpoint_sha256": checkpoint_sha256},
        "formal_export_psf128": {"output_resolution_mm": 1.0, "output_psf_factor": 1.0, "psf_samples": 128, "seed": 20260911, "checkpoint_sha256": checkpoint_sha256},
        "fingerprint_cosine": {"training_objective": False, "role": "read_only_diagnostic"},
    }
