"""Exact non-interpolating Q001 evaluation/export metadata."""

from __future__ import annotations


ROI_BY_STACK = {"sax": (slice(71, 216), slice(63, 192)), "2ch": (slice(71, 216), slice(63, 192)), "4ch": (slice(63, 192), slice(71, 216))}


def fixed_roi(stack: str) -> tuple[slice, slice]:
    try: return ROI_BY_STACK[stack]
    except KeyError as exc: raise ValueError(f"Unknown Q001 stack: {stack}") from exc


def evaluation_plan(checkpoint_sha256: str, route: str) -> dict[str, dict[str, object]]:
    if route not in {"Q001A", "Q001B"}: raise ValueError("Evaluation route must be Q001A or Q001B.")
    return {"mapping_central_no_psf": {"psf_samples": 1, "checkpoint_sha256": checkpoint_sha256, "primary": True, "interpolation": False}, "mapping_map_psf_K32": {"status": "INVALID_AS_MAP_PSF", "reason": "native-plane parameter fields are point samples; n_samples affected signal only", "checkpoint_sha256": checkpoint_sha256}, "map_domain_psf_K32": {"domain": "map", "psf_implementation": "trad.evaluation.scmr.map_domain_psf", "psf_samples": 32, "seed": 20260911, "checkpoint_sha256": checkpoint_sha256}, "signal_psf_K8": {"psf_samples": 8, "seed": 20260911, "checkpoint_sha256": checkpoint_sha256}}
