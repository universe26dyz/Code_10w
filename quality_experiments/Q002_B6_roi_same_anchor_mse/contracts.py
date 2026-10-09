"""Input and configuration guards for the isolated Q002 route."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping


Q002_EXPERIMENT_ID = "Q002_B6_roi_same_anchor_mse"
Q002S640_EXPERIMENT_ID = "Q002S640_B6_roi_same_anchor_mse"
STACKS = ("sax", "2ch", "4ch")


def cropped_observation_paths(root: str | Path) -> list[Path]:
    """Return only the three original B6 cropped observation archives."""

    prepared_root = Path(root)
    if "full_fov" in str(prepared_root).lower():
        raise ValueError("Q002 forbids full-FOV prepared inputs.")
    paths = [prepared_root / "CYJ" / stack / "observations.npz" for stack in STACKS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Q002 requires B6 cropped CYJ/{sax,2ch,4ch}/observations.npz: " + "; ".join(missing))
    return paths


def build_q002_route_config(
    resolved_config: Mapping[str, Any], *, cropped_prepared_root: str | Path, reconstruction_checkpoint: str | None = None,
    experiment_id: str = Q002_EXPERIMENT_ID, anchor_batch_size: int = 64,
) -> dict[str, Any]:
    """Copy B6 configuration and attach metadata without mutating its controls."""

    if reconstruction_checkpoint is not None:
        raise ValueError("Q002 must start a fresh reconstruction; B6/Q001 warm-start is forbidden.")
    if not isinstance(resolved_config.get("training"), Mapping) or not isinstance(resolved_config.get("decoder"), Mapping):
        raise ValueError("B6 resolved_config must contain training and decoder mappings.")
    root = Path(cropped_prepared_root)
    if "full_fov" in str(root).lower():
        raise ValueError("Q002 training root must be B6 cropped, not full-FOV.")
    expected_batch_size = {Q002_EXPERIMENT_ID: 64, Q002S640_EXPERIMENT_ID: 640}.get(experiment_id)
    if expected_batch_size is None or int(anchor_batch_size) != expected_batch_size:
        raise ValueError("Q002 route experiment_id must use its fixed anchor_batch_size.")
    sampling: dict[str, Any]
    if experiment_id == Q002_EXPERIMENT_ID:
        sampling = {"effective_scalar_budget": 640, "anchor_batch_size": 64, "weights_per_anchor": 10, "shared_training_psf_samples": 8}
    else:
        sampling = {
            "anchor_batch_size": 640,
            "weights_per_anchor": 10,
            "signal_residual_count": 6400,
            "shared_training_psf_samples": 8,
            "data_psf_inr_location_count": 5120,
            "B6_scalar_batch_size": 640,
        }
    return {
        "resolved_config": copy.deepcopy(dict(resolved_config)),
        "route": {
            "experiment_id": experiment_id,
            "parent": "frozen_mlp_B6",
            "training_prepared_root": str(root),
            "stack_initialization_prepared_root": str(root),
            "full_fov_used": False,
            "b6_reconstruction_warm_start": False,
            "sampling": sampling,
        },
    }
