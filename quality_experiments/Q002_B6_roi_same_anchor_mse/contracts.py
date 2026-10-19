"""Input and configuration guards for the isolated Q002 route."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping


Q002_EXPERIMENT_ID = "Q002_B6_roi_same_anchor_mse"
Q002S640_EXPERIMENT_ID = "Q002S640_B6_roi_same_anchor_mse"
Q002S640_K256_EXPERIMENT_ID = "Q002S640_K256"
Q003S640_EXPERIMENT_ID = "Q003S640_B6_roi_same_anchor_mse_plus_cosine"
Q003S640_10K_EXPERIMENT_ID = "Q003S640_B6_roi_same_anchor_mse_plus_cosine_10k"
Q004S640_K1_EXPERIMENT_ID = "Q004S640_K1_B6_roi_same_anchor_mse"
STACKS = ("sax", "2ch", "4ch")


def cropped_observation_paths(root: str | Path, subject_id: str = "CYJ") -> list[Path]:
    """Return only the three original B6 cropped observation archives."""

    prepared_root = Path(root)
    if "full_fov" in str(prepared_root).lower():
        raise ValueError("Q002 forbids full-FOV prepared inputs.")
    subject_paths = [prepared_root / subject_id / stack / "observations.npz" for stack in STACKS]
    direct_paths = [prepared_root / stack / "observations.npz" for stack in STACKS]
    paths = direct_paths if all(path.is_file() for path in direct_paths) else subject_paths
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Q002 requires B6 cropped {subject_id}/{{sax,2ch,4ch}}/observations.npz: " + "; ".join(missing))
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
    expected_batch_size = {Q002_EXPERIMENT_ID: 64, Q002S640_EXPERIMENT_ID: 640, Q002S640_K256_EXPERIMENT_ID: 640, Q003S640_EXPERIMENT_ID: 640, Q003S640_10K_EXPERIMENT_ID: 640, Q004S640_K1_EXPERIMENT_ID: 640}.get(experiment_id)
    if expected_batch_size is None or int(anchor_batch_size) != expected_batch_size:
        raise ValueError("Q002 route experiment_id must use its fixed anchor_batch_size.")
    sampling: dict[str, Any]
    if experiment_id == Q002_EXPERIMENT_ID:
        sampling = {"effective_scalar_budget": 640, "anchor_batch_size": 64, "weights_per_anchor": 10, "shared_training_psf_samples": 8}
    elif experiment_id == Q004S640_K1_EXPERIMENT_ID:
        sampling = {
            "anchor_batch_size": 640,
            "weights_per_anchor": 10,
            "signal_residual_count": 6400,
            "shared_training_psf_samples": 1,
            "data_psf_inr_location_count": 640,
            "B6_scalar_batch_size": 640,
        }
    elif experiment_id == Q002S640_K256_EXPERIMENT_ID:
        sampling = {
            "anchor_batch_size": 640,
            "weights_per_anchor": 10,
            "signal_residual_count": 6400,
            "shared_training_psf_samples": 256,
            "data_psf_inr_location_count": 163840,
            "B6_scalar_batch_size": 640,
            "psf_forward_chunk_size": 32,
        }
    else:
        sampling = {
            "anchor_batch_size": 640,
            "weights_per_anchor": 10,
            "signal_residual_count": 6400,
            "shared_training_psf_samples": 8,
            "data_psf_inr_location_count": 5120,
            "B6_scalar_batch_size": 640,
        }
    route = {
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
    if experiment_id in (Q003S640_EXPERIMENT_ID, Q003S640_10K_EXPERIMENT_ID):
        route["route"].update({
            "parent_experiment": Q002S640_EXPERIMENT_ID,
            "scientific_change": "add_fingerprint_cosine_only",
            "fingerprint_cosine": {"enabled": True, "weight": 1.0, "epsilon": 1.0e-8},
        })
    if experiment_id == Q003S640_10K_EXPERIMENT_ID:
        route["route"].update({
            "scientific_change": "add_fingerprint_cosine_only_for_primary_6k",
            "secondary_change": "continue_same_q003_optimization_to_10k",
            "continuation": {
                "stage_b_extension_iterations": 4000,
                "primary_checkpoint_iteration": 6000,
                "final_checkpoint_iteration": 10000,
                "extension_lr_policy": "continue_post_6000_optimizer_scheduler_state_no_reset_no_new_milestones",
            },
        })
    if experiment_id == Q004S640_K1_EXPERIMENT_ID:
        route["route"].update({
            "parent_experiment": Q002S640_EXPERIMENT_ID,
            "scientific_change": "training_psf_samples_8_to_central_1_only",
            "training_psf_semantics": "central_local_coordinate_no_gaussian_draw",
            "loss": "MSE_only",
        })
    if experiment_id == Q002S640_K256_EXPERIMENT_ID:
        route["route"].update({
            "parent_experiment": Q002S640_EXPERIMENT_ID,
            "scientific_change": "training_psf_samples_8_to_256_only",
            "loss": "MSE_only",
            "training_psf_semantics": "same-anchor local PSF neighbourhood samples averaged before the 10-weight joint-vector MSE",
            "psf_forward_chunking": {
                "enabled": True,
                "chunk_size": 32,
                "logical_samples_per_anchor": 256,
                "semantics": "one K256 neighbourhood draw per logical optimizer step; chunks only split INR/decoder forward accumulation with one backward and optimizer step",
            },
            "fit_first_n_samples_semantics": {
                "comparison_label": "matched per-observation training PSF sample count (K=256)",
                "matched_per_observation_training_psf_sample_count": True,
                "per_observation_training_psf_samples": {"proposed": 256, "fit_first_nesvor": 256},
                "same_observation_semantics": False,
                "same_forward_model": False,
                "same_loss": False,
                "same_batch_size": False,
                "same_total_sampled_locations_per_optimizer_step": False,
                "same_flops": False,
                "same_memory": False,
                "same_optimizer_exposure": False,
                "same_computational_budget": False,
            },
        })
    return route
