"""B6-resolved-config provenance and Q001 route metadata guards."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch


EXPERIMENT_IDS = {"Q001A_full_fov_full_recon", "Q001B_full_fov_reg_cropped_recon"}


def load_b6_resolved_config(model_path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(Path(model_path), map_location="cpu", weights_only=False)
    config = checkpoint.get("resolved_config") if isinstance(checkpoint, dict) else None
    if not isinstance(config, dict): raise ValueError("Formal B6 model.pt lacks resolved_config.")
    if not isinstance(config.get("training"), dict) or not isinstance(config.get("decoder"), dict): raise ValueError("Formal B6 resolved_config lacks training/decoder mappings.")
    return copy.deepcopy(config)


def build_route_config(resolved_config: dict[str, Any], experiment_id: str, *, full_prepared_root: str, cropped_prepared_root: str | None, reconstruction_checkpoint: str | None = None) -> dict[str, Any]:
    """Keep B6 training config byte-for-byte as source data; add route metadata separately."""

    if experiment_id not in EXPERIMENT_IDS: raise ValueError(f"Unsupported Q001 experiment: {experiment_id}")
    if reconstruction_checkpoint is not None: raise ValueError("Q001 must start new reconstruction; B6 reconstruction warm-start is forbidden.")
    if experiment_id.startswith("Q001A") and cropped_prepared_root is not None: raise ValueError("Q001A must not declare cropped training inputs.")
    if experiment_id.startswith("Q001B") and not cropped_prepared_root: raise ValueError("Q001B requires explicit cropped reconstruction inputs.")
    route = {"experiment_id": experiment_id, "full_fov_prepared_root": full_prepared_root, "training_prepared_root": full_prepared_root if experiment_id.startswith("Q001A") else cropped_prepared_root, "stack_initialization_prepared_root": full_prepared_root, "full_fov_used_for_stack_initialization_only": experiment_id.startswith("Q001B"), "b6_reconstruction_warm_start": False}
    return {"resolved_config": copy.deepcopy(resolved_config), "route": route}
