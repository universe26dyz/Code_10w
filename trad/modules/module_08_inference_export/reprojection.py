"""Native-plane, read-only reprojections from a trained quantitative model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


def _local_grid(height: int, width: int, spacing_rc: np.ndarray, device: torch.device) -> torch.Tensor:
    rows, cols = np.mgrid[0:height, 0:width]
    return torch.as_tensor(
        np.column_stack(((cols.ravel() - (width - 1) / 2.0) * spacing_rc[1], (rows.ravel() - (height - 1) / 2.0) * spacing_rc[0], np.zeros(height * width))),
        dtype=torch.float32,
        device=device,
    )


def export_native_plane_reprojections(
    model: torch.nn.Module,
    space: Any,
    prepared_inputs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    output_psf: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write all ten observed/predicted/residual planes and T1/T2 planes.

    Prepared arrays are opened read-only and are copied into output archives;
    group/weight/timing/TR/VPS metadata is retained verbatim.
    """

    output = Path(output_dir)
    device = next(model.parameters()).device
    output_psf = output_psf or {}
    n_samples = int(output_psf.get("n_samples", 1)) if bool(output_psf.get("enabled", False)) else 1
    paths: dict[str, Path] = {}
    offset = 0
    was_training = model.training; model.eval()
    for source_text in prepared_inputs:
        source = Path(source_text)
        with np.load(source, allow_pickle=False) as data:
            images = np.asarray(data["images"], dtype=np.float32)
            masks = np.asarray(data["masks"], dtype=bool)
            groups = np.asarray(data["group_idx"], dtype=np.int64)
            weights = np.asarray(data["weight_idx"], dtype=np.int64)
            timing = np.asarray(data["timing9_ms"], dtype=np.float32)
            spacing = np.asarray(data["pixel_spacing_rc_mm"], dtype=np.float32)
            copied = {key: np.asarray(data[key]) for key in ("group_idx", "weight_idx", "timing9_ms", "tr_ms", "vps", "masks")}
        prediction = np.full_like(images, np.nan, dtype=np.float32)
        t1 = np.full_like(images, np.nan, dtype=np.float32)
        t2 = np.full_like(images, np.nan, dtype=np.float32)
        for observation in range(images.shape[0]):
            group = offset + int(groups[observation])
            h, w = images.shape[1:]
            local = _local_grid(h, w, spacing[observation], device)
            group_idx = torch.full((local.shape[0],), group, dtype=torch.long, device=device)
            weight_idx = torch.full((local.shape[0],), int(weights[observation]), dtype=torch.long, device=device)
            timing_batch = torch.as_tensor(timing[observation], device=device).expand(local.shape[0], -1)
            with torch.no_grad():
                batch = {"xyz": space.local_to_train(local), "group_idx": group_idx, "weight_idx": weight_idx, "timing": timing_batch}
                signal = (model(batch, n_samples) * model.intensity_scale).reshape(h, w)
                world = model.rigid_psf.transform_local_to_ras(batch["xyz"], group_idx)
                fields = model.inr(world)
            keep = masks[observation]
            prediction[observation, keep] = signal.detach().cpu().numpy()[keep]
            t1[observation, keep] = fields["t1_ms"].reshape(h, w).detach().cpu().numpy()[keep]
            t2[observation, keep] = fields["t2_ms"].reshape(h, w).detach().cpu().numpy()[keep]
        label = source.parent.name
        signal_path = output / f"signal_reprojection_{label}.npz"
        np.savez_compressed(signal_path, observed=images, predicted=prediction, residual=prediction - images, **copied)
        parameter_path = output / f"t1_t2_native_plane_{label}.npz"
        np.savez_compressed(parameter_path, t1_ms=t1, t2_ms=t2, **copied)
        paths[f"signal_{label}"] = signal_path
        paths[f"t1_t2_{label}"] = parameter_path
        offset += int(np.unique(groups).size)
    if was_training: model.train()
    return paths
