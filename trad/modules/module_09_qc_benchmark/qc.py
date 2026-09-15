"""QC for requested artifacts; no repeated signal-parity or long benchmark work."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml


REQUIRED_OUTPUTS = (
    "T1_3D.nii.gz", "T2_3D.nii.gz", "B1_3D.nii.gz", "amplitude_3D.nii.gz",
    "model.pt", "final_rigid_poses.json", "training_log.csv", "config_resolved.yaml",
)


def _validate_intensity_normalization(provenance: object, source: str) -> float:
    if not isinstance(provenance, dict):
        raise ValueError(f"{source} lacks intensity normalization provenance.")
    if provenance.get("method") != "trimmed_mean" or provenance.get("lower_quantile") != 0.1 or provenance.get("upper_quantile") != 0.9:
        raise ValueError(f"{source} has unsupported intensity normalization provenance.")
    scale = provenance.get("scale")
    if not isinstance(scale, (int, float)) or not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"{source} intensity_scale must be finite and positive.")
    return float(scale)


def validate_smoke_outputs(output_dir: str | Path) -> dict[str, object]:
    """Verify finite exported fields and the rigid/no-deformation checkpoint contract."""

    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("nibabel is required for NIfTI QC.") from exc
    output = Path(output_dir)
    missing = [name for name in REQUIRED_OUTPUTS if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Smoke output lacks required artifacts: {missing}")
    finite = {}
    for name in REQUIRED_OUTPUTS[:4]:
        data = np.asarray(nib.load(output / name).get_fdata())
        finite[name] = bool(np.isfinite(data).all())
    if not all(finite.values()):
        raise FloatingPointError(f"Non-finite quantitative NIfTI fields: {finite}")
    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=False)
    checkpoint_scale = _validate_intensity_normalization(checkpoint.get("intensity_normalization"), "Checkpoint")
    state_names = set(checkpoint["model_state"])
    if "rigid_psf.axisangle" not in state_names or any("deform" in name.lower() for name in state_names):
        raise ValueError("Checkpoint violates rigid-only model contract.")
    with (output / "final_rigid_poses.json").open(encoding="utf-8") as handle:
        poses = json.load(handle)
    if poses.get("coordinate_convention") != "physical RAS mm; trans_first=true; world=R@(local+T)":
        raise ValueError("final_rigid_poses.json lacks the required physical RAS convention.")
    with (output / "training_log.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not all(np.isfinite(float(row["total"])) for row in rows):
        raise FloatingPointError("training_log.csv has no finite total losses.")
    with (output / "config_resolved.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("training"), dict):
        raise ValueError("config_resolved.yaml lacks training mapping.")
    resolved_scale = _validate_intensity_normalization(config["training"].get("intensity_normalization"), "Resolved config")
    if checkpoint_scale != resolved_scale:
        raise ValueError("Checkpoint and resolved config intensity_scale differ.")
    report = {"finite_volumes": finite, "iterations": len(rows), "rigid_tensor": True, "deformable": False, "intensity_scale": checkpoint_scale}
    with (output / "qc_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report
