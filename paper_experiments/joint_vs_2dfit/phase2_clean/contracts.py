"""Frozen Phase-2 scientific and provenance contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

APPROVED_DECODER_SHA256 = "f83280d4f73bfc9e0540b9606ee20bf83395339ad4e54dd0fb7be93ed9c9a7a9"
FROZEN_GRID = {"t1_ms": (20, 20, 500, 505, 5, 1500, 1520, 20, 2500), "t2_ms": (5, 5, 100, 110, 10, 200), "b1": (0.1, 0.05, 1.2), "constraint": "T1 > T2"}
BASELINE_POSE_POLICY = {"registration": "stack", "transformation_optimization": True, "t1_t2_independent": True, "t1_to_t2_pose_reuse": False, "slice_thickness_mm": 8.0, "output_resolution_mm": 1.0, "n_iter": 10000, "n_samples": 256, "inference_samples": 128, "pixel_variance_weight": 1.0}
PROPOSED_CONTRACT = {"source": "quality_experiments/Q002_B6_roi_same_anchor_mse", "coarsest_mm": 8.0, "finest_mm": 1.0, "scale": 1.5, "features_per_level": 2, "hashmap_log2": 19, "latent_dim": 32, "width": 64, "depth": 3, "batch": 640, "seed": 20260911, "stage_a_iterations": 2000, "stage_b_iterations": 4000, "amplitude_guidance": False, "group_shared_rigid_pose": True}
ELIGIBLE_REAL_SUBJECTS = ("CYJ", "DYZ", "HHZ", "HJL")

def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()

def assert_phase1_ready(gate: dict) -> None:
    if not (gate.get("renderer_convergence_status") == "pass" and gate.get("matlab_parity_status") == "pass" and gate.get("formal_synthetic_reconstruction_ready") is True):
        raise ValueError("Phase-1 successor gate is not formally ready")

def compare_observation_hashes(proposed: dict[str, str], fit_first: dict[str, str]) -> None:
    if proposed != fit_first: raise ValueError("observation hash mismatch between proposed and fit_first")

def observation_hashes(root: str | Path) -> dict[str, str]:
    root = Path(root); result = {stack: sha256(root / stack / "observations.npz") for stack in ("sax", "2ch", "4ch")}
    if len(result) != 3: raise ValueError("all three observation stacks are required")
    return result

def validate_proposed_subject(subject_id: str, tr_ms: float, vps: int) -> None:
    if subject_id == "DYL" or int(vps) != 87 or float(tr_ms) != 2.61:
        raise ValueError(f"{subject_id} is incompatible with approved decoder: expected TR=2.61/VPS=87; got TR={tr_ms}/VPS={vps}" if subject_id != "DYL" else "DYL rejected: VPS=85 is incompatible with approved VPS=87 decoder")

def write_immutable_json(path: str | Path, value: dict) -> Path:
    path = Path(path)
    if path.exists(): raise FileExistsError(f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n"); return path
