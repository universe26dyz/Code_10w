"""Read-only validation of the ready Q001 full-FOV input bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


STACK_CONTRACT = {"sax": (14, 140, [288, 256]), "2ch": (15, 150, [288, 256]), "4ch": (12, 120, [256, 288])}
REQUIRED_SEMANTICS = "MP-PCA(full-FOV MIND_mag_reg)"
ARTIFACTS = ("preprocessed_mat", "prepared_observations", "prepared_manifest", "timing", "qc_summary", "cropped_vs_full_qc")


def _artifact_path(root: Path, stack: str, name: str) -> Path:
    paths = {
        "preprocessed_mat": root / "full_fov_preprocessed" / "CYJ" / stack / "preprocessed.mat",
        "prepared_observations": root / "full_fov_prepared" / "CYJ" / stack / "observations.npz",
        "prepared_manifest": root / "full_fov_prepared" / "CYJ" / stack / "manifest.json",
        "timing": root / "full_fov_prepared" / "CYJ" / stack / "timing.npy",
        "qc_summary": root / "full_fov_prepared" / "CYJ" / stack / "qc_summary.json",
        "cropped_vs_full_qc": root / "qc" / "CYJ" / f"{stack}.json",
    }
    return paths[name]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_q001_input_bundle(root: str | Path) -> dict[str, Any]:
    """Reject any non-ready or hash-inconsistent full-FOV input bundle."""

    root = Path(root).resolve(); manifest_path = root / "q001_input_manifest.json"
    if not manifest_path.is_file(): raise FileNotFoundError(f"Missing Q001 input manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "READY_FOR_Q001_RECON_IMPLEMENTATION": raise ValueError("Q001 input manifest is not reconstruction-ready.")
    if manifest.get("formal_preprocessing_semantics") != REQUIRED_SEMANTICS: raise ValueError("Q001 input semantics are not formal full-FOV MP-PCA.")
    records = manifest.get("stacks")
    if not isinstance(records, dict): raise ValueError("Q001 input manifest stacks must be a mapping.")
    for stack, (groups, observations, shape) in STACK_CONTRACT.items():
        record = records.get(stack)
        if not isinstance(record, dict) or record.get("qc_passed") is not True: raise ValueError(f"{stack} strict input QC did not pass.")
        if (record.get("group_count"), record.get("observation_count"), record.get("full_image_shape_rows_cols")) != (groups, observations, shape): raise ValueError(f"{stack} Q001 input shape/count contract differs from formal CYJ values.")
        for name in ARTIFACTS:
            artifact = record.get(name)
            if not isinstance(artifact, dict): raise ValueError(f"{stack} input manifest lacks {name}.")
            expected = str(artifact.get("sha256", "")); source = _artifact_path(root, stack, name)
            if not source.is_file(): raise FileNotFoundError(f"{stack} input artifact missing: {source}")
            if sha256(source) != expected: raise ValueError(f"{stack} {name} SHA256 mismatch.")
    return manifest
