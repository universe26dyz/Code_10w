"""Record the current joint-method contract without running reconstruction."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

import yaml

from _common import json_safe, write_json


SCHEMA = {
    "images": "[Nobs,H,W]",
    "masks": "[Nobs,H,W]",
    "group_idx": "[Nobs]",
    "weight_idx": "[Nobs]",
    "stack_idx": "[Nobs]",
    "acquisition_time_ms": "[Nobs]",
    "timing9_ms": "[Nobs,9]",
    "affine_lps_rc": "[Nobs,4,4]",
    "pixel_spacing_rc_mm": "[Nobs,2]",
    "slice_thickness_mm": "[Nobs]",
    "tr_ms": "[Nobs]",
    "vps": "[Nobs]",
}


def _git(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=repo_root, text=True).strip()

    try:
        return {"sha": run("rev-parse", "HEAD"), "dirty_status": run("status", "--short"), "recent_commits": run("log", "-5", "--oneline").splitlines()}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": str(error)}


def build_report(*, repo_root: str | Path, protocol_path: str | Path, reference_paths_path: str | Path | None) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    protocol = yaml.safe_load(Path(protocol_path).read_text(encoding="utf-8"))
    references = None if reference_paths_path is None else yaml.safe_load(Path(reference_paths_path).read_text(encoding="utf-8"))
    return {
        "audit_type": "current_joint_method",
        "repo_root": str(root),
        "git": _git(root),
        "hhz_protocol": protocol,
        "current_reconstruction": {
            "reusable_modules": [
                "mlp/modules/module_04_quantitative_inr/quantitative_inr.py",
                "mlp/modules/module_05_signal_decoder/checkpoint_loader.py",
                "trad/modules/module_03_dataset_geometry/quantitative_point_dataset.py",
                "quality_experiments/Q002_B6_roi_same_anchor_mse/",
            ],
            "key_config": "mlp/configs/recon_server_train_example.yaml",
        },
        "inr_ranges": {"t1_ms": "(t2_ms, 2500]", "t2_ms": "[5, 200]", "b1": "[0.1, 1.2]", "amplitude": "> 0"},
        "expected_observation_schema": SCHEMA,
        "reference_path_config": references,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--reference-paths")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_json(args.output, json_safe(build_report(repo_root=args.repo_root, protocol_path=args.protocol, reference_paths_path=args.reference_paths)))
    print(Path(args.output))


if __name__ == "__main__":
    main()
