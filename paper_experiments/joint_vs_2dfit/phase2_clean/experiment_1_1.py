"""Dry-run CYJ Experiment 1.1 launcher for sparse Q002S640 reconstruction."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from quality_experiments.Q002_B6_roi_same_anchor_mse.contracts import Q002S640_EXPERIMENT_ID

from .contracts import APPROVED_DECODER_SHA256, B6_CONFIG_SOURCE_SHA256, sha256
from .sparse_profile_manifest import validate_sparse_profile_manifest


def build_experiment_1_1_command(manifest: dict[str, Any], *, expected_generator_git_sha: str, expected_manifest_sha256: str, b6_model: str | Path, decoder: str | Path, output: str | Path, device: str = "cuda:0", protocol: str = "trad/configs/protocol_hhz_v1.yaml", executable: str = sys.executable) -> str:
    """Build (but never execute) the one permitted Experiment 1.1 invocation."""
    if manifest.get("_manifest_path") is None or manifest.get("_manifest_sha256") is None:
        raise ValueError("manifest must be validated by validate_sparse_profile_manifest")
    if not isinstance(expected_generator_git_sha, str) or len(expected_generator_git_sha) != 40 or manifest.get("git_sha") != expected_generator_git_sha:
        raise ValueError("Experiment 1.1 requires an explicit, matching sparse generator git SHA")
    if not isinstance(expected_manifest_sha256, str) or len(expected_manifest_sha256) != 64 or manifest.get("_manifest_sha256") != expected_manifest_sha256:
        raise ValueError("Experiment 1.1 requires an explicit, matching sparse manifest SHA256")
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"refusing existing Experiment 1.1 output root: {target}")
    values = [
        executable, "-m", "quality_experiments.Q002_B6_roi_same_anchor_mse.run_q002_reconstruction",
        "--cropped-prepared-root", str(manifest["prepared_root"]),
        "--subject-id", str(manifest["subject"]),
        "--sparse-manifest", str(manifest["_manifest_path"]),
        "--expected-sparse-generator-git-sha", expected_generator_git_sha,
        "--expected-sparse-manifest-sha256", expected_manifest_sha256,
        "--b6-model", str(b6_model),
        "--signal-simulator", str(decoder),
        "--protocol", protocol,
        "--output", str(target),
        "--device", device,
        "--experiment-id", Q002S640_EXPERIMENT_ID,
        "--anchor-batch-size", "640",
    ]
    return " ".join(shlex.quote(value) for value in values)


def validate_dependencies(*, b6_model: str | Path, decoder: str | Path, protocol: str | Path) -> None:
    """Cheap, CPU-only dependency gate shared by dry-run and execution."""
    b6, frozen_decoder, protocol_path = Path(b6_model), Path(decoder), Path(protocol)
    if not b6.is_file() or sha256(b6) != B6_CONFIG_SOURCE_SHA256:
        raise ValueError("Experiment 1.1 B6 model SHA256 disagrees with the frozen contract")
    if not frozen_decoder.is_file() or sha256(frozen_decoder) != APPROVED_DECODER_SHA256:
        raise ValueError("Experiment 1.1 decoder SHA256 disagrees with the approved frozen decoder")
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Experiment 1.1 protocol does not exist: {protocol_path}")


def run(args: argparse.Namespace) -> str:
    """Validate pins and either print or explicitly execute Q002S640 once."""
    manifest = validate_sparse_profile_manifest(
        args.sparse_manifest, subject="CYJ", strict=True,
        expected_generator_git_sha=args.expected_generator_git_sha,
        expected_manifest_sha256=args.expected_manifest_sha256,
    )
    validate_dependencies(b6_model=args.b6_model, decoder=args.decoder, protocol=args.protocol)
    command = build_experiment_1_1_command(
        manifest, expected_generator_git_sha=args.expected_generator_git_sha,
        expected_manifest_sha256=args.expected_manifest_sha256,
        b6_model=args.b6_model, decoder=args.decoder, output=args.output,
        device=args.device, protocol=args.protocol,
    )
    if args.execute:
        subprocess.run(command, shell=True, check=True)
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse-manifest", required=True)
    parser.add_argument("--b6-model", required=True)
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--expected-generator-git-sha", required=True, help="commit recorded by the sparse-input generation run")
    parser.add_argument("--expected-manifest-sha256", required=True, help="immutable digest recorded for this named sparse case")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--protocol", default="trad/configs/protocol_hhz_v1.yaml")
    parser.add_argument("--execute", action="store_true", help="start the validated Q002S640 run; omitted means dry-run")
    args = parser.parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
