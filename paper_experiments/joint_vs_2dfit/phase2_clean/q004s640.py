"""Subject-general Q004S640 central-K1 reconstruction case adapter."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from .contracts import sha256, write_immutable_json


Q004_EXPERIMENT_ID = "Q004S640_K1_B6_roi_same_anchor_mse"
FORMAL_CASES = ("L100_fold0", "L50_fold0", "L50_fold1", "L25_fold0", "L25_fold1", "L25_fold2", "L25_fold3")
CYJ_CASES = FORMAL_CASES  # Backward-compatible import for the established CYJ route.


def _validate_case_manifest(subject: str, condition: str, sparse_manifest: str | Path) -> Path:
    if condition not in FORMAL_CASES:
        raise ValueError(f"unsupported formal condition: {condition}")
    path = Path(sparse_manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("subject") != subject or payload.get("condition") != condition.split("_", 1)[0]:
        raise ValueError("sparse manifest does not match the requested subject/condition")
    return path


def select_prepared_root(subject: str, condition: str, l100_prepared_root: str | Path, sparse_prepared_root: str | Path,
                         sparse_manifest: str | Path) -> Path:
    """Choose the immutable R0 source for L100 and the sparse source for held-out cases."""
    _validate_case_manifest(subject, condition, sparse_manifest)
    l100, sparse = Path(l100_prepared_root).resolve(), Path(sparse_prepared_root).resolve()
    if condition == "L100_fold0":
        if l100 != sparse:
            raise ValueError("L100 must reuse the R0/L100 prepared root as the sparse prepared root")
        return l100
    return sparse


def select_cyj_prepared_root(condition: str, l100_prepared_root: str | Path, sparse_prepared_root: str | Path,
                             sparse_manifest: str | Path) -> Path:
    """Compatibility adapter for existing CYJ-only callers."""
    return select_prepared_root("CYJ", condition, l100_prepared_root, sparse_prepared_root, sparse_manifest)


def build_q004s640_command(condition: str, prepared_root: str | Path, *, subject: str = "CYJ", decoder: str | Path, b6_model: str | Path,
                           output: str | Path, device: str = "cuda:0") -> str:
    if condition not in FORMAL_CASES:
        raise ValueError(f"unsupported formal condition: {condition}")
    args = [
        sys.executable, "-m", "quality_experiments.Q002_B6_roi_same_anchor_mse.run_q004_s640_k1_reconstruction",
        "--cropped-prepared-root", str(Path(prepared_root)), "--subject-id", subject,
        "--b6-model", str(Path(b6_model)), "--signal-simulator", str(Path(decoder)), "--output", str(Path(output)),
        "--device", device, "--experiment-id", Q004_EXPERIMENT_ID, "--anchor-batch-size", "640",
    ]
    return " ".join(shlex.quote(value) for value in args)


def run(args: argparse.Namespace) -> Path:
    manifest = _validate_case_manifest(args.subject, args.condition, args.sparse_manifest)
    prepared = select_prepared_root(args.subject, args.condition, args.l100_prepared_root, args.sparse_prepared_root, manifest)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    command = build_q004s640_command(args.condition, prepared, subject=args.subject, decoder=args.decoder, b6_model=args.b6_model, output=output, device=args.device)
    subprocess.run(command, shell=True, check=True)
    route_manifest = output / "experiment_manifest.json"
    if not route_manifest.is_file():
        raise FileNotFoundError(f"Q004 runner did not produce {route_manifest}")
    write_immutable_json(output / "phase2_case_manifest.json", {
        "schema": "phase4r_q004s640_subject_case/v2", "subject": args.subject, "condition": args.condition,
        "training_route": "Q004S640", "training_psf": {"samples": 1, "label": "K1"},
        "sparse_manifest": {"path": str(manifest.resolve()), "sha256": sha256(manifest)},
        "selected_prepared_root": str(prepared), "l100_r0_prepared_root": str(Path(args.l100_prepared_root).resolve()),
        "q004_experiment_manifest": {"path": str(route_manifest.resolve()), "sha256": sha256(route_manifest)},
    })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="CYJ"); parser.add_argument("--condition", required=True); parser.add_argument("--l100-prepared-root", required=True)
    parser.add_argument("--sparse-prepared-root", required=True); parser.add_argument("--sparse-manifest", required=True)
    parser.add_argument("--decoder", required=True); parser.add_argument("--b6-model", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest = _validate_case_manifest(args.subject, args.condition, args.sparse_manifest)
    prepared = select_prepared_root(args.subject, args.condition, args.l100_prepared_root, args.sparse_prepared_root, manifest)
    if not args.execute:
        print(build_q004s640_command(args.condition, prepared, subject=args.subject, decoder=args.decoder, b6_model=args.b6_model, output=args.output, device=args.device)); return
    print(run(args))


if __name__ == "__main__":
    main()
