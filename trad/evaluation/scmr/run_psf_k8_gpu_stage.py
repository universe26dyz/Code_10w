"""Server-only CUDA/tiny-cuda-nn Stage-1 PSF K=8 signal reprojection."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

_TRAD_ROOT = Path(__file__).resolve().parents[2]

from trad.modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset
from trad.modules.module_07_objective_training.trad_trainer import build_training_model, load_checkpoint
from trad.modules.module_07_objective_training.training_space import TrainingSpace
from trad.modules.module_08_inference_export.reprojection import export_native_plane_reprojections
from trad.third_party.nesvor.nesvor.inr import models as nesvor_models

STACKS = ("sax", "2ch", "4ch")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_TRAD_ROOT.parent, text=True).strip()


def _json_pose_difference(checkpoint: dict[str, Any], final_poses: Path) -> float:
    report = json.loads(final_poses.read_text(encoding="utf-8"))
    space = TrainingSpace.from_state_dict(checkpoint["training_space"])
    physical = space.axisangle_train_to_physical(checkpoint["trained_axisangle_train"]).detach().cpu().numpy()
    reported = np.asarray(report["axisangle_physical"], dtype=np.float32)
    if physical.shape != reported.shape:
        raise ValueError("final_rigid_poses.json shape differs from checkpoint pose tensor.")
    return float(np.max(np.abs(physical - reported)))


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stage 1 requires an available CUDA device and tiny-cuda-nn backend.")
    observations = [args.prepared_root / args.subject_id / stack / "observations.npz" for stack in STACKS]
    missing = [str(path) for path in observations if not path.is_file()]
    if missing or not args.checkpoint.is_file() or not args.config.is_file() or not args.final_poses.is_file():
        raise FileNotFoundError("Missing Stage-1 input: " + "; ".join(missing + [str(path) for path in (args.checkpoint, args.config, args.final_poses) if not path.is_file()]))
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty Stage-1 output: {output}")
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["training"] = dict(config["training"]); config["training"]["device"] = str(device)
    dataset = QuantPointDataset(observations, device=device)
    model, _, _ = build_training_model(dataset, config, _TRAD_ROOT / "configs" / "protocol_hhz_v1.yaml", device)
    if nesvor_models.USE_TORCH or not hasattr(model.inr.encoding, "params"):
        raise RuntimeError("Stage 1 did not instantiate the tiny-cuda-nn HashGrid backend.")
    checkpoint = load_checkpoint(args.checkpoint, model, device)
    trained = checkpoint["trained_axisangle_train"].detach().cpu()
    loaded = model.rigid_psf.axisangle.detach().cpu()
    if not torch.equal(loaded, trained):
        raise ValueError("Loaded model.rigid_psf.axisangle differs from checkpoint trained_axisangle_train.")
    space = TrainingSpace.from_state_dict(checkpoint["training_space"])
    output.mkdir(parents=True, exist_ok=False)
    signal_root = output / "signals"; signal_root.mkdir()
    export_native_plane_reprojections(model, space, observations, signal_root, output_psf={"enabled": True, "n_samples": 8}, evaluation_seed=20260921, export_parameter_maps=False)
    outputs = {}
    for stack in STACKS:
        path = signal_root / f"signal_reprojection_{stack}.npz"
        if not path.is_file():
            raise RuntimeError(f"Stage 1 did not create {path}")
        outputs[stack] = {"relative_path": str(path.relative_to(output)), "sha256": _sha256(path)}
    try:
        import tinycudann as tcnn
        tcnn_version = getattr(tcnn, "__version__", "installed")
    except ImportError as exc:  # pragma: no cover - guarded by backend assertion
        raise RuntimeError("tiny-cuda-nn import failed after backend construction.") from exc
    manifest = {
        "schema": "trad_psf_k8_stage1/v1", "status": "PASS", "repo_head": _git_head(), "subject_id": args.subject_id,
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": _sha256(args.checkpoint)},
        "config": {"path": str(args.config.resolve()), "sha256": _sha256(args.config)},
        "prepared_observations": {stack: {"path": str(path.resolve()), "sha256": _sha256(path)} for stack, path in zip(STACKS, observations)},
        "device": {"requested": str(device), "name": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda, "tinycudann": tcnn_version, "backend": "tiny-cuda-nn HashGrid"},
        "psf": {"enabled": True, "n_samples": 8, "seed": 20260921, "sampling_mode": "random Gaussian local PSF", "operation_order": "local PSF sample -> final Stage-B rigid pose -> INR -> Bloch signal per sample -> amplitude -> mean signal"},
        "final_pose": {"source": "model.pt model_state rigid_psf.axisangle / trained_axisangle_train", "sha256": _tensor_sha256(loaded), "shape": list(loaded.shape), "checkpoint_max_difference": float(torch.max(torch.abs(loaded - trained))), "json_physical_max_difference": _json_pose_difference(checkpoint, args.final_poses)},
        "signal_outputs": outputs,
    }
    (output / "stage1_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "TRANSFER_SHA256SUMS.txt").write_text("".join(f"{entry['sha256']}  {entry['relative_path']}\n" for entry in outputs.values()), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True); parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path); parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--final-poses", required=True, type=Path); parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), indent=2, sort_keys=True))
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
