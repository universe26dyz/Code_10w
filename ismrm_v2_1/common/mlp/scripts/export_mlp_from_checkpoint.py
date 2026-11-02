"""Checkpoint-only MLP export; this command never enters reconstruction training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset
from modules.module_05_signal_decoder.provenance import sha256_file
from modules.module_07_objective_training.mlp_trainer import build_mlp_training_model, load_mlp_reconstruction_checkpoint
from modules.module_07_objective_training.training_space import TrainingSpace
from modules.module_08_inference_export.export_quantitative import export_quantitative_outputs
from modules.module_08_inference_export.reprojection import export_native_plane_reprojections


_REQUIRED_CHECKPOINT_FIELDS = {
    "model_state", "resolved_config", "protocol_hhz_v1", "training_space", "trained_axisangle_train",
    "intensity_normalization", "source_mlp_checkpoint_path", "source_mlp_checkpoint_sha256",
    "source_mlp_checkpoint_metadata", "scientific_checkpoint",
}


def _canonical_protocol(record: Mapping[str, Any]) -> tuple[float, int, tuple[float, ...], tuple[float, ...], tuple[float, ...], int]:
    return (float(record["tr_ms"]), int(record["vps"]), tuple(float(value) for value in record["fa_deg"]), tuple(float(value) for value in record["ti_ms"]), tuple(float(value) for value in record["t2prep_ms"]), int(record["n_ramp_up"]))


def _prepared_observations(prepared_root: str | Path, subject_id: str, stack_names: Sequence[str]) -> list[Path]:
    observations = [Path(prepared_root) / subject_id / stack / "observations.npz" for stack in stack_names]
    missing = [str(path) for path in observations if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required prepared observations are missing: " + "; ".join(missing))
    return observations


def _validate_checkpoint_provenance(checkpoint: Mapping[str, Any]) -> Path:
    missing = _REQUIRED_CHECKPOINT_FIELDS.difference(checkpoint)
    if missing:
        raise ValueError(f"MLP reconstruction checkpoint is incomplete; missing: {sorted(missing)}")
    config = checkpoint["resolved_config"]
    if not isinstance(config, Mapping) or not isinstance(config.get("training"), Mapping) or not isinstance(config.get("decoder"), Mapping):
        raise ValueError("MLP reconstruction checkpoint resolved_config lacks training/decoder mappings.")
    normalization = checkpoint["intensity_normalization"]
    if not isinstance(normalization, Mapping) or normalization.get("method") != "trimmed_mean" or normalization.get("amplitude_export_units") != "original_input_intensity":
        raise ValueError("MLP reconstruction checkpoint lacks required intensity-normalization provenance.")
    if float(normalization.get("scale", 0.0)) <= 0:
        raise ValueError("MLP reconstruction checkpoint intensity normalization scale must be positive.")
    source_path = Path(str(checkpoint["source_mlp_checkpoint_path"]))
    if not source_path.is_file():
        raise FileNotFoundError(f"Frozen MLP provenance source is unavailable: {source_path}")
    if sha256_file(source_path) != checkpoint["source_mlp_checkpoint_sha256"]:
        raise ValueError("Frozen MLP checkpoint SHA256 does not match reconstruction provenance.")
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    source_metadata = checkpoint["source_mlp_checkpoint_metadata"]
    if not isinstance(source, Mapping) or not isinstance(source_metadata, Mapping):
        raise ValueError("Frozen MLP checkpoint provenance metadata must be mappings.")
    for key in ("scientific_checkpoint", "protocol_hhz_v1", "timing9_min_ms", "timing9_max_ms"):
        if key not in source or key not in source_metadata or source[key] != source_metadata[key]:
            raise ValueError(f"Frozen MLP {key} provenance no longer matches the reconstruction checkpoint.")
    if _canonical_protocol(source["protocol_hhz_v1"]) != _canonical_protocol(checkpoint["protocol_hhz_v1"]):
        raise ValueError("Frozen MLP protocol/TR/VPS provenance does not match the reconstruction checkpoint.")
    if bool(source["scientific_checkpoint"]) != bool(checkpoint["scientific_checkpoint"]):
        raise ValueError("Frozen MLP scientific_checkpoint flag does not match reconstruction provenance.")
    return source_path


def export_mlp_from_checkpoint(prepared_root: str | Path, subject_id: str, checkpoint_path: str | Path, output_dir: str | Path, *, output_resolution_mm: float | None = None, reproject: bool = False, stack_names: Sequence[str] = ("sax", "2ch", "4ch")) -> dict[str, str]:
    """Restore an MLP reconstruction checkpoint and write exports without training."""

    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("MLP reconstruction checkpoint must be a mapping.")
    source_path = _validate_checkpoint_provenance(checkpoint)
    config = dict(checkpoint["resolved_config"])
    config["decoder"] = dict(config["decoder"])
    config["decoder"]["checkpoint"] = str(source_path)
    device = torch.device(config["training"]["device"])
    observations = _prepared_observations(prepared_root, subject_id, stack_names)
    dataset = QuantPointDataset(observations, device=device)
    method_root = Path(__file__).resolve().parents[1]
    model, _, _, _ = build_mlp_training_model(dataset, config, method_root / "configs" / "protocol_hhz_v1.yaml", device)
    loaded = load_mlp_reconstruction_checkpoint(checkpoint_path, model, device)
    space = TrainingSpace.from_state_dict(loaded["training_space"])
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Checkpoint-only export output directory must be absent or empty.")
    output.mkdir(parents=True, exist_ok=True)
    export_cfg = config.get("export", {})
    if not isinstance(export_cfg, Mapping):
        raise ValueError("Checkpoint resolved_config.export must be a mapping.")
    resolution = float(export_cfg["output_resolution_mm"] if output_resolution_mm is None else output_resolution_mm)
    paths = export_quantitative_outputs(model, space, output, resolution, int(export_cfg["output_batch_size"]), dataset=dataset, export_config={"bbox": config.get("bbox", {}), **export_cfg})
    if reproject:
        paths.update(export_native_plane_reprojections(model, space, observations, output, output_psf=config.get("psf", {}).get("export", {})))
    provenance = {"reconstruction_checkpoint": str(Path(checkpoint_path).resolve()), "frozen_mlp_checkpoint": str(source_path.resolve()), "frozen_mlp_checkpoint_sha256": checkpoint["source_mlp_checkpoint_sha256"], "scientific_checkpoint": bool(checkpoint["scientific_checkpoint"]), "intensity_normalization": dict(checkpoint["intensity_normalization"]), "training_space_restored": True, "training_called": False}
    provenance_path = output / "checkpoint_export_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    paths["provenance"] = provenance_path
    return {key: str(value) for key, value in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", required=True); parser.add_argument("--subject-id", required=True)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--output-resolution", type=float); parser.add_argument("--reproject", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export_mlp_from_checkpoint(args.prepared_root, args.subject_id, args.checkpoint, args.output, output_resolution_mm=args.output_resolution, reproject=args.reproject), sort_keys=True))


if __name__ == "__main__":
    main()
