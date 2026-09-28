"""Read-only final-checkpoint K=8 native weighted-signal evaluation for D2."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mlp.modules.module_07_objective_training.mlp_trainer import build_mlp_training_model, load_mlp_reconstruction_checkpoint
from quality_experiments.D2_k8_signal_domain.core import STACKS, aggregate, export_k8_reprojections, method_specific_slice_rows, paired_slice_rows, prepare_empty_output, sha256, write_csv, write_json
from quality_experiments.D2_k8_signal_domain.figures import render_metric_summaries, render_residual_montage
from reconstruction_core.orchestration import build_training_model, load_checkpoint
from trad.modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset
from trad.modules.module_05_signal_decoder.decoder_factory import bloch_decoder_factory
from trad.modules.module_07_objective_training.training_space import TrainingSpace


EXPERIMENT_ID = "D2_BASELINE_SIGNAL_K8_V1"
EVALUATION_SEED = 20260911


def resolve_prepared_inputs(prepared_root: str | Path, subject_id: str) -> list[Path]:
    root = Path(prepared_root).expanduser().resolve()
    nested = [root / subject_id / stack / "observations.npz" for stack in STACKS]
    flat = [root / stack / "observations.npz" for stack in STACKS]
    available = [candidate for candidate in (nested, flat) if all(path.is_file() for path in candidate)]
    if len(available) != 1:
        raise FileNotFoundError(f"D2 requires exactly one explicit prepared layout under {root}: {nested} or {flat}.")
    return available[0]


def _checkpoint_config(path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("resolved_config")
    if not isinstance(config, dict):
        raise ValueError(f"D2 checkpoint lacks resolved_config: {path}")
    return config


def load_final_model(method: str, checkpoint_path: str | Path, prepared_inputs: list[Path], protocol_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, TrainingSpace, dict[str, Any]]:
    """Rebuild from the stored config then restore only the final state dict; no optimizer exists here."""

    checkpoint_path = Path(checkpoint_path).resolve()
    config = _checkpoint_config(checkpoint_path, device)
    dataset = QuantPointDataset(prepared_inputs, device=device)
    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    initial = raw.get("training_space", {}).get("group_axisangle_init_physical")
    if initial is None:
        raise ValueError(f"D2 checkpoint lacks training-space initial poses: {checkpoint_path}")
    if method == "Bloch":
        model, _built_space, _protocol, decoder_metadata = build_training_model(dataset, config, protocol_path, device, initial_group_axisangle_physical=initial, decoder_factory=bloch_decoder_factory)
        checkpoint = load_checkpoint(checkpoint_path, model, device)
    elif method == "FrozenMLP":
        model, _built_space, _protocol, decoder_metadata = build_mlp_training_model(dataset, config, protocol_path, device, initial_group_axisangle_physical=initial)
        checkpoint = load_mlp_reconstruction_checkpoint(checkpoint_path, model, device)
    else:
        raise ValueError(f"Unsupported D2 decoder method: {method}")
    model.eval()
    decoder_record = {key: value for key, value in decoder_metadata.items() if isinstance(value, (str, int, float, bool))}
    return model, TrainingSpace.from_state_dict(checkpoint["training_space"]), {"decoder": decoder_record, "intensity_normalization": checkpoint["intensity_normalization"], "checkpoint_sha256": sha256(checkpoint_path), "source_checkpoint": str(checkpoint_path)}


def _git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _summary(method_global: list[dict[str, Any]], method_stack: list[dict[str, Any]], paired_global: list[dict[str, Any]], paired_stack: list[dict[str, Any]], paired_weight: list[dict[str, Any]], disagreement: dict[str, Any]) -> str:
    lines = ["# D2 formal K=8 signal-domain result summary", "", "Formal K=8 result generated: YES", ""]
    lines.extend(["## PRIMARY STRICT PAIRED", "", "All decoder-comparison values below use: Bloch valid AND FrozenMLP valid AND finite observed AND finite both predictions.", ""])
    for method in ("Bloch", "FrozenMLP"):
        global_row = next(row for row in paired_global if row["method"] == method)
        lines.append(f"- {method} global: RMSE_signal={global_row['RMSE_signal']:.4g}, NRMSE={global_row['NRMSE']:.4g}, NCC={global_row['NCC']:.4g}.")
        for stack in STACKS:
            row = next(item for item in paired_stack if item["method"] == method and item["stack"] == stack)
            lines.append(f"  - {stack.upper()}: RMSE_signal={row['RMSE_signal']:.4g}, NRMSE={row['NRMSE']:.4g}, NCC={row['NCC']:.4g}.")
        weights = [row for row in paired_weight if row["method"] == method]
        best, worst = min(weights, key=lambda row: row["RMSE_signal"]), max(weights, key=lambda row: row["RMSE_signal"])
        lines.append(f"  - best weight by RMSE={best['weight_idx']} ({best['RMSE_signal']:.4g}); worst weight={worst['weight_idx']} ({worst['RMSE_signal']:.4g}).")
    lines.extend(["", "## SECONDARY METHOD-SPECIFIC", "", "These diagnostics retain each method's own valid finite support and are not the primary decoder comparison.", ""])
    for method in ("Bloch", "FrozenMLP"):
        global_row = next(row for row in method_global if row["method"] == method)
        lines.append(f"- {method} global: RMSE_signal={global_row['RMSE_signal']:.4g}, NRMSE={global_row['NRMSE']:.4g}, NCC={global_row['NCC']:.4g}.")
        for stack in STACKS:
            row = next(item for item in method_stack if item["method"] == method and item["stack"] == stack)
            lines.append(f"  - {stack.upper()}: RMSE_signal={row['RMSE_signal']:.4g}, NRMSE={row['NRMSE']:.4g}, NCC={row['NCC']:.4g}.")
    lines.extend([f"", f"- Bloch-vs-FrozenMLP predicted-signal disagreement on strict paired support: RMSE_signal={disagreement['RMSE_signal']:.4g}, MAE_signal={disagreement['MAE_signal']:.4g}, Pearson_r={disagreement['Pearson_r']:.4g}, NCC={disagreement['NCC']:.4g}."])
    return "\n".join(lines) + "\n"


def materialize_metric_outputs(output: Path, artifact_paths: dict[str, dict[str, Path]]) -> dict[str, Any]:
    """Build/write all secondary and primary metric levels, then render strict-paired figures."""

    method_specific = method_specific_slice_rows("Bloch", artifact_paths["Bloch"]) + method_specific_slice_rows("FrozenMLP", artifact_paths["FrozenMLP"])
    paired_rows, disagreement_rows = paired_slice_rows(artifact_paths["Bloch"], artifact_paths["FrozenMLP"])
    method_stack_weight = aggregate(method_specific, ("method", "stack", "weight_idx", "support_provenance"))
    method_stack = aggregate(method_specific, ("method", "stack", "support_provenance"))
    method_weight = aggregate(method_specific, ("method", "weight_idx", "support_provenance"))
    method_global = aggregate(method_specific, ("method", "support_provenance"))
    paired_stack_weight = aggregate(paired_rows, ("method", "stack", "weight_idx", "support_provenance"))
    paired_stack = aggregate(paired_rows, ("method", "stack", "support_provenance"))
    paired_weight = aggregate(paired_rows, ("method", "weight_idx", "support_provenance"))
    paired_global = aggregate(paired_rows, ("method", "support_provenance"))
    disagreement_weight = aggregate(disagreement_rows, ("weight_idx", "support_provenance"))
    disagreement_global = aggregate(disagreement_rows, ("support_provenance",))[0]
    write_csv(output / "metrics/method_specific_per_stack_weight.csv", method_stack_weight)
    write_csv(output / "metrics/method_specific_per_stack.csv", method_stack)
    write_csv(output / "metrics/method_specific_per_weight.csv", method_weight)
    write_json(output / "metrics/method_specific_global.json", {"schema": "code10w_d2_signal_metrics/v1", "rows": method_global})
    write_csv(output / "metrics/paired_common_support_per_stack_weight.csv", paired_stack_weight)
    write_csv(output / "metrics/paired_common_support_per_stack.csv", paired_stack)
    write_csv(output / "metrics/paired_common_support_per_weight.csv", paired_weight)
    write_json(output / "metrics/paired_common_support_global.json", {"schema": "code10w_d2_strict_paired_signal_metrics/v1", "support_provenance": "Bloch valid AND FrozenMLP valid AND finite observed AND finite both predictions", "rows": paired_global})
    write_json(output / "metrics/bloch_vs_mlp_prediction.json", {"schema": "code10w_d2_prediction_disagreement/v1", "global": disagreement_global, "per_weight": disagreement_weight})
    render_metric_summaries(paired_weight, disagreement_weight, output / "figures/summary")
    render_residual_montage(artifact_paths, output / "figures/residual_by_weight")
    return {"method_specific": method_specific, "method_stack_weight": method_stack_weight, "method_stack": method_stack, "method_weight": method_weight, "method_global": method_global, "paired_rows": paired_rows, "paired_stack_weight": paired_stack_weight, "paired_stack": paired_stack, "paired_weight": paired_weight, "paired_global": paired_global, "disagreement_weight": disagreement_weight, "disagreement_global": disagreement_global}


def run(args: argparse.Namespace) -> Path:
    output = prepare_empty_output(args.output)
    for relative in ("artifacts/Bloch", "artifacts/FrozenMLP", "metrics", "figures/residual_by_weight", "figures/summary", "logs", "commands"):
        (output / relative).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("D2 requested CUDA but torch.cuda.is_available() is false.")
    prepared = resolve_prepared_inputs(args.prepared_root, args.subject_id)
    artifact_paths: dict[str, dict[str, Path]] = {}
    model_provenance: dict[str, Any] = {}
    for method, run_root in (("Bloch", Path(args.trad_run)), ("FrozenMLP", Path(args.mlp_run))):
        checkpoint = run_root / "model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"D2 final checkpoint is missing: {checkpoint}")
        model, space, provenance = load_final_model(method, checkpoint, prepared, args.protocol, device)
        artifact_paths[method] = export_k8_reprojections(model, space, prepared, output / "artifacts" / method, evaluation_seed=EVALUATION_SEED)
        model_provenance[method] = provenance
    summaries = materialize_metric_outputs(output, artifact_paths)
    manifest = {"schema": "code10w_d2_k8_signal_domain/v1", "experiment_id": EXPERIMENT_ID, "status": "COMPLETE", "subject_id": args.subject_id, "psf_samples": 8, "evaluation_seed": EVALUATION_SEED, "analysis_git_commit": _git_commit(), "baseline_read_only": True, "reconstruction_rerun": False, "prepared_inputs": [{"path": str(path), "sha256": sha256(path)} for path in prepared], "protocol_provenance": str(Path(args.protocol).resolve()), "decoders": model_provenance, "artifacts": {method: {stack: {"path": str(path.relative_to(output)), "schema": "observed,predicted,residual,group_idx,weight_idx,timing9_ms,masks,tr_ms,vps", "psf_samples": 8, "evaluation_seed": EVALUATION_SEED} for stack, path in stacks.items()} for method, stacks in artifact_paths.items()}}
    write_json(output / "manifest.json", manifest)
    (output / "README.md").write_text("# D2 formal K=8 signal-domain output\n\nRead-only final-checkpoint inference. All new K=8 artifacts are distinct from historical K=1 exports.\n", encoding="utf-8")
    (output / "RESULT_SUMMARY.md").write_text(_summary(summaries["method_global"], summaries["method_stack"], summaries["paired_global"], summaries["paired_stack"], summaries["paired_weight"], summaries["disagreement_global"]), encoding="utf-8")
    (output / "logs/run.log").write_text("D2 formal K=8 signal-domain evaluation completed without reconstruction optimization.\n", encoding="utf-8")
    (output / "commands/run_command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True); parser.add_argument("--trad-run", required=True); parser.add_argument("--mlp-run", required=True)
    parser.add_argument("--prepared-root", required=True); parser.add_argument("--protocol", default="trad/configs/protocol_hhz_v1.yaml")
    parser.add_argument("--output", required=True); parser.add_argument("--device", default="cuda:0")
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
