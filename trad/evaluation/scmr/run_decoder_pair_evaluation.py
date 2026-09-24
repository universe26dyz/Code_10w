"""Read-only formal evaluation of Bloch and FrozenMLP PSF comparison artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .decoder_pair import (
    PARAMETERS,
    STACKS,
    evaluate_myocardium_metrics,
    evaluate_pair_metrics,
    load_decoder_pair,
    render_paired_all_slice_montages,
    render_paired_figure1,
    render_paired_figure2,
    render_paired_myocardium,
    select_representative_groups,
    validate_verified_reference,
)
from .quality_control import load_legacy_scmr_colorbars, write_range_audit
from .reference_2d import load_verified_reference, sha256


def _write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0]) if rows else ["subject_id"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _load_masks(bundle: str | Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    root = Path(bundle).expanduser().resolve()
    manifest_path, archive_path = root / "manifest.json", root / "sax_myocardium_masks.npz"
    if not manifest_path.is_file() or not archive_path.is_file():
        raise FileNotFoundError("Paired myocardium metrics require manifest.json and sax_myocardium_masks.npz.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "exact_native_myocardium_bundle/v2" or manifest.get("status") != "PASS":
        raise ValueError("Paired myocardium metrics require a PASS exact_native_myocardium_bundle/v2 bundle.")
    with np.load(archive_path, allow_pickle=False) as data:
        masks = {name: np.asarray(data[name], dtype=bool) for name in ("myocardium_core_1px", "myocardium_full", "myocardium_core_legacy")}
    return masks, {"root": str(root), "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path), "archive": str(archive_path), "archive_sha256": sha256(archive_path)}


def _range_loaded(artifact: Any) -> dict[str, dict[str, np.ndarray]]:
    return {
        stack: {
            "t1_ms": artifact.arrays["T1"][stack]["prediction"],
            "t2_ms": artifact.arrays["T2"][stack]["prediction"],
            "support": artifact.arrays["T1"][stack]["support"] & artifact.arrays["T2"][stack]["support"],
        }
        for stack in STACKS
    }


def _timing_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "status": "UNAVAILABLE"}
    data = json.loads(path.read_text(encoding="utf-8"))
    stages = data.get("run_level_ms", {})
    return {"path": str(path), "sha256": sha256(path), "status": "PRESENT", "decoder_type": data.get("decoder_type"), "optimization_total_ms": stages.get("optimization_total"), "total_runtime_ms": stages.get("total_runtime")}


def _timing_pair(trad_run: Path, mlp_run: Path) -> dict[str, Any]:
    values = {"Bloch": _timing_summary(trad_run / "timing_profile_summary.json"), "FrozenMLP": _timing_summary(mlp_run / "timing_profile_summary.json")}
    trad, mlp = values["Bloch"].get("optimization_total_ms"), values["FrozenMLP"].get("optimization_total_ms")
    values["optimization_speedup_bloch_over_frozen_mlp"] = float(trad) / float(mlp) if trad is not None and mlp not in (None, 0) else None
    total_trad, total_mlp = values["Bloch"].get("total_runtime_ms"), values["FrozenMLP"].get("total_runtime_ms")
    values["total_runtime_speedup_bloch_over_frozen_mlp"] = float(total_trad) / float(total_mlp) if total_trad is not None and total_mlp not in (None, 0) else None
    return values


def _git_commit() -> str:
    repo = Path(__file__).resolve().parents[3]
    result = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE"


def run(args: argparse.Namespace) -> Path:
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty decoder-pair evaluation output: {output}")
    pair = load_decoder_pair(args.subject_id, args.trad_eval, args.mlp_eval)
    reference = load_verified_reference(args.native_reference, args.subject_id, args.preprocessed_root)
    reference_sha = sha256(reference.manifest_path)
    validate_verified_reference(pair, reference, reference_sha)
    masks, mask_provenance = _load_masks(args.mask_bundle)
    metrics = evaluate_pair_metrics(pair)
    myocardium = evaluate_myocardium_metrics(pair, masks)
    selected = select_representative_groups(pair)
    trad_run, mlp_run = Path(args.trad_run).expanduser().resolve(), Path(args.mlp_run).expanduser().resolve()
    for run_root in (trad_run, mlp_run):
        for name in ("T1_3D.nii.gz", "T2_3D.nii.gz", "B1_3D.nii.gz", "amplitude_3D.nii.gz"):
            if not (run_root / name).is_file():
                raise FileNotFoundError(f"Decoder-pair evaluation requires existing run artifact: {run_root / name}")
    output.mkdir(parents=True, exist_ok=True)
    colorbars = load_legacy_scmr_colorbars(args.legacy_source)
    figure1 = render_paired_figure1(reference, trad_run, mlp_run, output / "figure1", colorbars, axis_name=args.plane_axis, plane_index=args.plane_index, reference_group=args.figure1_reference_group if args.figure1_reference_group is not None else selected["sax"])
    figure2 = render_paired_figure2(pair, selected, output / "figure2", colorbars)
    all_slices = render_paired_all_slice_montages(pair, metrics["paired_common_support"]["per_slice"], output / "all_slices", colorbars)
    myocardium_figures = render_paired_myocardium(pair, selected, masks, output / "myocardium", colorbars)
    range_audit = {
        "Bloch": write_range_audit(reference, _range_loaded(pair.trad), trad_run, output / "range_audit" / "trad", method_label="Bloch"),
        "FrozenMLP": write_range_audit(reference, _range_loaded(pair.mlp), mlp_run, output / "range_audit" / "mlp", method_label="FrozenMLP"),
    }
    range_summary = {
        method: json.loads(Path(details["summary_json"]).read_text(encoding="utf-8"))
        for method, details in range_audit.items()
    }
    _write_json(range_summary["Bloch"], output / "range_audit" / "trad_range_audit.json")
    _write_json(range_summary["FrozenMLP"], output / "range_audit" / "mlp_range_audit.json")
    _write_json({"definition": "method-neutral summaries of existing run exports; no map values were changed", "methods": range_summary}, output / "range_audit" / "pair_range_summary.json")
    timing = _timing_pair(trad_run, mlp_run)
    _write_csv(metrics["method_specific_per_slice"], output / "metrics" / "quantitative_per_slice.csv")
    _write_csv(metrics["method_specific_per_stack"], output / "metrics" / "quantitative_per_stack.csv")
    _write_json({"definition": metrics["definition"], "metrics": metrics["method_specific_global"]}, output / "metrics" / "quantitative_global.json")
    _write_json(myocardium, output / "metrics" / "myocardium_metrics.json")
    _write_json(metrics["paired_common_support"], output / "metrics" / "paired_common_support_metrics.json")
    manifest = {
        "schema": "decoder_pair_psf_postprocessing/v1",
        "status": "COMPLETE",
        "read_only_scope": "existing reconstruction volumes and existing map-domain PSF comparison artifacts only; no reconstruction or PSF reprojection was run",
        "subject_id": args.subject_id,
        "source_git_commit": _git_commit(),
        "trad_psf_evaluation": {"root": str(pair.trad.root), "manifest": str(pair.trad.manifest_path), "manifest_sha256": sha256(pair.trad.manifest_path)},
        "mlp_psf_evaluation": {"root": str(pair.mlp.root), "manifest": str(pair.mlp.manifest_path), "manifest_sha256": sha256(pair.mlp.manifest_path)},
        "trad_run": str(trad_run),
        "mlp_run": str(mlp_run),
        "native_reference": {"root": str(reference.root), "manifest": str(reference.manifest_path), "manifest_sha256": reference_sha, "verified_file_hashes": reference.file_hashes},
        "myocardium_bundle": mask_provenance,
        "metric_definitions": {"method_specific": "each method's existing map-domain common support", "paired_common_support": "Bloch existing common support AND FrozenMLP existing common support", "interpretation": metrics["definition"], "ssim": "SSIM is per native 2-D slice only; *_slice_mean and *_slice_median are descriptive slice aggregations, not voxel-pooled or volumetric SSIM."},
        "selected_representative_groups": selected,
        "figure1": figure1,
        "figure2": figure2,
        "all_slices": all_slices,
        "myocardium": myocardium_figures,
        "range_audit": range_audit,
        "timing_summary_sources": timing,
        "command": sys.argv,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(manifest, output / "metrics" / "decoder_pair_summary.json")
    (output / "README.md").write_text("# Decoder-pair PSF-matched evaluation\n\nFigure 1 is same-world-plane 3-D through-plane continuity QC. Figure 2 and `all_slices/` are map-domain PSF-matched native-plane comparisons. Quantitative values are consistency/agreement with verified native 2-D dictionary maps, not independent ground-truth accuracy.\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--trad-eval", required=True)
    parser.add_argument("--mlp-eval", required=True)
    parser.add_argument("--trad-run", required=True)
    parser.add_argument("--mlp-run", required=True)
    parser.add_argument("--native-reference", required=True)
    parser.add_argument("--preprocessed-root", required=True)
    parser.add_argument("--mask-bundle", required=True)
    parser.add_argument("--legacy-source", help="Optional external legacy asset override; defaults to vendored exact legacy LUTs.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--plane-axis", choices=("x", "y"), default="x")
    parser.add_argument("--plane-index", type=int)
    parser.add_argument("--figure1-reference-group", type=int)
    args = parser.parse_args()
    try:
        print(run(args))
    except (FileNotFoundError, ValueError, IndexError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
