"""Staged Phase-1.2 synthetic route built only from one full-3-D XCAT geometry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..xcat3d.build_quantitative_gt import build_quantitative_gt, write_quantitative_gt
from ..xcat3d.prepare_xcat_parameters import prepare_xcat_parameter_hand_off
from ..xcat3d.read_xcat_volume import load_xcat_inputs
from ..xcat3d.validate_quantitative_gt import assert_phase1_2_manifest_ready, validate_quantitative_gt, write_validation
from .build_geometry import recenter_geometry, select_modal_geometry_templates
from .render_observations import render_groups
from .select_timing_template import choose_timing_template, collect_manifest_candidates
from .simulate_hhz_signals import generate_signal_field
from .write_observations_npz import write_observations
from trad.modules.module_02_data_bridge.timing import compute_duration_before_acq


DEFAULT_OUTPUT_ROOT = Path("paper_experiments/joint_vs_2dfit/simulation/outputs/phase1_2_xcat3d_clean")


def acquisition_times_from_timing9(timing9_ms: np.ndarray, tr_ms: float = 2.61, vps: int = 87) -> np.ndarray:
    timing9 = np.asarray(timing9_ms, dtype=float)
    if timing9.shape != (9,):
        raise ValueError("timing9_ms must contain nine values")
    intervals = timing9 + float(tr_ms) * int(vps)
    intervals[[3, 6, 7, 8]] += np.array([150.0, 35.0, 45.0, 55.0])
    acquisition = np.concatenate(([0.0], np.cumsum(intervals)))
    recovered = compute_duration_before_acq(acquisition, tr_ms, vps, np.array([50.0, 150.0]), np.array([35.0, 45.0, 55.0]))[1:]
    if not np.allclose(recovered, timing9, rtol=0.0, atol=1e-9):
        raise AssertionError("synthetic acquisition_time_ms does not round-trip to timing9_ms")
    return acquisition


def _write_source_manifest(root: Path, inputs, qc: dict) -> Path:
    path = root / "synthetic_source_manifest.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    geometry = inputs.t1_ms.geometry
    manifest = {
        "phase": "1.2",
        "scientific_status": "candidate_until_geometry_acceptance",
        "source_family": "single_source_xcat_3d_quantitative",
        "legacy_invalid_route_forbidden": True,
        "xcat": {**inputs.manifest["xcat"], "shape": list(geometry.shape), "spacing_mm": list(geometry.spacing_mm), "origin_mm": list(geometry.origin_mm), "axis_order": geometry.axis_order, "world_convention": geometry.world_convention},
        "gt_contract": {"lv_myocardium_t1_ms": 1410.0, "lv_myocardium_t2_ms": 50.0, "b1": 1.0, "m0_policy": inputs.manifest["m0_policy"]},
        "geometry_gates": qc["geometry_gates"],
        "scientific_gt_geometry_gate": qc["scientific_gt_geometry_gate"],
        "external_xcat_input_gate": "PASS",
        "formal_synthetic_ready": False,
        "external_reference_provenance": {"hanson_2021_doi": "10.1002/mp.15036", "reference_implementation": "https://github.com/hmhanson/4DPhantomMRI", "mrxcat_audit_commit": "9f396a998f435525b234a304c502238ad5955fb2"},
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def build_gt(xcat_input_manifest: str | Path, output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> dict:
    inputs = load_xcat_inputs(xcat_input_manifest)
    gt = build_quantitative_gt(inputs)
    qc = validate_quantitative_gt(gt)
    if not qc["scientific_gt_geometry_gate"]:
        raise ValueError("Phase-1.2 geometry acceptance failed")
    root = Path(output_root)
    gt_path = write_quantitative_gt(gt, root)
    qc_path = write_validation(qc, root)
    manifest = _write_source_manifest(root, inputs, qc)
    return {"gt_maps": str(gt_path), "geometry_qc": str(qc_path), "synthetic_source_manifest": str(manifest), "scientific_gt_geometry_gate": True}


def simulate_signal_field(output_root: str | Path, prepared_root: str | Path, approved_decoder: str | Path) -> dict:
    root = Path(output_root)
    manifest = assert_phase1_2_manifest_ready(root / "synthetic_source_manifest.json")
    output = root / "clean_signal_field.npz"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    maps = np.load(root / "gt_maps.npz")
    approved = json.loads(Path(approved_decoder).read_text())
    selection = choose_timing_template(collect_manifest_candidates(prepared_root), approved)
    timing = np.asarray(selection["timing9_ms"], dtype=float)
    signals = generate_signal_field(maps["t1_ms"], maps["t2_ms"], maps["b1"], timing, signal_support_mask=maps["signal_support_mask"])
    signals *= maps["m0"][..., None]
    if not np.array_equal(signals[~maps["signal_support_mask"]], np.zeros((int((~maps["signal_support_mask"]).sum()), 10), dtype=np.float32)):
        raise AssertionError("background signal is not exactly zero")
    np.savez_compressed(output, signals=signals, origin_mm=maps["origin_mm"], spacing_mm=maps["spacing_mm"], signal_support_mask=maps["signal_support_mask"])
    (root / "timing_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    manifest["signal_field_generated"] = True
    (root / "signal_field_status.json").write_text(json.dumps({"generated": True, "timing_selection": selection}, indent=2) + "\n")
    return {"clean_signal_field": str(output)}


def render_observations(output_root: str | Path, prepared_root: str | Path) -> dict:
    root = Path(output_root)
    assert_phase1_2_manifest_ready(root / "synthetic_source_manifest.json")
    signal = np.load(root / "clean_signal_field.npz")
    selection = json.loads((root / "timing_selection.json").read_text())
    templates = select_modal_geometry_templates(prepared_root)
    center = np.asarray(signal["origin_mm"], dtype=float) + np.asarray(signal["spacing_mm"], dtype=float) * (np.asarray(signal["signals"].shape[:3]) - 1) / 2.0
    timing = np.asarray(selection["timing9_ms"], dtype=float)
    acquisition = acquisition_times_from_timing9(timing)
    summaries = {}
    for stack, template in templates.items():
        groups = recenter_geometry(template, center)
        images, masks = render_groups(signal["signals"], signal["origin_mm"], signal["spacing_mm"], groups)
        metadata = {"group_idx": np.repeat(np.arange(len(groups)), 10), "weight_idx": np.tile(np.arange(10), len(groups)), "stack_idx": np.zeros(len(groups) * 10, dtype=np.int64), "acquisition_time_ms": np.tile(acquisition, len(groups)), "timing9_ms": np.repeat(timing[None, :], len(groups) * 10, axis=0), "affine_lps_rc": np.repeat(np.asarray([item["affine_lps_rc"] for item in groups]), 10, axis=0), "pixel_spacing_rc_mm": np.concatenate([np.repeat([item["pixel_spacing_rc_mm"]], 10, axis=0) for item in groups]), "slice_thickness_mm": np.concatenate([np.repeat([item["slice_thickness_mm"]], 10) for item in groups]), "tr_ms": np.repeat([2.61], len(groups) * 10), "vps": np.repeat([87], len(groups) * 10)}
        result = write_observations(root / stack, images, masks, metadata, {"phase": "1.2", "stack": stack, "signal_field": str(root / "clean_signal_field.npz")})
        summaries[stack] = result
    return {"stacks": summaries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prepare-xcat", "validate-xcat-inputs", "build-gt", "simulate-signal-field", "render-observations", "validate-final"))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--xcat-input-manifest")
    parser.add_argument("--prepared-root")
    parser.add_argument("--approved-decoder")
    args = parser.parse_args()
    if args.stage == "prepare-xcat":
        result = prepare_xcat_parameter_hand_off(args.output_root)
    elif args.stage == "validate-xcat-inputs":
        if not args.xcat_input_manifest:
            parser.error("--xcat-input-manifest is required")
        result = validate_quantitative_gt(build_quantitative_gt(load_xcat_inputs(args.xcat_input_manifest)))
    elif args.stage == "build-gt":
        if not args.xcat_input_manifest:
            parser.error("--xcat-input-manifest is required")
        result = build_gt(args.xcat_input_manifest, args.output_root)
    elif args.stage == "simulate-signal-field":
        if not args.prepared_root or not args.approved_decoder:
            parser.error("--prepared-root and --approved-decoder are required")
        result = simulate_signal_field(args.output_root, args.prepared_root, args.approved_decoder)
    elif args.stage == "render-observations":
        if not args.prepared_root:
            parser.error("--prepared-root is required")
        result = render_observations(args.output_root, args.prepared_root)
    else:
        result = assert_phase1_2_manifest_ready(Path(args.output_root) / "synthetic_source_manifest.json")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
