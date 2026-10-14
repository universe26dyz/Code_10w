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
from .build_geometry import apply_global_rigid_alignment, select_modal_geometry_templates
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


def _write_source_manifest(root: Path, inputs, gt, qc: dict) -> Path:
    path = root / "synthetic_source_manifest.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    geometry = gt.geometry
    manifest = {
        "phase": "1.2",
        "scientific_status": "candidate_until_geometry_acceptance",
        "source_family": "single_source_xcat_3d_quantitative",
        "legacy_invalid_route_forbidden": True,
        "xcat": {**inputs.manifest["xcat"], "raw_volumes": inputs.manifest["volumes"], "shape": list(geometry.shape), "spacing_mm": [1.0, 1.0, 1.0], "origin_mm": list(geometry.origin_mm), "axis_order": "xyz", "world_convention": "LPS"},
        "gt_contract": {"lv_myocardium_t1_ms": 1410.0, "lv_myocardium_t2_ms": 50.0, "b1": 1.0, "m0_policy": inputs.manifest["m0_policy"]},
        "geometry_gates": qc["geometry_gates"],
        "scientific_gt_geometry_gate": qc["scientific_gt_geometry_gate"],
        "external_xcat_input_gate": "PASS",
        "formal_synthetic_ready": False,
        "spatial_integration": {"method": "midpoint quadrature in signal space with linear interpolation", "inplane_samples_row_col": [6, 6], "thickness_samples": 9},
        "external_reference_provenance": {"hanson_2021_doi": "10.1002/mp.15036", "reference_implementation": "https://github.com/hmhanson/4DPhantomMRI", "reference_commit": "97558db05f4e344865f472de974b9f4d042a1a15"},
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
    manifest = _write_source_manifest(root, inputs, gt, qc)
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
    maps = np.load(root / "gt_maps.npz"); myo = maps["myocardium_mask"].astype(bool)
    indices = np.argwhere(myo); world = np.asarray(maps["origin_mm"], float)[None, :] + indices * np.asarray(maps["spacing_mm"], float)[None, :]
    center = world.mean(axis=0); eigenvalues, eigenvectors = np.linalg.eigh(np.cov((world - center).T)); long_axis = eigenvectors[:, np.argmax(eigenvalues)]
    if long_axis[np.argmax(np.abs(long_axis))] < 0: long_axis *= -1
    templates = select_modal_geometry_templates(prepared_root)
    aligned, alignment_qc = apply_global_rigid_alignment(templates, center, long_axis)
    alignment_qc.update({"lv_centroid_mm": center.tolist(), "lv_pca_eigenvalues": eigenvalues.tolist(), "lv_long_axis": long_axis.tolist(), "lv_alignment_gate": bool(abs(alignment_qc["sax_normal_dot_lv_long_axis"]) >= 0.999 and alignment_qc["max_pairwise_center_distance_error_mm"] <= 1e-6)})
    if not alignment_qc["lv_alignment_gate"]: raise ValueError("LV PCA geometry alignment gate failed")
    (root / "geometry_alignment_qc.json").write_text(json.dumps(alignment_qc, indent=2) + "\n")
    timing = np.asarray(selection["timing9_ms"], dtype=float)
    acquisition = acquisition_times_from_timing9(timing)
    summaries = {}
    for stack, groups in aligned.items():
        images, masks = render_groups(signal["signals"], signal["origin_mm"], signal["spacing_mm"], groups, inplane_samples=6, thickness_samples=9)
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
        from ..xcat3d.finalize import finalize_phase1_2
        result = {"phase1_2_final_manifest": str(finalize_phase1_2(args.output_root))}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
