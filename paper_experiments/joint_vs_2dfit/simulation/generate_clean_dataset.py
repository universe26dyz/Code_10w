"""Generate the Phase 1 clean MRXCAT-backed prepared-observation candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from ..mrxcat.convert_labels import load_vti_array
from ..mrxcat.define_gt_tissue_maps import build_gt_maps, save_gt_maps
from ..mrxcat.inspect_case import inspect_repo
from .build_geometry import recenter_geometry, select_modal_geometry_templates
from .render_observations import integrate_signal_volume, render_groups
from .select_timing_template import choose_timing_template, collect_manifest_candidates
from .simulate_hhz_signals import generate_signal_field
from .write_observations_npz import write_observations


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def generate(
    mrxcat_repo: str | Path,
    prepared_root: str | Path,
    approved_decoder: str | Path,
    output_root: str | Path,
    matlab_parity: dict | None = None,
    inplane_samples: int = 2,
    thickness_samples: int = 3,
) -> dict:
    repo = Path(mrxcat_repo)
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    inspection = inspect_repo(repo)
    vti_path = "inpuData/Background/XCAT_vti/Image_01.vti"
    labels, info = load_vti_array(vti_path, "labels", git_repo=repo)
    cardiac = np.isin(labels, [1, 2, 5, 6, 7, 8])
    coords = np.argwhere(cardiac)
    if coords.size == 0:
        raise ValueError("MRXCAT example contains no LV/RV tissue labels")
    lo = np.maximum(coords.min(axis=0) - np.array([16, 16, 1]), 0)
    hi = np.minimum(coords.max(axis=0) + np.array([17, 17, 2]), np.asarray(labels.shape))
    labels = labels[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    # MRXCAT/XCAT VTI coordinates are metres; project geometry is DICOM LPS mm.
    origin_mm = np.asarray(info.origin) * 1000.0 + np.asarray(info.spacing) * 1000.0 * lo
    spacing_mm = np.asarray(info.spacing) * 1000.0
    maps = build_gt_maps(labels)
    gt_path = output_root / "gt_maps.npz"
    save_gt_maps(maps, str(gt_path))

    approved = json.loads(Path(approved_decoder).read_text())
    selection = choose_timing_template(collect_manifest_candidates(prepared_root), approved)
    (output_root / "timing_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    geometry_templates = select_modal_geometry_templates(prepared_root)
    (output_root / "geometry_templates.json").write_text(json.dumps(geometry_templates, indent=2) + "\n")

    signals = generate_signal_field(maps.t1_ms, maps.t2_ms, maps.b1, np.asarray(selection["timing9_ms"]), tr_ms=2.61, vps=87)
    np.savez_compressed(output_root / "clean_signal_field.npz", signals=signals, origin_mm=origin_mm, spacing_mm=spacing_mm)
    source_manifest = {
        "phase": "1",
        "acquisition_date_utc": "2026-10-12",
        "mrxcat": inspection.to_dict(),
        "mrxcat_commit": inspection.commit,
        "mrxcat_label_object": subprocess.check_output(["git", "-C", str(repo), "rev-parse", f"HEAD:{vti_path}"], text=True).strip(),
        "mrxcat_label_sha256": hashlib.sha256(subprocess.check_output(["git", "-C", str(repo), "cat-file", "blob", f"HEAD:{vti_path}"])).hexdigest(),
        "source_vti_path": str(repo / vti_path),
        "example_case_paths": {
            "3d_xcat_labels": vti_path,
            "lv_mesh": "inpuData/Patient1/Mesh.vtk",
            "provided_tissue_property_example": "outputData/Patient1/Tissue_properties_initial_frame/T1.vti",
            "repository_readme": "README.md",
            "license_and_use_notice": "MRXCAT_mri/README.md",
        },
        "source_vti_units": info.source_units,
        "crop_min_voxel": lo.tolist(),
        "crop_max_exclusive_voxel": hi.tolist(),
        "crop_shape": list(labels.shape),
        "gt_contract": {"m0": 1.0, "pd_scaling": False, "primary_myocardium_t1_ms": 1410.0, "primary_myocardium_t2_ms": 50.0, "b1": 1.0},
        "timing_selection": selection,
        "geometry_templates": geometry_templates,
        "spatial_integration": {"inplane_samples": inplane_samples, "thickness_samples": thickness_samples, "rule": "midpoint quadrature in signal space; linear signal interpolation"},
        "forbidden_routes": {"frozen_mlp": False, "noise": False, "motion": False, "sparse": False, "reconstruction": False},
        "matlab_parity": matlab_parity,
    }
    (output_root / "synthetic_source_manifest.json").write_text(json.dumps(source_manifest, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x) + "\n")

    summary = {"stacks": {}, "timing_selection": selection, "source_manifest": str(output_root / "synthetic_source_manifest.json")}
    center = origin_mm + spacing_mm * (np.asarray(labels.shape, dtype=float) - 1) / 2.0
    for stack in ("sax", "2ch", "4ch"):
        groups = recenter_geometry(geometry_templates[stack], center)
        images, masks = render_groups(signals, origin_mm, spacing_mm, groups, inplane_samples, thickness_samples)
        # Representative-pixel convergence check uses a finer through-slice grid.
        rep = groups[min(len(groups) // 2, len(groups) - 1)]
        fine = integrate_signal_volume(signals, origin_mm, spacing_mm, np.asarray(rep["affine_lps_rc"]), rep["image_shape_rows_cols"][0], rep["image_shape_rows_cols"][1], inplane_samples, max(thickness_samples + 2, 5))
        coarse = images[min(len(groups) // 2, len(groups) - 1) * 10 : min(len(groups) // 2, len(groups) - 1) * 10 + 10]
        convergence = float(np.max(np.abs(fine - np.moveaxis(coarse, 0, -1))))
        metadata = {field: np.concatenate([np.repeat([g[field]], 10, axis=0) if np.asarray(g[field]).ndim else np.repeat([g[field]], 10) for g in groups]) for field in ("pixel_spacing_rc_mm", "slice_thickness_mm")}
        metadata.update({
            "group_idx": np.repeat(np.arange(len(groups)), 10),
            "weight_idx": np.tile(np.arange(10), len(groups)),
            "stack_idx": np.zeros(len(groups) * 10, dtype=np.int64),
            "acquisition_time_ms": np.zeros(len(groups) * 10, dtype=float),
            "timing9_ms": np.repeat(np.asarray(selection["timing9_ms"])[None, :], len(groups) * 10, axis=0),
            "affine_lps_rc": np.repeat(np.asarray([g["affine_lps_rc"] for g in groups]), 10, axis=0),
            "tr_ms": np.repeat([2.61], len(groups) * 10),
            "vps": np.repeat([87], len(groups) * 10),
        })
        stack_dir = output_root / stack
        result = write_observations(stack_dir, images, masks, metadata, {"stack": stack, "geometry_source": geometry_templates[stack], "convergence_max_abs": convergence, "signal_field": str(output_root / "clean_signal_field.npz")})
        summary["stacks"][stack] = {"result": result, "convergence_max_abs": convergence, "group_count": len(groups), "image_shape": list(images.shape)}
    (output_root / "qc_summary.json").write_text(json.dumps(summary, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mrxcat-repo", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--approved-decoder", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.mrxcat_repo, args.prepared_root, args.approved_decoder, args.output_root), indent=2))


if __name__ == "__main__":
    main()
