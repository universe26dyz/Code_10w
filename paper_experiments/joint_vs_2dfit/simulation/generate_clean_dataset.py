"""Generate the Phase 1 clean MRXCAT-backed prepared-observation candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from ..mrxcat.convert_labels import load_vti_array
from ..mrxcat.define_gt_tissue_maps import build_gt_maps, save_gt_maps
from ..mrxcat.inspect_case import inspect_repo
from ..mrxcat.mesh_voxelize import load_tetra_mesh, mesh_axes, voxelize_tetrahedra
from .build_geometry import apply_global_rigid_alignment, recenter_geometry, select_modal_geometry_templates
from .render_observations import integrate_signal_volume, render_groups
from .select_timing_template import choose_timing_template, collect_manifest_candidates
from .simulate_hhz_signals import generate_signal_field
from .write_observations_npz import write_observations
from trad.modules.module_02_data_bridge.timing import compute_duration_before_acq


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _progress(message: str) -> None:
    print(f"phase1_1: {message}", flush=True, file=sys.stderr)


def acquisition_times_from_timing9(timing9_ms: np.ndarray, tr_ms: float, vps: int) -> np.ndarray:
    """Construct coherent HHZ acquisition timestamps that round-trip to timing9."""

    timing9 = np.asarray(timing9_ms, dtype=float)
    if timing9.shape != (9,):
        raise ValueError("timing9_ms must contain nine values")
    intervals = timing9 + float(tr_ms) * int(vps)
    intervals[[3, 6, 7, 8]] += np.array([150.0, 35.0, 45.0, 55.0])
    acq = np.concatenate(([0.0], np.cumsum(intervals)))
    recovered = compute_duration_before_acq(acq, tr_ms, vps, np.array([50.0, 150.0]), np.array([35.0, 45.0, 55.0]))[1:]
    if not np.allclose(recovered, timing9, rtol=0.0, atol=1e-9):
        raise AssertionError("synthetic acquisition_time_ms does not round-trip to timing9_ms")
    return acq


def _resample_labels_nearest(labels: np.ndarray, source_origin_mm: np.ndarray, source_spacing_mm: np.ndarray, origin_mm: np.ndarray, spacing_mm: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    grid = np.indices(shape, dtype=float).reshape(3, -1).T
    world = origin_mm[None, :] + grid * spacing_mm[None, :]
    coords = ((world - source_origin_mm[None, :]) / source_spacing_mm[None, :]).T
    values = map_coordinates(labels.astype(np.float32), coords, order=0, mode="constant", cval=0.0, prefilter=False)
    return values.reshape(shape).astype(np.int16)


def _convergence_metrics(value: np.ndarray, reference: np.ndarray) -> dict:
    delta = np.asarray(value, dtype=float) - np.asarray(reference, dtype=float)
    ref = np.asarray(reference, dtype=float)
    scale = float(np.max(np.abs(ref)))
    l2 = float(np.linalg.norm(ref.ravel()))
    value_l2 = float(np.linalg.norm(value.ravel()))
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "normalized_max_abs": float(np.max(np.abs(delta)) / scale) if scale else 0.0,
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "normalized_rmse": float(np.sqrt(np.mean(delta**2)) / scale) if scale else 0.0,
        "relative_l2": float(np.linalg.norm(delta.ravel()) / l2) if l2 else 0.0,
        "cosine_error": float(1.0 - np.dot(value.ravel(), ref.ravel()) / (value_l2 * l2)) if value_l2 and l2 else 0.0,
    }


def _signal_support_mapping() -> list[dict]:
    return [
        {"label_ids": [0, 15, 16], "meaning": "XCAT air/background", "source": "myTexturizer/phantomModel.py", "support": False, "m0": 0.0, "t1_t2_b1": None},
        {"label_ids": [1], "meaning": "LV myocardium", "source": "MakePhantom.py/myLabels.LV_wall + Mesh.vtk", "support": True, "m0": 1.0, "t1_t2_b1": [1410.0, 50.0, 1.0]},
        {"label_ids": [2, 3, 4, 10], "meaning": "XCAT muscle", "source": "myTexturizer/phantomModel.py", "support": True, "m0": 1.0, "t1_t2_b1": [1000.0, 50.0, 1.0]},
        {"label_ids": [5, 6, 7, 8], "meaning": "XCAT blood", "source": "myTexturizer/phantomModel.py", "support": True, "m0": 1.0, "t1_t2_b1": [1500.0, 120.0, 1.0]},
        {"label_ids": [13, 40, 41, 42, 43, 52], "meaning": "XCAT liver", "source": "myTexturizer/phantomModel.py", "support": True, "m0": 1.0, "t1_t2_b1": [661.0, 57.0, 1.0]},
        {"label_ids": [50, 99], "meaning": "XCAT fat", "source": "myTexturizer/phantomModel.py", "support": True, "m0": 1.0, "t1_t2_b1": [350.0, 70.0, 1.0]},
        {"label_ids": [31, 32, 33, 34, 35, 51], "meaning": "XCAT bone", "source": "myTexturizer/phantomModel.py", "support": True, "m0": 1.0, "t1_t2_b1": [250.0, 20.0, 1.0]},
        {"label_ids": "all other non-air labels", "meaning": "XCAT unknown tissue", "source": "myTexturizer/phantomModel.py", "support": True, "m0": 1.0, "t1_t2_b1": [750.0, 60.0, 1.0]},
    ]


def _phase1_1_metadata(groups: list[dict], timing9: np.ndarray, acquisition_time_ms: np.ndarray) -> dict:
    metadata = {field: np.concatenate([np.repeat([g[field]], 10, axis=0) if np.asarray(g[field]).ndim else np.repeat([g[field]], 10) for g in groups]) for field in ("pixel_spacing_rc_mm", "slice_thickness_mm")}
    metadata.update({
        "group_idx": np.repeat(np.arange(len(groups)), 10), "weight_idx": np.tile(np.arange(10), len(groups)),
        "stack_idx": np.zeros(len(groups) * 10, dtype=np.int64), "acquisition_time_ms": np.tile(acquisition_time_ms, len(groups)),
        "timing9_ms": np.repeat(timing9[None, :], len(groups) * 10, axis=0),
        "affine_lps_rc": np.repeat(np.asarray([g["affine_lps_rc"] for g in groups]), 10, axis=0),
        "tr_ms": np.repeat([2.61], len(groups) * 10), "vps": np.repeat([87], len(groups) * 10),
    })
    return metadata


def render_phase1_1_stack(mrxcat_repo: str | Path, output_root: str | Path, stack: str) -> dict:
    """Render one already-prepared Phase-1.1 stack without overwriting it."""

    if stack not in {"sax", "2ch", "4ch"}:
        raise ValueError("stack must be sax, 2ch, or 4ch")
    repo, output_root = Path(mrxcat_repo), Path(output_root)
    signal = np.load(output_root / "clean_signal_field.npz")
    selection = json.loads((output_root / "timing_selection.json").read_text())
    templates = json.loads((output_root / "geometry_templates.json").read_text())
    spatial_qc = json.loads((output_root / "spatial_integration_convergence_qc.json").read_text())
    mesh = load_tetra_mesh("inpuData/Patient1/Mesh.vtk", git_repo=repo)
    center, long_axis, _ = mesh_axes(mesh)
    aligned, alignment_qc = apply_global_rigid_alignment(templates, center, long_axis)
    accepted = spatial_qc[stack]["accepted_resolution"]
    timing9 = np.asarray(selection["timing9_ms"], dtype=float)
    acq = acquisition_times_from_timing9(timing9, 2.61, 87)
    groups = aligned[stack]
    _progress(f"render and write observations: {stack}")
    images, masks = render_groups(signal["signals"], signal["origin_mm"], signal["spacing_mm"], groups, int(accepted["inplane_samples_row_col"][0]), int(accepted["thickness_samples"]))
    result = write_observations(output_root / stack, images, masks, _phase1_1_metadata(groups, timing9, acq), {"phase": "1.1", "stack": stack, "signal_field": str(output_root / "clean_signal_field.npz"), "global_alignment": alignment_qc, "spatial_integration": accepted})
    return {"result": result, "group_count": len(groups), "image_shape": list(images.shape), "foreground_mask_fraction": float(masks.mean())}


def finalize_phase1_1(mrxcat_repo: str | Path, output_root: str | Path) -> dict:
    """Write final Phase-1.1 provenance only after all non-overwritten stacks exist."""

    repo, output_root = Path(mrxcat_repo), Path(output_root)
    if (output_root / "synthetic_source_manifest.json").exists():
        raise FileExistsError("refusing to overwrite completed Phase-1.1 provenance")
    missing = [stack for stack in ("sax", "2ch", "4ch") if not (output_root / stack / "observations.npz").exists()]
    if missing:
        raise FileNotFoundError(f"cannot finalize before stack observations exist: {missing}")
    inspection = inspect_repo(repo)
    vti_path, mesh_path = "inpuData/Background/XCAT_vti/Image_01.vti", "inpuData/Patient1/Mesh.vtk"
    selection = json.loads((output_root / "timing_selection.json").read_text())
    templates = json.loads((output_root / "geometry_templates.json").read_text())
    alignment_qc = json.loads((output_root / "geometry_alignment_qc.json").read_text())
    spatial_qc = json.loads((output_root / "spatial_integration_convergence_qc.json").read_text())
    parity_path = output_root / "matlab_parity.json"
    parity = json.loads(parity_path.read_text()) if parity_path.exists() else {"status": "unresolved_external_gate", "reason": "MATLAB parity was not run"}
    mesh_data = np.load(output_root / "mesh_primary_myo.npz")
    timing9 = np.asarray(selection["timing9_ms"], dtype=float)
    acq = acquisition_times_from_timing9(timing9, 2.61, 87)
    chosen = spatial_qc["sax"]["accepted_resolution"]
    stack_summaries = {}
    for stack in ("sax", "2ch", "4ch"):
        data = np.load(output_root / stack / "observations.npz")
        stack_summaries[stack] = {"group_count": int(data["images"].shape[0] // 10), "image_shape": list(data["images"].shape), "foreground_mask_fraction": float(data["masks"].mean())}
    source_manifest = {
        "phase": "1.1", "acquisition_date_utc": "2026-10-12", "supersedes_candidate": "simulation/outputs/phase1_clean_mrxcat (retained unchanged)",
        "mrxcat": inspection.to_dict(), "mrxcat_commit": inspection.commit,
        "mrxcat_label_object": subprocess.check_output(["git", "-C", str(repo), "rev-parse", f"HEAD:{vti_path}"], text=True).strip(), "mrxcat_label_sha256": hashlib.sha256(subprocess.check_output(["git", "-C", str(repo), "cat-file", "blob", f"HEAD:{vti_path}"])).hexdigest(),
        "mrxcat_mesh_object": subprocess.check_output(["git", "-C", str(repo), "rev-parse", f"HEAD:{mesh_path}"], text=True).strip(), "mrxcat_mesh_sha256": hashlib.sha256(subprocess.check_output(["git", "-C", str(repo), "cat-file", "blob", f"HEAD:{mesh_path}"])).hexdigest(),
        "mesh_primary_myocardium": {"path": "mesh_primary_myo.npz", "coordinate_units": "mm", "source_mesh_units": "m", "point_count": int(mesh_data["point_count"]), "tetra_count": int(mesh_data["tetra_count"]), "isotropic_spacing_mm": 1.0, "voxelization": "exact tetra barycentric containment", "blood_pool": "coarse supplied XCAT VTI nearest-neighbour resample (surrounding nuisance only)"},
        "signal_support_mapping": _signal_support_mapping(), "gt_contract": {"m0": "1.0 on signal_support_mask, 0.0 outside", "pd_scaling": False, "primary_myocardium_t1_ms": 1410.0, "primary_myocardium_t2_ms": 50.0, "b1": 1.0},
        "timing_selection": selection, "acquisition_time_ms": {"values": acq.tolist(), "derivation": "selected timing9 + TR*VPS + original HHZ TI/T2prep intervals", "round_trip_verified": True},
        "geometry_templates": templates, "geometry_alignment_qc": alignment_qc,
        "spatial_integration": {"accepted": chosen, "convergence_qc": "spatial_integration_convergence_qc.json", "rule": "midpoint quadrature in signal space; linear signal interpolation"},
        "forbidden_routes": {"frozen_mlp": False, "noise": False, "motion": False, "sparse": False, "reconstruction": False}, "matlab_parity": parity,
    }
    (output_root / "synthetic_source_manifest.json").write_text(json.dumps(source_manifest, indent=2) + "\n")
    summary = {"stacks": stack_summaries, "timing_selection": selection, "source_manifest": str(output_root / "synthetic_source_manifest.json")}
    (output_root / "qc_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _progress("complete")
    return summary


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


def generate_phase1_1(
    mrxcat_repo: str | Path,
    prepared_root: str | Path,
    approved_decoder: str | Path,
    output_root: str | Path,
    matlab_parity: dict | None = None,
    run_matlab_parity: bool = False,
    prepare_only: bool = False,
) -> dict:
    """Generate the versioned Phase-1.1 corrected clean dataset only."""

    repo, output_root = Path(mrxcat_repo), Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    _progress("inspect MRXCAT and voxelize mesh")
    inspection = inspect_repo(repo)
    vti_path, mesh_path = "inpuData/Background/XCAT_vti/Image_01.vti", "inpuData/Patient1/Mesh.vtk"
    labels, info = load_vti_array(vti_path, "labels", git_repo=repo)
    source_origin_mm, source_spacing_mm = np.asarray(info.origin) * 1000.0, np.asarray(info.spacing) * 1000.0
    mesh = load_tetra_mesh(mesh_path, git_repo=repo)
    mesh_mask, origin_mm, spacing_mm = voxelize_tetrahedra(mesh, spacing_mm=1.0, padding_mm=12.0)
    labels_hr = _resample_labels_nearest(labels, source_origin_mm, source_spacing_mm, origin_mm, spacing_mm, mesh_mask.shape)
    labels_hr[mesh_mask] = 1
    maps = build_gt_maps(labels_hr, myocardium_mask_override=mesh_mask)
    gt_path = output_root / "gt_maps.npz"
    save_gt_maps(maps, str(gt_path))
    np.savez_compressed(
        output_root / "mesh_primary_myo.npz",
        myocardium_mask=mesh_mask,
        origin_mm=origin_mm,
        spacing_mm=spacing_mm,
        point_count=len(mesh.points_mm),
        tetra_count=len(mesh.cells),
    )

    approved = json.loads(Path(approved_decoder).read_text())
    selection = choose_timing_template(collect_manifest_candidates(prepared_root), approved)
    timing9 = np.asarray(selection["timing9_ms"], dtype=float)
    if run_matlab_parity:
        from .validate_hhz_parity import matlab_parity as validate_matlab_parity

        matlab_parity = validate_matlab_parity(timing9, tr_ms=2.61, vps=87)
    if matlab_parity is not None:
        (output_root / "matlab_parity.json").write_text(json.dumps(matlab_parity, indent=2) + "\n")
    _progress("derive timing and global geometry alignment")
    acquisition_time_ms = acquisition_times_from_timing9(timing9, tr_ms=2.61, vps=87)
    (output_root / "timing_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    geometry_templates = select_modal_geometry_templates(prepared_root)
    (output_root / "geometry_templates.json").write_text(json.dumps(geometry_templates, indent=2) + "\n")
    mesh_center, mesh_long_axis, mesh_transverse = mesh_axes(mesh)
    aligned, alignment_qc = apply_global_rigid_alignment(geometry_templates, mesh_center, mesh_long_axis)
    alignment_qc.update({"mesh_transverse_axis": mesh_transverse.tolist(), "global_transform_only": True})
    (output_root / "geometry_alignment_qc.json").write_text(json.dumps(alignment_qc, indent=2) + "\n")

    signals = generate_signal_field(
        maps.t1_ms, maps.t2_ms, maps.b1, timing9, tr_ms=2.61, vps=87, signal_support_mask=maps.signal_support_mask,
    )
    if not np.array_equal(signals[~maps.signal_support_mask], np.zeros((int((~maps.signal_support_mask).sum()), 10), dtype=np.float32)):
        raise AssertionError("background signal is not exactly zero")
    np.savez_compressed(
        output_root / "clean_signal_field.npz",
        signals=signals,
        origin_mm=origin_mm,
        spacing_mm=spacing_mm,
        signal_support_mask=maps.signal_support_mask,
    )

    quadrature_levels = [(2, 2, 3), (4, 4, 5), (6, 6, 9)]
    chosen = (4, 4, 5)
    blood = np.isin(labels_hr, [5, 6, 7, 8]).astype(np.float32)[..., None]
    myo = mesh_mask.astype(np.float32)[..., None]
    summary = {"stacks": {}, "timing_selection": selection, "source_manifest": str(output_root / "synthetic_source_manifest.json")}
    spatial_qc: dict[str, dict] = {}
    for stack in ("sax", "2ch", "4ch"):
        _progress(f"convergence and anatomy QC: {stack}")
        groups = aligned[stack]
        representative = groups[len(groups) // 2]
        level_images = {
            f"{a}x{b}x{c}": integrate_signal_volume(signals, origin_mm, spacing_mm, np.asarray(representative["affine_lps_rc"]), *representative["image_shape_rows_cols"], a, c if a == b else c)
            for a, b, c in quadrature_levels
        }
        # ``integrate_signal_volume`` uses one in-plane count for both axes;
        # the prescribed hierarchy is square in plane.
        reference = level_images["6x6x9"]
        rows = {}
        for a, b, c in quadrature_levels:
            key = f"{a}x{b}x{c}"
            rows[key] = {"inplane_samples_row_col": [a, b], "thickness_samples": c, **_convergence_metrics(level_images[key], reference)}
        if rows["4x4x5"]["normalized_rmse"] >= 0.005 or rows["4x4x5"]["normalized_max_abs"] >= 0.01:
            chosen = (6, 6, 9)
        # Explicit anatomy intersection QC in the same integration domain.
        myo_hits, blood_hits = 0, 0
        for group in groups:
            myo_img = integrate_signal_volume(myo, origin_mm, spacing_mm, np.asarray(group["affine_lps_rc"]), *group["image_shape_rows_cols"], chosen[0], chosen[2])
            blood_img = integrate_signal_volume(blood, origin_mm, spacing_mm, np.asarray(group["affine_lps_rc"]), *group["image_shape_rows_cols"], chosen[0], chosen[2])
            myo_hits += int(np.count_nonzero(myo_img[..., 0] > 0.0))
            blood_hits += int(np.count_nonzero(blood_img[..., 0] > 0.0))
        if myo_hits == 0 or blood_hits == 0:
            raise AssertionError(f"{stack} does not intersect both LV myocardium and blood")
        spatial_qc[stack] = {"representative_group_idx": int(representative["group_idx"]), "reference": "6x6x9", "levels": rows, "myocardium_intersection_pixels": myo_hits, "blood_intersection_pixels": blood_hits}

    # Select one resolution globally after the full hierarchy is known, then
    # render all observations once.  A stricter stack promotes all views.
    if any(item["levels"]["4x4x5"]["normalized_rmse"] >= 0.005 or item["levels"]["4x4x5"]["normalized_max_abs"] >= 0.01 for item in spatial_qc.values()):
        chosen = (6, 6, 9)
    for entry in spatial_qc.values():
        entry["accepted_resolution"] = {"inplane_samples_row_col": [chosen[0], chosen[0]], "thickness_samples": chosen[2]}
    (output_root / "spatial_integration_convergence_qc.json").write_text(json.dumps(spatial_qc, indent=2) + "\n")
    if prepare_only:
        return {"prepared": True, "timing_selection": selection, "accepted_resolution": {"inplane_samples": chosen[0], "thickness_samples": chosen[2]}}
    for stack in ("sax", "2ch", "4ch"):
        _progress(f"render and write observations: {stack}")
        groups = aligned[stack]
        images, masks = render_groups(signals, origin_mm, spacing_mm, groups, chosen[0], chosen[2])
        metadata = {field: np.concatenate([np.repeat([g[field]], 10, axis=0) if np.asarray(g[field]).ndim else np.repeat([g[field]], 10) for g in groups]) for field in ("pixel_spacing_rc_mm", "slice_thickness_mm")}
        metadata.update({
            "group_idx": np.repeat(np.arange(len(groups)), 10),
            "weight_idx": np.tile(np.arange(10), len(groups)),
            "stack_idx": np.zeros(len(groups) * 10, dtype=np.int64),
            "acquisition_time_ms": np.tile(acquisition_time_ms, len(groups)),
            "timing9_ms": np.repeat(timing9[None, :], len(groups) * 10, axis=0),
            "affine_lps_rc": np.repeat(np.asarray([g["affine_lps_rc"] for g in groups]), 10, axis=0),
            "tr_ms": np.repeat([2.61], len(groups) * 10),
            "vps": np.repeat([87], len(groups) * 10),
        })
        result = write_observations(output_root / stack, images, masks, metadata, {"phase": "1.1", "stack": stack, "signal_field": str(output_root / "clean_signal_field.npz"), "global_alignment": alignment_qc, "spatial_integration": {"inplane_samples": chosen[0], "thickness_samples": chosen[2]}})
        summary["stacks"][stack] = {"result": result, "group_count": len(groups), "image_shape": list(images.shape), "foreground_mask_fraction": float(masks.mean())}

    source_manifest = {
        "phase": "1.1", "acquisition_date_utc": "2026-10-12", "supersedes_candidate": "simulation/outputs/phase1_clean_mrxcat (retained unchanged)",
        "mrxcat": inspection.to_dict(), "mrxcat_commit": inspection.commit,
        "mrxcat_label_object": subprocess.check_output(["git", "-C", str(repo), "rev-parse", f"HEAD:{vti_path}"], text=True).strip(),
        "mrxcat_label_sha256": hashlib.sha256(subprocess.check_output(["git", "-C", str(repo), "cat-file", "blob", f"HEAD:{vti_path}"])).hexdigest(),
        "mrxcat_mesh_object": subprocess.check_output(["git", "-C", str(repo), "rev-parse", f"HEAD:{mesh_path}"], text=True).strip(),
        "mrxcat_mesh_sha256": hashlib.sha256(subprocess.check_output(["git", "-C", str(repo), "cat-file", "blob", f"HEAD:{mesh_path}"])).hexdigest(),
        "mesh_primary_myocardium": {"path": "mesh_primary_myo.npz", "coordinate_units": "mm", "source_mesh_units": "m", "point_count": int(len(mesh.points_mm)), "tetra_count": int(len(mesh.cells)), "isotropic_spacing_mm": 1.0, "voxelization": "exact tetra barycentric containment", "blood_pool": "coarse supplied XCAT VTI nearest-neighbour resample (surrounding nuisance only)"},
        "signal_support_mapping": _signal_support_mapping(),
        "gt_contract": {"m0": "1.0 on signal_support_mask, 0.0 outside", "pd_scaling": False, "primary_myocardium_t1_ms": 1410.0, "primary_myocardium_t2_ms": 50.0, "b1": 1.0},
        "timing_selection": selection,
        "acquisition_time_ms": {"values": acquisition_time_ms.tolist(), "derivation": "selected timing9 + TR*VPS + original HHZ TI/T2prep intervals", "round_trip_verified": True},
        "geometry_templates": geometry_templates, "geometry_alignment_qc": alignment_qc,
        "spatial_integration": {"accepted": {"inplane_samples": chosen[0], "thickness_samples": chosen[2]}, "convergence_qc": "spatial_integration_convergence_qc.json", "rule": "midpoint quadrature in signal space; linear signal interpolation"},
        "forbidden_routes": {"frozen_mlp": False, "noise": False, "motion": False, "sparse": False, "reconstruction": False}, "matlab_parity": matlab_parity,
    }
    (output_root / "synthetic_source_manifest.json").write_text(json.dumps(source_manifest, indent=2) + "\n")
    (output_root / "qc_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _progress("complete")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mrxcat-repo", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--approved-decoder", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--phase1-1", action="store_true", help="generate the corrected, versioned Phase-1.1 dataset")
    parser.add_argument("--matlab-parity-json", help="existing MATLAB parity result to preserve in provenance")
    parser.add_argument("--run-matlab-parity", action="store_true", help="run the original MATLAB HHZ parity gate before writing provenance")
    parser.add_argument("--stage", choices=("all", "prepare", "render", "finalize"), default="all")
    parser.add_argument("--stack", choices=("sax", "2ch", "4ch"), help="stack for --stage render")
    args = parser.parse_args()
    if not args.phase1_1 and args.stage != "all":
        parser.error("--stage is only valid with --phase1-1")
    if args.phase1_1 and args.stage == "render":
        if args.stack is None:
            parser.error("--stack is required for --stage render")
        print(json.dumps(render_phase1_1_stack(args.mrxcat_repo, args.output_root, args.stack), indent=2))
        return
    if args.phase1_1 and args.stage == "finalize":
        print(json.dumps(finalize_phase1_1(args.mrxcat_repo, args.output_root), indent=2))
        return
    parity = json.loads(Path(args.matlab_parity_json).read_text()) if args.matlab_parity_json else None
    builder = generate_phase1_1 if args.phase1_1 else generate
    kwargs = {"matlab_parity": parity}
    if args.phase1_1:
        kwargs["run_matlab_parity"] = args.run_matlab_parity
        kwargs["prepare_only"] = args.stage == "prepare"
    print(json.dumps(builder(args.mrxcat_repo, args.prepared_root, args.approved_decoder, args.output_root, **kwargs), indent=2))


if __name__ == "__main__":
    main()
