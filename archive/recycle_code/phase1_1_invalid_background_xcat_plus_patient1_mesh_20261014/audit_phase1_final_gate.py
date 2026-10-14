"""Non-destructive Phase-1.2 renderer convergence and final-gate audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from ..mrxcat.mesh_voxelize import load_tetra_mesh, mesh_axes
from .build_geometry import apply_global_rigid_alignment
from .render_observations import foreground_mask, integrate_signal_volume
from .validate_hhz_parity import build_reference_config


LEVELS = ((4, 4, 5), (6, 6, 9), (8, 8, 13))
OPTIONAL_LEVEL = (10, 10, 17)
STACKS = ("sax", "2ch", "4ch")


def _level_key(level: tuple[int, int, int]) -> str:
    return f"{level[0]}x{level[1]}x{level[2]}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def comparison_metrics(value: np.ndarray, reference: np.ndarray, myocardium_intersecting: np.ndarray) -> dict:
    """Signal-domain comparison on the finer-level foreground support."""

    value, reference = np.asarray(value, dtype=float), np.asarray(reference, dtype=float)
    if value.shape != reference.shape or value.ndim != 3:
        raise ValueError("value/reference must be equally shaped (rows, cols, weights)")
    foreground = foreground_mask(np.moveaxis(reference, -1, 0))
    myo = np.asarray(myocardium_intersecting, dtype=bool) & foreground
    if not np.any(foreground):
        raise ValueError("foreground is empty")
    delta = value - reference
    diff = delta[foreground]
    ref = reference[foreground]
    scale = float(np.max(np.abs(ref)))
    l2 = float(np.linalg.norm(ref.ravel()))
    value_l2 = float(np.linalg.norm(value[foreground].ravel()))
    raw = np.abs(diff)
    whole_max = float(raw.max())
    myo_max = float(np.abs(delta[myo]).max()) if np.any(myo) else None
    rmse = float(np.sqrt(np.mean(diff**2)))
    cosine = float(1.0 - np.dot(value[foreground].ravel(), ref.ravel()) / (value_l2 * l2)) if value_l2 and l2 else 0.0
    return {
        "foreground_pixel_count": int(foreground.sum()),
        "myocardium_intersecting_pixel_count": int(myo.sum()),
        "max_abs": whole_max,
        "normalized_max_abs": whole_max / scale if scale else 0.0,
        "whole_foreground_max_abs": whole_max,
        "whole_foreground_normalized_max_abs": whole_max / scale if scale else 0.0,
        "myocardium_intersecting_pixel_max_abs": myo_max,
        "myocardium_intersecting_pixel_normalized_max_abs": myo_max / scale if myo_max is not None and scale else None,
        "p99_abs": float(np.percentile(raw, 99.0)),
        "p99_normalized_abs": float(np.percentile(raw, 99.0) / scale) if scale else 0.0,
        "rmse": rmse,
        "normalized_rmse": rmse / scale if scale else 0.0,
        "relative_l2": float(np.linalg.norm(diff.ravel()) / l2) if l2 else 0.0,
        "fingerprint_cosine_error": cosine,
        "normalization_scale_max_abs_reference": scale,
    }


def meets_plateau(metrics: dict) -> bool:
    return metrics["normalized_rmse"] < 0.005 and metrics["p99_normalized_abs"] < 0.01


def choose_accepted_level(pair_metrics: dict[str, dict[str, dict]]) -> tuple[tuple[int, int, int] | None, str]:
    """Select the lowest global level meeting Phase-1.2 plateau criteria."""

    six_eight = pair_metrics["6x6x9_vs_8x8x13"]
    if all(meets_plateau(item) for item in six_eight.values()):
        return (6, 6, 9), "6x6x9_vs_8x8x13 passes normalized RMSE < 0.5% and P99 < 1% for every stack"
    eight_ten = pair_metrics.get("8x8x13_vs_10x10x17")
    if eight_ten and all(meets_plateau(item) for item in eight_ten.values()):
        return (8, 8, 13), "6x6x9 fails plateau; 8x8x13_vs_10x10x17 passes for every stack"
    return None, "no tested global quadrature level reached the required plateau"


def _probe_matlab() -> dict:
    matlab = shutil.which("matlab")
    probe = {"which_matlab": matlab, "batch_command": 'matlab -batch "disp(version); exit"', "octave": shutil.which("octave")}
    if matlab is None:
        probe.update({"returncode": None, "stdout": "", "stderr": "", "status": "unavailable"})
        return probe
    try:
        completed = subprocess.run([matlab, "-batch", "disp(version); exit"], text=True, capture_output=True, timeout=45)
        probe.update({"returncode": int(completed.returncode), "stdout": completed.stdout, "stderr": completed.stderr, "status": "available" if completed.returncode == 0 else "execution_failed"})
    except subprocess.TimeoutExpired as exc:
        probe.update({"returncode": None, "stdout": exc.stdout or "", "stderr": exc.stderr or "", "status": "timeout"})
    return probe


def _frozen_contract(root: Path) -> dict:
    selection = json.loads((root / "timing_selection.json").read_text())
    source = json.loads((root / "synthetic_source_manifest.json").read_text())
    gt = source["gt_contract"]
    frozen = {
        "mrxcat_commit": source["mrxcat_commit"],
        "myocardium_t1_t2_ms": [gt["primary_myocardium_t1_ms"], gt["primary_myocardium_t2_ms"]],
        "b1": gt["b1"],
        "m0": gt["m0"],
        "pd_scaling": gt["pd_scaling"],
        "timing_source": {key: selection["source"][key] for key in ("subject_id", "stack_name", "group_idx")},
        "tr_ms": selection["tr_ms"],
        "vps": selection["vps"],
        "geometry_source_subjects": {stack: json.loads((root / "geometry_templates.json").read_text())[stack]["source_subject"] for stack in STACKS},
        "one_global_rigid_alignment": json.loads((root / "geometry_alignment_qc.json").read_text())["global_transform_only"],
        "primary_myocardium_gt": "1-mm mesh-derived exact tetra barycentric containment",
    }
    expected = (frozen["myocardium_t1_t2_ms"] == [1410.0, 50.0] and frozen["b1"] == 1.0 and frozen["pd_scaling"] is False and frozen["timing_source"] == {"subject_id": "CYJ", "stack_name": "sax", "group_idx": 5} and frozen["tr_ms"] == 2.61 and frozen["vps"] == 87 and frozen["one_global_rigid_alignment"] is True)
    frozen["verified"] = expected
    return frozen


def _hashes(root: Path, convergence_path: Path) -> dict:
    names = ["clean_signal_field.npz", "gt_maps.npz", "mesh_primary_myo.npz", "timing_selection.json", "geometry_templates.json", "geometry_alignment_qc.json", "synthetic_source_manifest.json", "matlab_parity.json", "matlab_hhz_reference_config.json"]
    names += [f"{stack}/observations.npz" for stack in STACKS]
    result = {name: _sha256(root / name) for name in names}
    result[convergence_path.name] = _sha256(convergence_path)
    return result


def audit(output_root: str | Path, mrxcat_repo: str | Path) -> dict:
    root, repo = Path(output_root), Path(mrxcat_repo)
    convergence_path, gate_path, config_path = (root / "phase1_final_convergence.json", root / "phase1_final_gate.json", root / "matlab_hhz_reference_config.json")
    existing = [p for p in (convergence_path, gate_path, config_path) if p.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite Phase-1.2 output(s): " + ", ".join(str(p) for p in existing))
    signal = np.load(root / "clean_signal_field.npz")
    myo = np.load(root / "mesh_primary_myo.npz")["myocardium_mask"].astype(np.float32)[..., None]
    templates = json.loads((root / "geometry_templates.json").read_text())
    mesh = load_tetra_mesh("inpuData/Patient1/Mesh.vtk", git_repo=repo)
    center, long_axis, _ = mesh_axes(mesh)
    aligned, _ = apply_global_rigid_alignment(templates, center, long_axis)
    levels_by_stack: dict[str, dict[str, np.ndarray]] = {}
    myo_intersections: dict[str, np.ndarray] = {}
    for stack in STACKS:
        representative = aligned[stack][len(aligned[stack]) // 2]
        rows, cols = representative["image_shape_rows_cols"]
        affine = np.asarray(representative["affine_lps_rc"])
        levels_by_stack[stack] = {_level_key(level): integrate_signal_volume(signal["signals"], signal["origin_mm"], signal["spacing_mm"], affine, rows, cols, level[0], level[2]) for level in LEVELS}
        myo_intersections[stack] = integrate_signal_volume(myo, signal["origin_mm"], signal["spacing_mm"], affine, rows, cols, 8, 13)[..., 0] > 0.0
    pairs = ((LEVELS[0], LEVELS[1]), (LEVELS[1], LEVELS[2]))
    pair_metrics: dict[str, dict[str, dict]] = {}
    for coarse, fine in pairs:
        key = f"{_level_key(coarse)}_vs_{_level_key(fine)}"
        pair_metrics[key] = {stack: comparison_metrics(levels_by_stack[stack][_level_key(coarse)], levels_by_stack[stack][_level_key(fine)], myo_intersections[stack]) for stack in STACKS}
    accepted, decision = choose_accepted_level(pair_metrics)
    if accepted is None:
        for stack in STACKS:
            representative = aligned[stack][len(aligned[stack]) // 2]
            rows, cols = representative["image_shape_rows_cols"]
            levels_by_stack[stack][_level_key(OPTIONAL_LEVEL)] = integrate_signal_volume(signal["signals"], signal["origin_mm"], signal["spacing_mm"], np.asarray(representative["affine_lps_rc"]), rows, cols, OPTIONAL_LEVEL[0], OPTIONAL_LEVEL[2])
        pair_metrics["8x8x13_vs_10x10x17"] = {stack: comparison_metrics(levels_by_stack[stack]["8x8x13"], levels_by_stack[stack]["10x10x17"], myo_intersections[stack]) for stack in STACKS}
        accepted, decision = choose_accepted_level(pair_metrics)
    convergence = {
        "phase": "1.2", "integration_domain": "signal-domain midpoint quadrature with linear signal interpolation", "representative_groups": {stack: int(aligned[stack][len(aligned[stack]) // 2]["group_idx"]) for stack in STACKS},
        "levels_run": [_level_key(level) for level in LEVELS] + ([_level_key(OPTIONAL_LEVEL)] if "8x8x13_vs_10x10x17" in pair_metrics else []),
        "pairwise_metrics": pair_metrics,
        "acceptance_rule": {"normalized_rmse_lt": 0.005, "p99_normalized_abs_lt": 0.01, "max_abs_interpretation": "reported for whole foreground and myocardium-intersecting pixels; not a sole acceptance criterion because boundary partial volume is expected"},
        "renderer_convergence_status": "pass" if accepted else "fail",
        "accepted_quadrature": {"inplane_samples_row_col": [accepted[0], accepted[1]], "thickness_samples": accepted[2]} if accepted else None,
        "decision": decision,
    }
    convergence_path.write_text(json.dumps(convergence, indent=2) + "\n")
    selection = json.loads((root / "timing_selection.json").read_text())
    config_path.write_text(json.dumps(build_reference_config(np.asarray(selection["timing9_ms"], dtype=float), float(selection["tr_ms"]), int(selection["vps"])), indent=2) + "\n")
    prior_parity = json.loads((root / "matlab_parity.json").read_text())
    environment = _probe_matlab()
    parity_status = prior_parity.get("status", "unresolved_external_gate")
    blockers = []
    if convergence["renderer_convergence_status"] != "pass":
        blockers.append("renderer convergence did not reach the specified plateau")
    if parity_status != "pass":
        blockers.append("original MATLAB HHZ parity remains an unresolved external gate")
    gate = {
        "phase": "1.2", "renderer_convergence_status": convergence["renderer_convergence_status"], "accepted_quadrature": convergence["accepted_quadrature"],
        "matlab_parity_status": parity_status, "matlab_prior_result": prior_parity, "matlab_environment_probe": environment,
        "formal_synthetic_reconstruction_ready": bool(convergence["renderer_convergence_status"] == "pass" and parity_status == "pass"),
        "blockers": blockers, "frozen_contract": _frozen_contract(root), "relevant_sha256": _hashes(root, convergence_path),
        "external_matlab_reference": {"config": str(config_path.resolve()), "exporter": str((Path(__file__).resolve().parents[3] / "trad/modules/module_05_signal_decoder/export_matlab_signal_reference.m").resolve()), "expected_output": str((root / "matlab_hhz_reference.mat").resolve())},
    }
    gate_path.write_text(json.dumps(gate, indent=2) + "\n")
    return {"convergence": convergence, "gate": gate, "convergence_path": str(convergence_path), "gate_path": str(gate_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write non-destructive Phase-1.2 final validation artifacts.")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mrxcat-repo", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.output_root, args.mrxcat_repo), indent=2))


if __name__ == "__main__":
    main()
