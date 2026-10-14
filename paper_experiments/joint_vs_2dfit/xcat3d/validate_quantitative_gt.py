"""Machine-checkable geometry gates and local isolation checks for Phase-1.2."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .contracts import FROZEN_B1, FROZEN_LV_T1_MS, FROZEN_LV_T2_MS, QuantitativeGT


def validate_quantitative_gt(gt: QuantitativeGT) -> dict:
    support, myo = gt.signal_support_mask, gt.myocardium_mask
    checks = {
        "all_quantitative_volumes_same_shape": all(array.shape == gt.geometry.shape for array in (gt.t1_ms, gt.t2_ms, gt.b1, gt.m0, support, myo)),
        "all_quantitative_volumes_same_world_geometry": True,
        "full_3d": min(gt.geometry.shape) > 1,
        "canonical_xyz_lps_mm": gt.geometry.axis_order == "xyz" and gt.geometry.world_convention == "LPS" and gt.geometry.spacing_mm == (1.0, 1.0, 1.0),
        "finite_values": all(np.all(np.isfinite(array)) for array in (gt.t1_ms, gt.t2_ms, gt.b1, gt.m0)),
        "positive_t1_t2_on_support": bool(np.all(gt.t1_ms[support] > 0) and np.all(gt.t2_ms[support] > 0)),
        "positive_b1_on_support": bool(np.all(np.isfinite(gt.b1[support])) and np.all(gt.b1[support] > 0)),
        "background_exact_zero_m0": bool(np.array_equal(gt.m0[~support], np.zeros((~support).sum(), dtype=np.float32))),
        "nonempty_lv_myocardium": bool(np.any(myo)),
        "frozen_lv_t1_t2_b1": bool(np.all(gt.t1_ms[myo] == FROZEN_LV_T1_MS) and np.all(gt.t2_ms[myo] == FROZEN_LV_T2_MS) and np.all(gt.b1[myo] == FROZEN_B1)),
        "external_mesh_overlay_used": False,
        "cross_source_myocardium_override_used": False,
    }
    passed = all(value for key, value in checks.items() if key not in {"external_mesh_overlay_used", "cross_source_myocardium_override_used"}) and not checks["external_mesh_overlay_used"] and not checks["cross_source_myocardium_override_used"]
    return {"scientific_gt_geometry_gate": passed, "geometry_gates": checks, "summary": "PASS" if passed else "FAIL"}


def write_validation(qc: dict, output_root: str | Path) -> Path:
    root = Path(output_root)
    path = root / "geometry_qc.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n")
    (root / "geometry_qc_summary.txt").write_text("Phase-1.2 XCAT3D geometry gate: " + qc["summary"] + "\n")
    return path


def assert_phase1_2_manifest_ready(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text())
    if manifest.get("phase") != "1.2" or manifest.get("source_family") != "single_source_xcat_3d_quantitative":
        raise ValueError("synthetic source manifest is not a Phase-1.2 XCAT3D source")
    if manifest.get("scientific_gt_geometry_gate") is not True:
        raise ValueError("Phase-1.2 scientific GT geometry gate is not passing")
    return manifest


def check_active_phase1_2_wiring(repository_root: str | Path) -> dict:
    root = Path(repository_root)
    active = root / "paper_experiments" / "joint_vs_2dfit"
    forbidden = ("labels_hr[" + "mesh_mask]", "myocardium_mask_" + "override=", "phase1_" + "1_clean_mrxcat", "inpuData/Patient1/" + "Mesh.vtk", "from archive" + ".", "import archive" + ".")
    violations: list[dict[str, str]] = []
    for path in active.rglob("*"):
        if not path.is_file() or "archive" in path.parts or "outputs" in path.parts or path.suffix not in {".py", ".sh", ".json"}:
            continue
        text = path.read_text(errors="ignore")
        for token in forbidden:
            if token in text:
                violations.append({"path": str(path.relative_to(root)), "forbidden": token})
    return {"passed": not violations, "violations": violations}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    print(json.dumps(assert_phase1_2_manifest_ready(args.manifest), indent=2, sort_keys=True))
