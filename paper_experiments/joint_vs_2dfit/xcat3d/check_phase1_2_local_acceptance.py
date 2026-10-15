"""Local, data-free acceptance gate for the Phase-1.2 XCAT3D route."""
from __future__ import annotations

import json
from pathlib import Path

from .validate_quantitative_gt import check_active_phase1_2_wiring


def local_acceptance_report(repository_root: str | Path) -> dict:
    root = Path(repository_root)
    wiring = check_active_phase1_2_wiring(root)
    archive = root / "archive/recycle_code/phase1_1_invalid_background_xcat_plus_patient1_mesh_20261014"
    required_archive = (archive / "README_INVALID_ROUTE.md", archive / "PROVENANCE.json")
    archived = all(path.is_file() for path in required_archive)
    tombstone = root / "paper_experiments/joint_vs_2dfit/simulation/generate_clean_dataset.py"
    tombstone_ok = tombstone.is_file() and "retired_phase1_1" in tombstone.read_text()
    xcat_root = root / "paper_experiments/joint_vs_2dfit/xcat3d"
    try:
        cardiac_spec = json.loads((xcat_root / "cardiac_roi_phantom_spec.json").read_text())
        geometry_spec = json.loads((xcat_root / "cardiac_roi_geometry.template.json").read_text())
        cardiac_spec_ok = len(cardiac_spec.get("structures", {})) >= 12 and cardiac_spec.get("frozen_lv") == {"t1_ms": 1410.0, "t2_ms": 50.0, "b1": 1.0} and bool(geometry_spec.get("xcat_geometry_block"))
    except (OSError, ValueError, TypeError):
        cardiac_spec_ok = False
    reference_mechanism_ok = (xcat_root / "prepare_xcat_parameters.py").is_file()
    passed = wiring["passed"] and archived and tombstone_ok and cardiac_spec_ok and reference_mechanism_ok
    return {
        "LOCAL_CODE_ACCEPTANCE": "PASS" if passed else "FAIL",
        "REFERENCE_SOURCE_ANCHORING": "PASS" if reference_mechanism_ok else "FAIL",
        "REFERENCE_GENERATION_MECHANISM": "PASS" if reference_mechanism_ok else "FAIL",
        "CARDIAC_PHANTOM_SPECIFICATION": "PASS" if cardiac_spec_ok else "FAIL",
        "EXTERNAL_XCAT_RUNTIME_GATE": "UNVERIFIED",
        "XCAT_RAW_VOLUME_GATE": "UNVERIFIED",
        "EXTERNAL_XCAT_INPUT_GATE": "UNVERIFIED",
        "SCIENTIFIC_GT_GEOMETRY_GATE": "UNVERIFIED",
        "FORMAL_SYNTHETIC_READY": "FAIL",
        "checks": {"active_wiring": wiring, "archive_present": archived, "tombstone_present": tombstone_ok},
    }


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    report = local_acceptance_report(root)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("PHASE1_2_LOCAL_ACCEPTANCE=" + report["LOCAL_CODE_ACCEPTANCE"])
    if report["LOCAL_CODE_ACCEPTANCE"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
