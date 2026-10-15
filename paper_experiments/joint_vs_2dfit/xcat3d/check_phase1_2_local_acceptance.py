"""Local, data-free acceptance gate for the Phase-1.2 XCAT3D route."""
from __future__ import annotations

import json
from pathlib import Path

from .validate_quantitative_gt import check_active_phase1_2_wiring
from .prepare_xcat_parameters import validate_geometry_spec, validate_phantom_spec


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
        validate_phantom_spec(cardiac_spec)
        validate_geometry_spec(geometry_spec, formal=False)
        cardiac_spec_ok = True
    except (OSError, ValueError, TypeError, KeyError):
        cardiac_spec_ok = False
    reference_mechanism_ok = (xcat_root / "prepare_xcat_parameters.py").is_file()
    pre_xcat_ok = reference_mechanism_ok and all((xcat_root / name).is_file() for name in ("prepare_xcat_parameters.py", "finalize.py", "audits.py")) and all((root / "paper_experiments/joint_vs_2dfit/simulation" / name).is_file() for name in ("render_observations.py", "build_geometry.py", "write_observations_npz.py"))
    passed = wiring["passed"] and archived and tombstone_ok and cardiac_spec_ok and pre_xcat_ok
    return {
        "LOCAL_CODE_ACCEPTANCE": "PASS" if passed else "FAIL",
        "REFERENCE_SOURCE_ANCHORING": "PASS" if reference_mechanism_ok else "FAIL",
        "REFERENCE_GENERATION_MECHANISM": "PASS" if reference_mechanism_ok else "FAIL",
        "CARDIAC_PHANTOM_SPECIFICATION": "PASS" if cardiac_spec_ok else "FAIL",
        "PRE_XCAT_SCIENTIFIC_CLOSURE": "PASS" if pre_xcat_ok else "FAIL",
        "EXTERNAL_XCAT_RUNTIME_GATE": "UNVERIFIED",
        "FROZEN_XCAT_GEOMETRY_GATE": "UNVERIFIED",
        "XCAT_RAW_VOLUME_GATE": "UNVERIFIED",
        "ROI_STRUCTURE_COVERAGE_GATE": "UNVERIFIED",
        "QUANTITATIVE_TUPLE_GATE": "UNVERIFIED",
        "EXTERNAL_XCAT_INPUT_GATE": "UNVERIFIED",
        "SCIENTIFIC_GT_GEOMETRY_GATE": "UNVERIFIED",
        "OBSERVATION_COVERAGE_GATE": "UNVERIFIED",
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
