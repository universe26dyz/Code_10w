"""Read-only evidence audit for the historical 2D-fit-first final-pose policy."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import sha256, write_json


def audit_pose_policy(old_root: str | Path) -> dict:
    root = Path(old_root).resolve()
    evidence = [
        root / "method_repositories/2D_fit_first/python/evaluation/geometry/validate_registered_geometry.py",
        root / "method_repositories/2D_fit_first/EVALUATION_PIPELINE_GUIDE.md",
        root / "method_repositories/2D_fit_first/EVALUATION_CODE_IMPLEMENTATION_SUMMARY.md",
    ]
    existing = [path for path in evidence if path.is_file()]
    texts = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in existing)
    independent = "T1 and T2 final poses are unexpectedly all identical" in texts and "T1 与 T2 分开保存" in texts
    return {
        "audit_type": "2dfit_pose_policy",
        "t1_reconstruction": "independent NeSVoR T1 reconstruction with its own registered/final poses" if independent else "unresolved",
        "t2_reconstruction": "independent NeSVoR T2 reconstruction with its own registered/final poses" if independent else "unresolved",
        "shared_final_pose": False if independent else "unresolved",
        "pose_optimizer": "NeSVoR stack registration (independent T1/T2 runs)" if independent else "unresolved",
        "pose_transfer_rule": "native ROI membership is mapped separately to each parameter's final affine; no T1-to-T2 pose transfer" if independent else "unresolved",
        "evidence_files": [str(path) for path in existing],
        "evidence_hashes": {str(path): sha256(path) for path in existing},
        "unresolved_reason": None if independent else "Required independent-pose evidence was not found in the supplied old-engineering root.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_json(args.output, audit_pose_policy(args.old_root))
    print(Path(args.output))


if __name__ == "__main__":
    main()
