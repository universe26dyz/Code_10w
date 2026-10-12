"""Read-only evidence audit for historical 2D-fit-first T1/T2 final poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import sha256, write_json


def _artifact_chain(root: Path, subject: str, version: str) -> dict[str, dict] | None:
    base = root / "subjects" / subject / "results" / "2D_fit_first" / "experiments" / "20260822_v5_full_pipeline_v1" / "output" / version
    chains: dict[str, dict] = {}
    for parameter in ("T1", "T2"):
        result = base / parameter / "result.json"
        artifacts = sorted((base / parameter / "registered_slices").glob("*.nii.gz"))
        if not result.is_file() or not artifacts:
            return None
        config = json.loads(result.read_text(encoding="utf-8"))
        chains[parameter] = {
            "reconstruction_result": str(result), "reconstruction_result_sha256": sha256(result),
            "registration": config.get("registration"), "no_transformation_optimization": config.get("no_transformation_optimization"),
            "final_pose_artifact": str(artifacts[0]), "final_pose_artifact_sha256": sha256(artifacts[0]),
            "registered_slice_count": len(artifacts),
            "registered_slice_sha256": {path.name: sha256(path) for path in artifacts},
        }
    return chains


def _geometry_evidence(root: Path, subject: str, version: str) -> dict | None:
    path = root / "subjects" / subject / "results" / "2D_fit_first" / "evaluations" / "20260823_dual_full_v1" / version / "manifests" / "registered_geometry_validation.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return {"path": str(path), "sha256": sha256(path), "status": value.get("status"), "t1_t2_pose_difference": value.get("t1_t2_pose_difference")}


def audit_pose_policy(old_root: str | Path, *, subject: str = "CYJ", version: str = "nesvor_v5_128_w1") -> dict:
    root = Path(old_root).resolve()
    chain = _artifact_chain(root, subject, version)
    code = root / "method_repositories" / "2D_fit_first" / "python" / "evaluation" / "roi" / "map_native_roi_to_final_pose.py"
    geometry = _geometry_evidence(root, subject, version)
    independent = False
    conflict = None
    if chain is None:
        conflict = "T1/T2 independent reconstruction and registered-slice artifacts are incomplete."
    else:
        t1, t2 = chain["T1"], chain["T2"]
        shared_hashes = set(t1["registered_slice_sha256"].values()) & set(t2["registered_slice_sha256"].values())
        if t1["registration"] != "stack" or t2["registration"] != "stack" or t1["no_transformation_optimization"] is not False or t2["no_transformation_optimization"] is not False:
            conflict = "T1/T2 result configuration does not prove independent stack transformation optimization."
        elif shared_hashes:
            conflict = "T1/T2 registered-slice artifacts share bytes; possible pose copy/reuse requires review."
        elif geometry and geometry["t1_t2_pose_difference"].get("all_identical") is not False:
            conflict = "Registered-geometry validation does not prove distinct T1/T2 final poses."
        else:
            independent = True
    evidence = [path for path in (code,) if path.is_file()]
    if geometry: evidence.append(Path(geometry["path"]))
    if chain:
        evidence.extend(Path(chain[parameter]["reconstruction_result"]) for parameter in ("T1", "T2"))
        evidence.extend(Path(chain[parameter]["final_pose_artifact"]) for parameter in ("T1", "T2"))
    return {
        "audit_type": "2dfit_pose_policy", "subject": subject, "version": version,
        "t1_reconstruction": "independent NeSVoR T1 reconstruction with its own registered/final poses" if independent else "unresolved",
        "t2_reconstruction": "independent NeSVoR T2 reconstruction with its own registered/final poses" if independent else "unresolved",
        "shared_final_pose": False if independent else "unresolved",
        "pose_optimizer": "NeSVoR stack registration, separately optimized for T1 and T2" if independent else "unresolved",
        "pose_transfer_rule": "No T1-to-T2 pose copy/reuse: evaluation maps each parameter to its own registered artifact." if independent else "unresolved",
        "artifact_chain": chain or {}, "registered_geometry_evidence": geometry,
        "evidence_files": [str(path) for path in evidence], "evidence_hashes": {str(path): sha256(path) for path in evidence},
        "unresolved_reason": conflict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", required=True); parser.add_argument("--subject", default="CYJ")
    parser.add_argument("--version", default="nesvor_v5_128_w1"); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_json(args.output, audit_pose_policy(args.old_root, subject=args.subject, version=args.version))
    print(Path(args.output))


if __name__ == "__main__": main()
