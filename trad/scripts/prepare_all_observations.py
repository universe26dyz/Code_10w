"""Prepare exactly the server subject/stack entries listed in a deployment manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.deployment_manifest import load_deployment_manifest
from modules.module_02_data_bridge.prepare_observations import prepare_observations


def prepare_all_observations(manifest_path: str | Path, preprocessed_root: str | Path, prepared_root: str | Path) -> dict[str, object]:
    root, destination = Path(preprocessed_root), Path(prepared_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Preprocessed root does not exist: {root}")
    entries = load_deployment_manifest(manifest_path)
    batch_qc_path = destination / "PREPARED_BATCH_QC.json"
    if batch_qc_path.exists():
        raise FileExistsError(f"Prepared batch QC already exists: {batch_qc_path}")
    reports = []
    for entry in entries:
        subject_id, stack = entry["subject_id"], entry["stack"]
        mat = root / subject_id / stack / "preprocessed.mat"
        if not mat.is_file():
            raise FileNotFoundError(f"Required transferred preprocessed MAT is missing: {mat}")
        output = destination / subject_id / stack
        result = prepare_observations(mat, entry["server_dicom_dir"], stack, output)
        required = ("observations.npz", "manifest.json", "timing.npy", "qc_summary.json")
        missing = [name for name in required if not (output / name).is_file()]
        if missing:
            raise RuntimeError(f"Observation preparation did not create required outputs for {subject_id}/{stack}: {missing}")
        reports.append({"subject_id": subject_id, "stack": stack, "preprocessed_mat": str(mat), "prepared_dir": str(output), **result["qc_summary"]})
    report: dict[str, object] = {"schema_version": "prepared-batch-qc-v1", "prepared_count": len(reports), "entries": reports}
    destination.mkdir(parents=True, exist_ok=True)
    batch_qc_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--preprocessed-root", required=True)
    parser.add_argument("--prepared-root", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_all_observations(args.manifest, args.preprocessed_root, args.prepared_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
