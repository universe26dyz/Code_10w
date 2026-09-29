"""Write the provenance manifest for a Q001 full-FOV CYJ input generation run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .cropped_vs_full_qc import validate_acceptance


STACKS = ("sax", "2ch", "4ch")
FORMAL_SEMANTICS = "MP-PCA(full-FOV MIND_mag_reg)"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_record(path: Path) -> dict[str, str] | None:
    return {"path": str(path.resolve()), "sha256": _sha256(path)} if path.is_file() else None


def _git_commit() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def build_manifest(output_root: str | Path, baseline_prepared_root: str | Path, *, allow_incomplete: bool = False) -> dict[str, Any]:
    output_root, baseline_prepared_root = Path(output_root), Path(baseline_prepared_root)
    records: dict[str, Any] = {}
    all_passed = True
    for stack in STACKS:
        mat = output_root / "full_fov_preprocessed" / "CYJ" / stack / "preprocessed.mat"
        prepared = output_root / "full_fov_prepared" / "CYJ" / stack
        baseline = baseline_prepared_root / "CYJ" / stack
        qc_path = output_root / "qc" / "CYJ" / f"{stack}.json"
        record: dict[str, Any] = {
            "dicom": None, "preprocessed_mat": _path_record(mat), "prepared_observations": _path_record(prepared / "observations.npz"),
            "prepared_manifest": _path_record(prepared / "manifest.json"), "timing": _path_record(prepared / "timing.npy"),
            "qc_summary": _path_record(prepared / "qc_summary.json"), "cropped_vs_full_qc": _path_record(qc_path), "qc_passed": False,
        }
        if (prepared / "manifest.json").is_file():
            manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
            dicom = manifest.get("dicom_provenance", {})
            record["dicom"] = {"directory": dicom.get("directory"), "file_count": dicom.get("file_count")}
        if (prepared / "observations.npz").is_file():
            with np.load(prepared / "observations.npz", allow_pickle=False) as arrays:
                record["group_count"] = int(np.unique(arrays["group_idx"]).size)
                record["observation_count"] = int(arrays["images"].shape[0])
                record["full_image_shape_rows_cols"] = [int(value) for value in arrays["images"].shape[1:]]
        if qc_path.is_file():
            qc = json.loads(qc_path.read_text(encoding="utf-8"))
            record["baseline_cropped_image_shape_rows_cols"] = qc.get("cropped_shape_rows_cols")
            record["crop_offsets_zero_based"] = qc.get("full_crop_offsets_zero_based")
            try:
                record["qc_passed"] = validate_acceptance(qc)
            except ValueError as exc:
                record["qc_failure"] = str(exc)
        if not record["qc_passed"]: all_passed = False
        records[stack] = record
    result = {
        "schema": "q001_input_manifest/v1", "subject_id": "CYJ", "git_commit": _git_commit(),
        "formal_preprocessing_semantics": FORMAL_SEMANTICS, "q001_output_root": str(output_root.resolve()),
        "baseline_prepared_root": str(baseline_prepared_root.resolve()), "stacks": records,
        "status": "READY_FOR_Q001_RECON_IMPLEMENTATION" if all_passed else "FAILED_INPUT_QC",
    }
    (output_root / "q001_input_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not all_passed and not allow_incomplete:
        raise ValueError("Q001 input manifest requires strict QC to pass for sax, 2ch, and 4ch.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True); parser.add_argument("--baseline-prepared-root", required=True); parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_manifest(args.output_root, args.baseline_prepared_root, allow_incomplete=args.allow_incomplete), indent=2))


if __name__ == "__main__": main()
