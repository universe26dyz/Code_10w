"""Read-only discovery and geometry audit for legacy manual myocardial masks.

This module deliberately does not resample, repair, or derive any mask.  A
manual label map may be integrated only after its affine header agrees with the
explicit native reference that was segmented.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


_MASK_FILENAMES = {
    "outer_filled": "Segmentation_1-out-label.nii.gz",
    "inner_filled": "Segmentation_2-in-label.nii.gz",
    "landmark_labels": "Segmentation_3-label_point.nii.gz",
}


def classify_affine_relation(source_affine: np.ndarray, reference_affine: np.ndarray, *, atol: float = 1e-5) -> dict[str, Any]:
    """Return PASS only for identical physical voxel geometry.

    Same array shape is intentionally not a sufficient condition.  The returned
    voxel transform documents why a non-identical header is a hard stop rather
    than an invitation to guess a transpose/flip/translation.
    """

    source = np.asarray(source_affine, dtype=float)
    reference = np.asarray(reference_affine, dtype=float)
    if source.shape != (4, 4) or reference.shape != (4, 4):
        raise ValueError("NIfTI affine matrices must be 4x4.")
    if not np.isfinite(source).all() or not np.isfinite(reference).all():
        raise ValueError("NIfTI affine matrices must be finite.")
    transform = np.linalg.solve(reference, source)
    status = "PASS" if np.allclose(transform, np.eye(4), atol=atol, rtol=0.0) else "STOP"
    return {
        "status": status,
        "reason": "geometry_identical" if status == "PASS" else "geometry_header_mismatch",
        "voxel_transform_source_to_reference": np.round(transform, 8).tolist(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_voxels(values: np.ndarray, *, is_label_map: bool) -> dict[str, Any]:
    """Describe labels exactly while keeping continuous reference images compact."""

    values = np.asarray(values)
    if is_label_map:
        return {"label_values": np.unique(values).tolist()}
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {"finite_min": None, "finite_max": None}
    return {"finite_min": float(finite.min()), "finite_max": float(finite.max())}


def _nifti_record(path: Path, *, is_label_map: bool) -> tuple[dict[str, Any], np.ndarray]:
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - environment-specific import guard
        raise RuntimeError("Legacy-mask audit requires nibabel.") from exc
    image = nib.load(str(path))
    values = np.asanyarray(image.dataobj)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "type": "NIfTI-1/2 gzip",
        "shape_xyz": list(values.shape),
        "dtype": str(values.dtype),
        **summarize_voxels(values, is_label_map=is_label_map),
        "nonzero_voxels_per_slice": np.count_nonzero(values, axis=(0, 1)).astype(int).tolist(),
        "zooms_xyz_mm": [float(x) for x in image.header.get_zooms()[:3]],
        "affine_ras": np.round(image.affine, 8).tolist(),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "size_bytes": stat.st_size,
        "sha256": _sha256(path),
    }, np.asarray(image.affine, dtype=float)


def audit_legacy_subject(subject_id: str, subjects_root: str | Path) -> dict[str, Any]:
    """Discover one legacy segmentation triad and validate it against its composite.

    A duplicate or missing triad is a loud STOP; selecting files by directory
    order would make a provenance claim without evidence.
    """

    subject_root = Path(subjects_root).expanduser().resolve() / subject_id
    roi_root = subject_root / "data" / "roi"
    outer = sorted(roi_root.glob(f"*/{_MASK_FILENAMES['outer_filled']}"))
    if len(outer) != 1:
        return {
            "schema": "legacy_myocardium_mask_audit/v1",
            "subject_id": subject_id,
            "status": "STOP",
            "reason": "missing_or_ambiguous_outer_segmentation",
            "matches": [str(path.resolve()) for path in outer],
        }
    source_dir = outer[0].parent
    missing = [name for name, filename in _MASK_FILENAMES.items() if not (source_dir / filename).is_file()]
    composite = source_dir / "sax_composite_reference.nii.gz"
    if missing or not composite.is_file():
        return {
            "schema": "legacy_myocardium_mask_audit/v1",
            "subject_id": subject_id,
            "status": "STOP",
            "reason": "incomplete_manual_segmentation_bundle",
            "source_directory": str(source_dir.resolve()),
            "missing": missing + ([] if composite.is_file() else ["sax_composite_reference"]),
        }
    reference, reference_affine = _nifti_record(composite, is_label_map=False)
    masks: dict[str, Any] = {}
    relations: dict[str, Any] = {}
    for name, filename in _MASK_FILENAMES.items():
        record, affine = _nifti_record(source_dir / filename, is_label_map=True)
        masks[name] = record
        relations[name] = classify_affine_relation(affine, reference_affine)
        if record["shape_xyz"] != reference["shape_xyz"]:
            relations[name] = {
                **relations[name],
                "status": "STOP",
                "reason": "shape_and_or_geometry_mismatch",
            }
    status = "PASS" if all(value["status"] == "PASS" for value in relations.values()) else "STOP"
    return {
        "schema": "legacy_myocardium_mask_audit/v1",
        "subject_id": subject_id,
        "status": status,
        "source_coordinate_system": "NIfTI RAS affine",
        "source_directory": str(source_dir.resolve()),
        "composite_reference": reference,
        "manual_masks": masks,
        "geometry_relations_to_composite": relations,
        "mask_semantics_from_legacy_code": {
            "outer_filled": "epicardium-enclosed filled area",
            "inner_filled": "endocardium-enclosed filled area / blood-pool candidate",
            "landmark_labels": "anterior labels [1,3,5], inferior labels [2,4,6]",
            "myocardium_full": "outer_filled AND NOT inner_filled",
            "myocardium_core": "myocardium_full distance-transform erosion > 1.0 mm",
        },
        "integration_allowed": status == "PASS",
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects-root", required=True)
    parser.add_argument("--subject-id", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = {subject: audit_legacy_subject(subject, args.subjects_root) for subject in args.subject_id}
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if any(value["status"] != "PASS" for value in report.values()):
        raise SystemExit("STOP: at least one legacy manual-mask geometry audit failed")


if __name__ == "__main__":
    main()
