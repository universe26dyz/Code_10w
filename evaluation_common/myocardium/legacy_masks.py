"""Discovery and exact-lattice geometry audit for legacy myocardial masks."""

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


def _corners(shape: tuple[int, int, int]) -> np.ndarray:
    return np.array(
        [[x, y, z, 1.0] for x in (0, shape[0] - 1) for y in (0, shape[1] - 1) for z in (0, shape[2] - 1)],
        dtype=float,
    ).T


def classify_affine_relation(
    source_affine: np.ndarray,
    reference_affine: np.ndarray,
    source_shape: tuple[int, int, int] | list[int] | None = None,
    reference_shape: tuple[int, int, int] | list[int] | None = None,
    *,
    atol: float = 2e-4,
) -> dict[str, Any]:
    """Classify identical, exactly reorientable, and non-equivalent lattices.

    An exact reorientation is restricted to an axis permutation and/or reversal
    whose integer translation maps the complete source extent onto the complete
    reference extent.  It therefore requires no interpolation.
    """

    source = np.asarray(source_affine, dtype=float)
    reference = np.asarray(reference_affine, dtype=float)
    if source.shape != (4, 4) or reference.shape != (4, 4):
        raise ValueError("NIfTI affine matrices must be 4x4.")
    if not np.isfinite(source).all() or not np.isfinite(reference).all():
        raise ValueError("NIfTI affine matrices must be finite.")
    transform = np.linalg.solve(reference, source)
    linear = transform[:3, :3]
    rounded_linear = np.rint(linear).astype(int)
    signed_permutation = bool(
        np.allclose(linear, rounded_linear, atol=atol, rtol=0.0)
        and np.all(np.isin(rounded_linear, (-1, 0, 1)))
        and np.all(np.count_nonzero(rounded_linear, axis=0) == 1)
        and np.all(np.count_nonzero(rounded_linear, axis=1) == 1)
    )
    translation = transform[:3, 3]
    rounded_translation = np.rint(translation).astype(int)
    integer_translation = bool(np.allclose(translation, rounded_translation, atol=atol, rtol=0.0))

    shapes_available = source_shape is not None and reference_shape is not None
    if shapes_available:
        source_shape_tuple = tuple(int(value) for value in source_shape)
        reference_shape_tuple = tuple(int(value) for value in reference_shape)
        if len(source_shape_tuple) != 3 or len(reference_shape_tuple) != 3:
            raise ValueError("Source and reference shapes must have three dimensions.")
    else:
        source_shape_tuple = reference_shape_tuple = None

    source_axis_for_reference: list[int] = []
    axis_signs_reference: list[int] = []
    if signed_permutation:
        source_axis_for_reference = np.argmax(np.abs(rounded_linear), axis=1).astype(int).tolist()
        axis_signs_reference = [int(rounded_linear[i, j]) for i, j in enumerate(source_axis_for_reference)]

    shape_permutation = False
    corner_mapping = False
    lattice_mapping = False
    physical_error_max: float | None = None
    if signed_permutation and integer_translation and shapes_available:
        assert source_shape_tuple is not None and reference_shape_tuple is not None
        shape_permutation = all(
            reference_shape_tuple[i] == source_shape_tuple[source_axis_for_reference[i]] for i in range(3)
        )
        expected_translation = np.array(
            [0 if sign > 0 else reference_shape_tuple[i] - 1 for i, sign in enumerate(axis_signs_reference)]
        )
        extent_translation = bool(np.array_equal(rounded_translation, expected_translation))
        if shape_permutation and extent_translation:
            mapped = transform @ _corners(source_shape_tuple)
            expected = _corners(reference_shape_tuple)
            mapped_set = {tuple(np.rint(mapped[:3, i]).astype(int)) for i in range(mapped.shape[1])}
            expected_set = {tuple(expected[:3, i].astype(int)) for i in range(expected.shape[1])}
            corner_mapping = bool(
                np.allclose(mapped[:3], np.rint(mapped[:3]), atol=atol, rtol=0.0)
                and mapped_set == expected_set
            )
            lattice_mapping = corner_mapping
            source_world = source @ _corners(source_shape_tuple)
            reference_world = reference @ np.vstack((np.rint(mapped[:3]), np.ones(mapped.shape[1])))
            physical_error_max = float(np.max(np.abs(source_world - reference_world)))

    identity = bool(
        shapes_available
        and source_shape_tuple == reference_shape_tuple
        and np.allclose(transform, np.eye(4), atol=atol, rtol=0.0)
    )
    exact = bool(
        signed_permutation
        and integer_translation
        and shape_permutation
        and corner_mapping
        and lattice_mapping
        and physical_error_max is not None
        and physical_error_max <= max(atol, 1e-3)
    )
    geometry_class = "IDENTICAL_GRID" if identity else "EXACT_REORIENTABLE_GRID" if exact else "NON_EQUIVALENT_GRID"
    status = "PASS" if geometry_class != "NON_EQUIVALENT_GRID" else "STOP"
    return {
        "status": status,
        "geometry_class": geometry_class,
        "reason": {
            "IDENTICAL_GRID": "geometry_identical",
            "EXACT_REORIENTABLE_GRID": "exact_signed_axis_reorientation",
            "NON_EQUIVALENT_GRID": "geometry_not_exactly_reorientable",
        }[geometry_class],
        "voxel_transform_source_to_reference": np.round(transform, 8).tolist(),
        "signed_permutation_pass": signed_permutation,
        "integer_translation_pass": integer_translation,
        "shape_permutation_pass": shape_permutation,
        "corner_mapping_pass": corner_mapping,
        "lattice_mapping_pass": lattice_mapping,
        "physical_coordinate_error_max": physical_error_max,
        "source_axis_for_reference": source_axis_for_reference,
        "axis_signs_reference": axis_signs_reference,
        "integer_translation_reference": rounded_translation.tolist() if integer_translation else None,
    }


def reorient_labels_exact(
    source_values: np.ndarray,
    relation: dict[str, Any],
    reference_shape: tuple[int, int, int] | list[int],
) -> np.ndarray:
    """Apply a verified permutation/reversal without interpolation."""

    values = np.asarray(source_values)
    target_shape = tuple(int(value) for value in reference_shape)
    if relation.get("status") != "PASS" or relation.get("geometry_class") not in {
        "IDENTICAL_GRID",
        "EXACT_REORIENTABLE_GRID",
    }:
        raise ValueError("Refusing to reorient labels without a verified exact lattice relation.")
    axes = tuple(int(value) for value in relation["source_axis_for_reference"])
    signs = tuple(int(value) for value in relation["axis_signs_reference"])
    if sorted(axes) != [0, 1, 2] or any(sign not in (-1, 1) for sign in signs):
        raise ValueError("Invalid exact-reorientation axis metadata.")
    result = np.transpose(values, axes=axes)
    for axis, sign in enumerate(signs):
        if sign < 0:
            result = np.flip(result, axis=axis)
    if result.shape != target_shape:
        raise ValueError(f"Reoriented label shape {result.shape} does not match {target_shape}.")
    if np.count_nonzero(result) != np.count_nonzero(values) or not np.array_equal(np.unique(result), np.unique(values)):
        raise ValueError("Exact reorientation changed label values or voxel counts.")
    return result


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
            "schema": "legacy_myocardium_mask_audit/v2",
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
            "schema": "legacy_myocardium_mask_audit/v2",
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
        relations[name] = classify_affine_relation(
            affine,
            reference_affine,
            record["shape_xyz"],
            reference["shape_xyz"],
        )
    status = "PASS" if all(value["status"] == "PASS" for value in relations.values()) else "STOP"
    return {
        "schema": "legacy_myocardium_mask_audit/v2",
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
