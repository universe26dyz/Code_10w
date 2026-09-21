"""Build a zero-interpolation CYJ myocardial mask bundle on the current SAX grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from .legacy_masks import classify_affine_relation, reorient_labels_exact


_LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


def myocardium_core_legacy(full: np.ndarray, spacing_rc_mm: tuple[float, float]) -> np.ndarray:
    """Reproduce the historical per-slice ``EDT > 1.0 mm`` definition exactly."""

    values = np.asarray(full, dtype=bool)
    if values.ndim != 3:
        raise ValueError("myocardium_full must be [group,row,col].")
    result = np.zeros_like(values)
    for group in range(values.shape[0]):
        if values[group].any():
            result[group] = distance_transform_edt(values[group], sampling=spacing_rc_mm) > 1.0
    return result


def myocardium_core_1px(full: np.ndarray) -> np.ndarray:
    """Erode each native SAX plane by exactly one in-plane pixel layer."""

    values = np.asarray(full, dtype=bool)
    if values.ndim != 3:
        raise ValueError("myocardium_full must be [group,row,col].")
    result = np.zeros_like(values)
    structure = np.ones((3, 3), dtype=bool)
    for group in range(values.shape[0]):
        result[group] = binary_erosion(values[group], structure=structure, iterations=1, border_value=0)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _load_exact(path: Path, composite_affine: np.ndarray, composite_shape: tuple[int, int, int]) -> tuple[np.ndarray, dict[str, Any]]:
    image = nib.load(str(path))
    values = np.asanyarray(image.dataobj)
    relation = classify_affine_relation(image.affine, composite_affine, values.shape, composite_shape)
    if relation["status"] != "PASS":
        raise ValueError(f"Legacy label is not exactly reorientable: {path}")
    return reorient_labels_exact(values, relation, composite_shape), relation


def _crop_offsets(full_affine: np.ndarray, crop_affine: np.ndarray) -> tuple[int, int]:
    relation = np.linalg.solve(full_affine, crop_affine)
    if not np.allclose(relation[:3, :3], np.eye(3), atol=5e-7, rtol=0.0):
        raise ValueError("Prepared crop axes differ from the full grid.")
    offset = relation[:3, 3]
    rounded = np.rint(offset).astype(int)
    if not np.allclose(offset, rounded, atol=5e-5, rtol=0.0) or rounded[2] != 0:
        raise ValueError(f"Prepared crop is not an integer in-plane crop: {offset.tolist()}")
    return int(rounded[0]), int(rounded[1])


def _save_nifti(path: Path, group_row_col: np.ndarray, affine_ras_rc: np.ndarray) -> None:
    data = np.moveaxis(group_row_col, 0, 2)
    nib.save(nib.Nifti1Image(data, affine_ras_rc), str(path))


def _write_qc(
    qc_root: Path,
    anatomical: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    outer: np.ndarray,
    inner: np.ndarray,
    full: np.ndarray,
    core_legacy: np.ndarray,
    core_1px: np.ndarray,
    landmarks: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    qc_root.mkdir(parents=True, exist_ok=False)
    for group in range(t1.shape[0]):
        figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        panels = (
            (anatomical[group], "native composite", float(np.nanpercentile(anatomical[group], 99.5))),
            (t1[group], "T1", 2500.0),
            (t2[group], "T2", 200.0),
        )
        for axis, (image, title, vmax) in zip(axes, panels):
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=max(vmax, 1e-6), origin="lower")
            if outer[group].any():
                axis.contour(outer[group], levels=[0.5], colors=["yellow"], linewidths=0.8)
            if inner[group].any():
                axis.contour(inner[group], levels=[0.5], colors=["cyan"], linewidths=0.8)
            if full[group].any():
                axis.contour(full[group], levels=[0.5], colors=["orange"], linewidths=1.0)
            if core_legacy[group].any():
                axis.contour(core_legacy[group], levels=[0.5], colors=["magenta"], linewidths=0.7, linestyles="dashed")
            if core_1px[group].any():
                axis.contour(core_1px[group], levels=[0.5], colors=["lime"], linewidths=1.2)
            points = np.argwhere(landmarks[group] > 0)
            if points.size:
                axis.scatter(points[:, 1], points[:, 0], c="cyan", s=7)
            axis.set_title(f"{title} group {group:02d}")
            axis.axis("off")
        figure.suptitle("outer=yellow, inner=cyan, full=orange, legacy EDT core=magenta dashed, 1px core=lime")
        figure.savefig(qc_root / f"sax_group_{group:02d}.png", dpi=150)
        plt.close(figure)


def build_bundle(
    *,
    subject_id: str,
    audit_json: Path,
    prepared_sax: Path,
    native_reference: Path,
    output: Path,
    repo: Path,
) -> Path:
    """Create the derived native bundle and fail on any unverifiable geometry."""

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing bundle: {output}")
    audit = json.loads(audit_json.read_text(encoding="utf-8"))[subject_id]
    if audit["status"] != "PASS":
        raise ValueError(f"Legacy audit for {subject_id} is not PASS.")
    source_dir = Path(audit["source_directory"])
    composite_path = Path(audit["composite_reference"]["path"])
    composite_image = nib.load(str(composite_path))
    composite_shape = tuple(int(value) for value in composite_image.shape)

    legacy_names = {
        "outer_filled": "Segmentation_1-out-label.nii.gz",
        "inner_filled": "Segmentation_2-in-label.nii.gz",
        "landmark_labels": "Segmentation_3-label_point.nii.gz",
    }
    composite_values: dict[str, np.ndarray] = {}
    label_relations: dict[str, Any] = {}
    for name, filename in legacy_names.items():
        composite_values[name], label_relations[name] = _load_exact(
            source_dir / filename, composite_image.affine, composite_shape
        )

    prepared_manifest_path = prepared_sax / "manifest.json"
    observations_path = prepared_sax / "observations.npz"
    prepared_manifest = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
    observations = np.load(observations_path)
    groups = prepared_manifest["groups"]
    if [group["group_idx"] for group in groups] != list(range(len(groups))):
        raise ValueError("Prepared SAX groups are not ordered 0..N-1.")
    if len(groups) != composite_shape[2]:
        raise ValueError("Legacy composite and prepared SAX group counts differ.")
    if not np.array_equal(observations["group_idx"], np.repeat(np.arange(len(groups)), 10)):
        raise ValueError("observations.npz group order does not match the manifest.")
    if not np.array_equal(observations["weight_idx"], np.tile(np.arange(10), len(groups))):
        raise ValueError("observations.npz weight order is not 0..9 within each group.")

    full_affine_lps = np.asarray(groups[0]["full_affine_lps_rc"], dtype=float)
    if len(groups) < 2:
        raise ValueError("At least two SAX groups are required to establish stack order.")
    group_step_lps = (
        np.asarray(groups[1]["full_affine_lps_rc"], dtype=float)[:3, 3]
        - full_affine_lps[:3, 3]
    )
    full_stack_affine_lps = full_affine_lps.copy()
    full_stack_affine_lps[:3, 2] = group_step_lps
    full_affine_ras = _LPS_TO_RAS @ full_stack_affine_lps
    full_shape = (composite_shape[1], composite_shape[0], composite_shape[2])
    composite_to_full = classify_affine_relation(
        composite_image.affine,
        full_affine_ras,
        composite_shape,
        full_shape,
    )
    if composite_to_full["status"] != "PASS":
        raise ValueError("Composite reference is not exactly reorientable to current full prepared SAX geometry.")
    full_values = {
        name: reorient_labels_exact(values, composite_to_full, full_shape)
        for name, values in composite_values.items()
    }
    composite_anatomical_full = reorient_labels_exact(
        np.asanyarray(composite_image.dataobj), composite_to_full, full_shape
    )

    crop_shape = tuple(int(value) for value in groups[0]["image_shape_rows_cols"])
    crop_offsets = []
    for group in groups:
        full = np.asarray(group["full_affine_lps_rc"], dtype=float)
        crop = np.asarray(group["affine_lps_rc"], dtype=float)
        expected_origin = full_affine_lps[:3, 3] + int(group["group_idx"]) * group_step_lps
        if not np.allclose(full[:3, 3], expected_origin, atol=2e-4, rtol=0.0):
            raise ValueError("Prepared full SAX group geometry is not one ordered 3-D lattice.")
        if tuple(group["image_shape_rows_cols"]) != crop_shape:
            raise ValueError("Prepared SAX crop shape changes between groups.")
        crop_offsets.append(_crop_offsets(full, crop))
    if len(set(crop_offsets)) != 1:
        raise ValueError("Prepared SAX crop offset changes between groups.")
    row0, col0 = crop_offsets[0]
    row1, col1 = row0 + crop_shape[0], col0 + crop_shape[1]
    if row1 > full_shape[0] or col1 > full_shape[1]:
        raise ValueError("Prepared crop extends outside the full grid.")
    cropped = {name: values[row0:row1, col0:col1, :] for name, values in full_values.items()}
    cropped = {name: np.moveaxis(values, 2, 0) for name, values in cropped.items()}
    composite_anatomical = np.moveaxis(composite_anatomical_full[row0:row1, col0:col1, :], 2, 0)

    outer = cropped["outer_filled"] > 0
    inner_raw = cropped["inner_filled"] > 0
    blood = inner_raw & outer
    full_myo = outer & ~blood
    spacing_rc = tuple(float(value) for value in groups[0]["pixel_spacing_rc_mm"])
    core_legacy = myocardium_core_legacy(full_myo, spacing_rc)
    core_1px = myocardium_core_1px(full_myo)

    reference_npz = np.load(native_reference / "sax.npz")
    t1 = np.asarray(reference_npz["t1_ms"])
    t2 = np.asarray(reference_npz["t2_ms"])
    if t1.shape != outer.shape or t2.shape != outer.shape:
        raise ValueError("Native reference and derived mask shapes differ.")
    reference_nii = nib.load(str(native_reference / "sax_T1_stack_ms.nii.gz"))
    crop_stack_affine_lps = np.asarray(groups[0]["affine_lps_rc"], dtype=float).copy()
    crop_stack_affine_lps[:3, 2] = group_step_lps
    crop_affine_ras = _LPS_TO_RAS @ crop_stack_affine_lps
    if not np.allclose(reference_nii.affine, crop_affine_ras, atol=5e-5, rtol=0.0):
        raise ValueError("Native reference affine does not match current prepared SAX geometry.")

    output.mkdir(parents=True, exist_ok=False)
    arrays = {
        "outer_filled": outer.astype(np.uint8),
        "inner_filled": inner_raw.astype(np.uint8),
        "blood_pool": blood.astype(np.uint8),
        "myocardium_full": full_myo.astype(np.uint8),
        "myocardium_core_legacy": core_legacy.astype(np.uint8),
        "myocardium_core_1px": core_1px.astype(np.uint8),
        "landmark_labels": cropped["landmark_labels"].astype(np.int16),
        "group_idx": np.arange(len(groups), dtype=np.int64),
    }
    np.savez_compressed(output / "sax_myocardium_masks.npz", **arrays)
    for name, values in arrays.items():
        if name != "group_idx":
            _save_nifti(output / f"{name}.nii.gz", values, crop_affine_ras)
    _save_nifti(output / "native_composite_reference.nii.gz", composite_anatomical.astype(np.float32), crop_affine_ras)
    _write_qc(
        output / "qc_all_slices",
        composite_anatomical,
        t1,
        t2,
        outer,
        inner_raw,
        full_myo,
        core_legacy,
        core_1px,
        cropped["landmark_labels"],
    )

    source_files = [composite_path] + [source_dir / filename for filename in legacy_names.values()]
    manifest = {
        "schema": "exact_native_myocardium_bundle/v2",
        "status": "PASS",
        "subject_id": subject_id,
        "repo_head": _git_head(repo),
        "method": "signed axis permutation/reversal plus integer crop; no interpolation",
        "core_definitions": {
            "myocardium_core_legacy": "distance_transform_edt(myocardium_full, sampling=row/col spacing) > 1.0 mm",
            "myocardium_core_1px": "binary_erosion(myocardium_full, structure=ones((3,3)), iterations=1, border_value=0), independently per SAX slice",
        },
        "shape_group_row_col": list(outer.shape),
        "group_idx": list(range(len(groups))),
        "crop_offset_full_rc": [row0, col0],
        "spacing_rc_mm": list(spacing_rc),
        "legacy_to_composite": label_relations,
        "composite_to_current_full_prepared": composite_to_full,
        "source_files": {str(path.resolve()): _sha256(path) for path in source_files},
        "prepared_files": {
            str(prepared_manifest_path.resolve()): _sha256(prepared_manifest_path),
            str(observations_path.resolve()): _sha256(observations_path),
        },
        "native_reference_manifest": {
            "path": str((native_reference / "native_reference_manifest.json").resolve()),
            "sha256": _sha256(native_reference / "native_reference_manifest.json"),
        },
        "per_slice_counts": [
            {
                "group_idx": group,
                "myocardium_full": int(full_myo[group].sum()),
                "myocardium_core_legacy": int(core_legacy[group].sum()),
                "myocardium_core_1px": int(core_1px[group].sum()),
                "core_1px_full_fraction": float(core_1px[group].sum() / full_myo[group].sum()) if full_myo[group].any() else None,
            }
            for group in range(len(groups))
        ],
        "core_1px_summary": {
            "annotated_slices": int(np.count_nonzero(np.any(full_myo, axis=(1, 2)))),
            "nonempty_slices": int(np.count_nonzero(np.any(core_1px, axis=(1, 2)))),
            "empty_slices": int(np.count_nonzero(np.any(full_myo, axis=(1, 2)) & ~np.any(core_1px, axis=(1, 2)))),
        },
        "inner_outside_outer_voxels_trimmed": int(np.count_nonzero(inner_raw & ~outer)),
        "outputs": {},
    }
    for path in sorted(output.glob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["outputs"][path.name] = _sha256(path)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--audit-json", required=True, type=Path)
    parser.add_argument("--prepared-sax", required=True, type=Path)
    parser.add_argument("--native-reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo", default=Path(__file__).resolve().parents[2], type=Path)
    args = parser.parse_args()
    print(build_bundle(**vars(args)))


if __name__ == "__main__":
    main()
