"""Fail-closed derivation of Final5 cropped ROI bundles from legacy full-FOV ROI."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..phase2_clean.contracts import sha256, write_immutable_json
from .final5_adapter import ROI_FILENAMES


DERIVATION_SCHEMA = "final5_legacy_roi_derivation_config/v1"
DERIVED_SCHEMA = "final5_cropped_roi_bundle/v1"


def _record(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _verified_file(record: Any, name: str) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"derivation config requires {name} record")
    path = Path(str(record.get("path", "")))
    if not path.is_file() or record.get("sha256") != sha256(path):
        raise ValueError(f"{name} path/SHA256 mismatch")
    return path


def _load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != DERIVATION_SCHEMA or not isinstance(payload.get("subject"), str):
        raise ValueError("invalid Final5 legacy ROI derivation config identity")
    return payload


def _standard_groups(record: Any, subject: str) -> dict[int, int]:
    standard_path = _verified_file(record, "standardization_manifest")
    payload = json.loads(standard_path.read_text(encoding="utf-8"))
    sax = payload.get("stacks", {}).get("sax") if isinstance(payload.get("stacks"), dict) else None
    values = sax.get("standardized_to_original_group_idx") if isinstance(sax, dict) else None
    if payload.get("subject") != subject or not isinstance(values, list):
        raise ValueError("standardization manifest lacks subject-matched SAX group identity")
    groups: dict[int, int] = {}
    for value in values:
        if not isinstance(value, dict) or not isinstance(value.get("derived_group_idx"), int) or not isinstance(value.get("original_group_idx"), int):
            raise ValueError("standardization SAX group record is malformed")
        groups[value["derived_group_idx"]] = value["original_group_idx"]
    return groups


def _plane_indices(*, legacy_affine: np.ndarray, target_affine: np.ndarray, legacy_slice: int,
                   target_slice: int, target_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns, rows = target_shape
    xx, yy = np.meshgrid(np.arange(columns), np.arange(rows), indexing="ij")
    target = np.stack((xx.ravel(), yy.ravel(), np.full(xx.size, target_slice), np.ones(xx.size)))
    legacy = np.linalg.inv(legacy_affine) @ target_affine @ target
    rounded = np.rint(legacy[:3])
    if not np.allclose(legacy[:3], rounded, atol=1e-5, rtol=0):
        raise ValueError("Final5 target grid is not an integer physical crop/orientation of the legacy ROI")
    source_x, source_y, source_z = (value.astype(int) for value in rounded)
    if not np.all(source_z == legacy_slice):
        raise ValueError("Final5 target plane does not correspond to its declared legacy ROI slice")
    source_x, source_y, source_z = (value.reshape(target_shape) for value in (source_x, source_y, source_z))
    steps = []
    if columns > 1:
        steps.append((int(source_x[1, 0] - source_x[0, 0]), int(source_y[1, 0] - source_y[0, 0])))
    if rows > 1:
        steps.append((int(source_x[0, 1] - source_x[0, 0]), int(source_y[0, 1] - source_y[0, 0])))
    unit_axes = {(-1, 0), (1, 0), (0, -1), (0, 1)}
    if any(step not in unit_axes for step in steps) or len(set(steps)) != len(steps) or (len(steps) == 2 and steps[0][0] * steps[1][0] + steps[0][1] * steps[1][1] != 0):
        raise ValueError("Final5 target grid is not a unit-grid crop/orientation of the legacy ROI")
    return source_x, source_y, source_z


def _write_qc_montage(path: Path, reference: np.ndarray, outer: np.ndarray, inner: np.ndarray) -> None:
    """Write one compact visual check of the directly copied Final5-grid masks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = reference.shape[2]; columns = min(4, count); rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False)
    for index, axis in enumerate(axes.flat):
        axis.set_axis_off()
        if index >= count:
            continue
        axis.imshow(reference[:, :, index].T, cmap="gray", origin="lower")
        if np.any(outer[:, :, index]):
            axis.contour(outer[:, :, index].T, levels=[.5], colors=["gold"], linewidths=1)
        if np.any(inner[:, :, index]):
            axis.contour(inner[:, :, index].T, levels=[.5], colors=["tomato"], linewidths=1)
        axis.set_title(f"Final5 SAX slice {index}")
    figure.tight_layout(); figure.savefig(path, dpi=160); plt.close(figure)


def derive_final5_cropped_roi(*, config: str | Path, output: str | Path) -> Path:
    """Create one immutable Final5 ROI bundle using direct, verified voxel copies only."""
    import nibabel as nib

    config_path = Path(config)
    payload = _load_config(config_path)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    legacy_root = Path(str(payload.get("legacy_roi", {}).get("path", "")))
    legacy_paths = {name: legacy_root / name for name in ROI_FILENAMES}
    if not all(path.is_file() for path in legacy_paths.values()):
        raise ValueError("legacy ROI folder lacks one or more canonical AHA files")
    reference_path = _verified_file(payload.get("final5_sax_reference"), "final5_sax_reference")
    standard_groups = _standard_groups(payload.get("standardization_manifest"), payload["subject"])
    reference = nib.load(str(reference_path))
    if len(reference.shape) != 3:
        raise ValueError("Final5 SAX reference must be 3-D")
    legacy = {name: nib.load(str(path)) for name, path in legacy_paths.items()}
    outer = legacy["Segmentation_1-out-label.nii.gz"]
    if any(image.shape != outer.shape or not np.allclose(image.affine, outer.affine, atol=1e-6, rtol=0) for image in legacy.values()):
        raise ValueError("legacy outer/inner/point/composite ROI geometry differs")

    planes = payload.get("planes")
    if not isinstance(planes, list) or len(planes) != reference.shape[2]:
        raise ValueError("derivation config requires one explicit plane record for every Final5 reference slice")
    derived = {name: np.zeros(reference.shape, dtype=np.asarray(image.dataobj).dtype) for name, image in legacy.items() if name != "sax_composite_reference.nii.gz"}
    seen_target: set[int] = set(); seen_native: set[int] = set(); manifest_planes: list[dict[str, Any]] = []
    for plane in planes:
        if not isinstance(plane, dict):
            raise ValueError("derivation plane record is malformed")
        native_group = plane.get("native_group_idx"); original_group = plane.get("original_group_idx")
        legacy_slice = plane.get("legacy_roi_slice_index"); target_slice = plane.get("target_reference_slice_index")
        if not all(isinstance(value, int) for value in (native_group, original_group, legacy_slice, target_slice)):
            raise ValueError("derivation plane requires integer group and slice identities")
        if native_group in seen_native or target_slice in seen_target or not 0 <= target_slice < reference.shape[2] or not 0 <= legacy_slice < outer.shape[2]:
            raise ValueError("derivation plane group/slice identity is duplicate or out of range")
        if standard_groups.get(native_group) != original_group:
            raise ValueError("derivation plane group identity disagrees with the standardization manifest")
        source_x, source_y, source_z = _plane_indices(legacy_affine=outer.affine, target_affine=reference.affine,
                                                        legacy_slice=legacy_slice, target_slice=target_slice,
                                                        target_shape=reference.shape[:2])
        if source_x.min() < 0 or source_y.min() < 0 or source_x.max() >= outer.shape[0] or source_y.max() >= outer.shape[1]:
            raise ValueError("Final5 target crop lies outside legacy ROI bounds")
        for name, image in legacy.items():
            if name == "sax_composite_reference.nii.gz":
                continue
            source = np.asarray(image.dataobj)
            source_plane = source[:, :, legacy_slice]
            copied = source[source_x, source_y, source_z]
            if name != "Segmentation_3-label_point.nii.gz" and np.count_nonzero(copied) != np.count_nonzero(source_plane):
                raise ValueError("Final5 crop clips a legacy myocardium ROI voxel")
            if name == "Segmentation_3-label_point.nii.gz" and np.count_nonzero(copied) != np.count_nonzero(source_plane):
                raise ValueError("Final5 crop clips a legacy AHA landmark")
            derived[name][:, :, target_slice] = copied
        seen_native.add(native_group); seen_target.add(target_slice)
        manifest_planes.append({
            "native_group_idx": native_group, "original_group_idx": original_group,
            "legacy_roi_slice_index": legacy_slice, "target_reference_slice_index": target_slice,
            "legacy_index_origin_xy": [int(source_x[0, 0]), int(source_y[0, 0])],
            "legacy_index_step_x": [int(source_x[1, 0] - source_x[0, 0]) if source_x.shape[0] > 1 else 0, int(source_y[1, 0] - source_y[0, 0]) if source_y.shape[0] > 1 else 0],
            "legacy_index_step_y": [int(source_x[0, 1] - source_x[0, 0]) if source_x.shape[1] > 1 else 0, int(source_y[0, 1] - source_y[0, 0]) if source_y.shape[1] > 1 else 0],
        })
    if seen_target != set(range(reference.shape[2])):
        raise ValueError("derivation plane records do not cover every Final5 reference slice")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        for name, values in derived.items():
            nib.save(nib.Nifti1Image(values, reference.affine), str(staging / name))
        shutil.copy2(reference_path, staging / "sax_composite_reference.nii.gz")
        _write_qc_montage(staging / "roi_qc_montage.png", np.asarray(reference.dataobj),
                          derived["Segmentation_1-out-label.nii.gz"] > 0,
                          derived["Segmentation_2-in-label.nii.gz"] > 0)
        manifest = {
            "schema": DERIVED_SCHEMA, "subject": payload["subject"], "resampling": False,
            "derivation_config": _record(config_path), "legacy_roi": {name: _record(path) for name, path in legacy_paths.items()},
            "final5_sax_reference": _record(reference_path), "standardization_manifest": _record(_verified_file(payload["standardization_manifest"], "standardization_manifest")),
            "planes": manifest_planes, "outputs": {},
        }
        for name in ROI_FILENAMES:
            manifest["outputs"][name] = _record(staging / name)
        manifest["outputs"]["roi_qc_montage.png"] = _record(staging / "roi_qc_montage.png")
        write_immutable_json(staging / "roi_derivation_manifest.json", manifest)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def derivation_readiness(config: str | Path) -> dict[str, str]:
    """Return an explicit local-input state without creating a bundle."""
    try:
        payload = _load_config(config)
        _verified_file(payload.get("final5_sax_reference"), "final5_sax_reference")
        _standard_groups(payload.get("standardization_manifest"), payload["subject"])
        legacy_root = Path(str(payload.get("legacy_roi", {}).get("path", "")))
        if not all((legacy_root / name).is_file() for name in ROI_FILENAMES):
            raise FileNotFoundError(legacy_root)
    except (FileNotFoundError, ValueError) as exc:
        return {"status": "INPUT_PENDING", "reason": str(exc)}
    return {"status": "READY"}
