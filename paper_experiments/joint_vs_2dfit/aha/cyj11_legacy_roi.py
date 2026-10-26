"""CYJ1.1 manual ROI admission on an explicitly declared native SAX lattice."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .cyj11_adapter import apply_roi_transform
from ..phase2_clean.contracts import sha256
from ..phase2_clean.final_native_grid_metrics import native_ras_affine


ROI_FILENAMES = ("Segmentation_1-out-label.nii.gz", "Segmentation_2-in-label.nii.gz",
                 "Segmentation_3-label_point.nii.gz", "sax_composite_reference.nii.gz")
DERIVATION_SCHEMA = "cyj11_native_roi_derivation_config/v1"
ROI_BUNDLE_SCHEMA = "cyj11_native_roi_bundle/v1"


def mapped_manual_roi_plane(outer_volume: Any, inner_volume: Any, *, roi_slice_index: int,
                            roi_transform: str, expected_shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Return one exact manual ROI plane; interpolation and implicit orientation are impossible."""
    outer, inner = np.asarray(outer_volume), np.asarray(inner_volume)
    if outer.shape != inner.shape or outer.ndim != 3:
        raise ValueError("CYJ1.1 outer/inner manual ROI volumes must share one 3-D geometry")
    if not isinstance(roi_slice_index, int) or not 0 <= roi_slice_index < outer.shape[2]:
        raise ValueError("CYJ1.1 ROI slice index is outside the manual ROI volume")
    return {
        "outer": apply_roi_transform(outer[:, :, roi_slice_index], roi_transform, expected_shape) > 0,
        "inner": apply_roi_transform(inner[:, :, roi_slice_index], roi_transform, expected_shape) > 0,
    }


def load_cyj11_roi_plane(bundle_root: str | Path, *, group: int,
                         expected_shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Load one immutable, already-native-grid manual ROI plane by declared group ID."""
    root = Path(bundle_root)
    manifest_path = root / "roi_derivation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "cyj11_native_roi_bundle/v1":
        raise ValueError("CYJ1.1 ROI bundle schema is invalid")
    entries = [item for item in manifest.get("planes", []) if isinstance(item, dict)
               and item.get("native_group_idx") == int(group)]
    if len(entries) != 1:
        raise ValueError(f"CYJ1.1 ROI bundle lacks one unique native group {group}")
    entry = entries[0]
    record = entry.get("plane")
    if not isinstance(record, dict):
        raise ValueError("CYJ1.1 ROI bundle plane provenance is missing")
    plane_path = Path(str(record.get("path", "")))
    if not plane_path.is_file() or record.get("sha256") != sha256(plane_path):
        raise ValueError("CYJ1.1 ROI bundle plane path/SHA256 mismatch")
    with np.load(plane_path, allow_pickle=False) as archive:
        if any(name not in archive for name in ("outer", "inner", "points")):
            raise ValueError("CYJ1.1 ROI bundle plane arrays are incomplete")
        result = {name: np.asarray(archive[name]) for name in ("outer", "inner", "points")}
    if any(value.shape != tuple(expected_shape) or value.ndim != 2 for value in result.values()):
        raise ValueError("CYJ1.1 ROI bundle/native SAX plane shape mismatch")
    return {"outer": result["outer"] > 0, "inner": result["inner"] > 0, "points": result["points"]}


def _record(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _verified_record(record: Any, label: str) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"CYJ1.1 ROI derivation requires {label} provenance")
    path = Path(str(record.get("path", "")))
    if not path.is_file() or record.get("sha256") != sha256(path):
        raise ValueError(f"CYJ1.1 ROI derivation {label} path/SHA256 mismatch")
    return path


def _native_plane_indices(source_affine: np.ndarray, target_affine: np.ndarray, source_slice: int,
                          target_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, columns = target_shape
    row, column = np.indices((rows, columns), dtype=float)
    target = np.vstack((row.ravel(), column.ravel(), np.zeros(row.size), np.ones(row.size)))
    source = np.linalg.inv(np.asarray(source_affine, float)) @ np.asarray(target_affine, float) @ target
    rounded = np.rint(source[:3])
    if not np.allclose(source[:3], rounded, atol=1e-5, rtol=0):
        raise ValueError("CYJ1.1 native plane is not an integer crop/orientation of the manual ROI")
    x, y, z = (value.astype(int).reshape(target_shape) for value in rounded)
    if not np.all(z == int(source_slice)):
        raise ValueError("CYJ1.1 native plane does not map to its declared manual ROI slice")
    steps = []
    if rows > 1: steps.append((int(x[1, 0] - x[0, 0]), int(y[1, 0] - y[0, 0])))
    if columns > 1: steps.append((int(x[0, 1] - x[0, 0]), int(y[0, 1] - y[0, 0])))
    axes = {(-1, 0), (1, 0), (0, -1), (0, 1)}
    if any(step not in axes for step in steps) or len(set(steps)) != len(steps):
        raise ValueError("CYJ1.1 native plane is not a signed-permutation manual ROI lattice")
    return x, y, z


def _composite_to_manual_index_transform(manual: Any, composite: Any) -> np.ndarray:
    """Admit only an exact integer signed-permutation composite provenance grid."""
    if len(manual.shape) != 3 or len(composite.shape) != 3:
        raise ValueError("CYJ1.1 manual ROI and composite reference must be 3-D")
    transform = np.linalg.inv(np.asarray(manual.affine, float)) @ np.asarray(composite.affine, float)
    rounded = np.rint(transform)
    if not np.allclose(transform, rounded, atol=1e-5, rtol=0):
        raise ValueError("CYJ1.1 composite reference is not an integer manual-ROI lattice transform")
    linear = rounded[:3, :3].astype(int)
    if not (np.all(np.isin(linear, (-1, 0, 1)))
            and np.all(np.sum(np.abs(linear), axis=0) == 1)
            and np.all(np.sum(np.abs(linear), axis=1) == 1)
            and np.array_equal(rounded[3], (0, 0, 0, 1))):
        raise ValueError("CYJ1.1 composite reference is not a signed-permutation manual-ROI grid")
    corners = np.array(np.meshgrid(*[(0, size - 1) for size in composite.shape], indexing="ij")).reshape(3, -1)
    mapped = rounded @ np.vstack((corners, np.ones(corners.shape[1])))
    if not np.all((mapped[:3] >= 0) & (mapped[:3] < np.asarray(manual.shape)[:, None])):
        raise ValueError("CYJ1.1 composite reference extends outside manual ROI geometry")
    return rounded.astype(int)


def derive_cyj11_native_roi_bundle(*, config: str | Path, output: str | Path) -> Path:
    """Copy manual ROI pixels to every explicit native SAX plane with no resampling."""
    import nibabel as nib
    from .cyj11_adapter import load_cyj11_native_sax
    from ..phase2_clean.contracts import write_immutable_json

    config, output = Path(config), Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    payload = json.loads(config.read_text(encoding="utf-8"))
    if payload.get("schema") != DERIVATION_SCHEMA or not isinstance(payload.get("subject"), str):
        raise ValueError("invalid CYJ1.1 ROI derivation configuration identity")
    prepared = _verified_record(payload.get("prepared_sax"), "prepared SAX")
    source_root = Path(str(payload.get("source_roi", {}).get("path", "")))
    source_paths = {name: source_root / name for name in ROI_FILENAMES}
    if not all(path.is_file() for path in source_paths.values()):
        raise FileNotFoundError("CYJ1.1 source ROI lacks canonical four-file set")
    source = {name: nib.load(str(path)) for name, path in source_paths.items()}
    outer = source["Segmentation_1-out-label.nii.gz"]
    triad = ("Segmentation_1-out-label.nii.gz", "Segmentation_2-in-label.nii.gz", "Segmentation_3-label_point.nii.gz")
    if any(image.shape != outer.shape or not np.allclose(image.affine, outer.affine, atol=1e-6, rtol=0)
           for name, image in source.items() if name in triad):
        raise ValueError("CYJ1.1 outer/inner/point manual ROI geometry differs")
    composite_transform = _composite_to_manual_index_transform(outer, source["sax_composite_reference.nii.gz"])
    native_root = Path(str(payload.get("native_sax_root", "")))
    native = load_cyj11_native_sax(native_root, prepared)
    index_by_group = {int(group): index for index, group in enumerate(native["group_idx"])}
    planes = payload.get("planes")
    if not isinstance(planes, list) or len(planes) != len(index_by_group):
        raise ValueError("CYJ1.1 ROI derivation requires one explicit plane per native SAX group")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        seen: set[int] = set(); manifest_planes: list[dict[str, Any]] = []
        for item in planes:
            if not isinstance(item, dict) or not isinstance(item.get("native_group_idx"), int) or not isinstance(item.get("roi_slice_index"), int):
                raise ValueError("CYJ1.1 ROI derivation plane requires native group and ROI slice integers")
            group, source_slice = int(item["native_group_idx"]), int(item["roi_slice_index"])
            if group in seen or group not in index_by_group or not 0 <= source_slice < outer.shape[2]:
                raise ValueError("CYJ1.1 ROI derivation plane identity is duplicate or out of range")
            index = index_by_group[group]; shape = tuple(map(int, native["t1_ms"][index].shape))
            x, y, z = _native_plane_indices(outer.affine, native_ras_affine(native["affine_lps_rc"][index]), source_slice, shape)
            if x.min() < 0 or y.min() < 0 or x.max() >= outer.shape[0] or y.max() >= outer.shape[1]:
                raise ValueError("CYJ1.1 native crop lies outside manual ROI bounds")
            copied = {"outer": np.asarray(source["Segmentation_1-out-label.nii.gz"].dataobj)[x, y, z],
                      "inner": np.asarray(source["Segmentation_2-in-label.nii.gz"].dataobj)[x, y, z],
                      "points": np.asarray(source["Segmentation_3-label_point.nii.gz"].dataobj)[x, y, z]}
            for name, values in copied.items():
                original = np.asarray(source[{"outer": "Segmentation_1-out-label.nii.gz", "inner": "Segmentation_2-in-label.nii.gz", "points": "Segmentation_3-label_point.nii.gz"}[name]].dataobj)[:, :, source_slice]
                if np.count_nonzero(values) != np.count_nonzero(original):
                    raise ValueError(f"CYJ1.1 native crop clips manual ROI {name} voxels")
            plane_path = staging / "planes" / f"group{group:03d}.npz"; plane_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(plane_path, **copied)
            final_plane_path = output / "planes" / plane_path.name
            manifest_planes.append({"native_group_idx": group, "roi_slice_index": source_slice, "shape": list(shape),
                                    "plane": {"path": str(final_plane_path.resolve()), "sha256": sha256(plane_path)}})
            seen.add(group)
        if seen != set(index_by_group):
            raise ValueError("CYJ1.1 ROI derivation does not cover each native SAX group")
        write_immutable_json(staging / "roi_derivation_manifest.json", {
            "schema": ROI_BUNDLE_SCHEMA, "subject": payload["subject"], "resampling": False,
            "derivation_config": _record(config), "prepared_sax": _record(prepared),
            "native_sax": {name: _record(native_root / name) for name in ("native_maps.npz", "geometry.npz", "manifest.json")},
            "source_roi": {name: _record(path) for name, path in source_paths.items()}, "planes": manifest_planes,
            "composite_index_to_manual_index": composite_transform.tolist(),
            "native_mapping": "all target pixels must map to an integer signed-permutation source ROI lattice; no interpolation",
            "immutable_output": True,
        })
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def roi_derivation_readiness(config: str | Path) -> dict[str, str]:
    """Validate the static inputs for a future write without creating a bundle."""
    try:
        payload = json.loads(Path(config).read_text(encoding="utf-8"))
        if payload.get("schema") != DERIVATION_SCHEMA: raise ValueError("invalid CYJ1.1 ROI derivation schema")
        _verified_record(payload.get("prepared_sax"), "prepared SAX")
        source = Path(str(payload.get("source_roi", {}).get("path", "")))
        if not all((source / name).is_file() for name in ROI_FILENAMES): raise FileNotFoundError(source)
        if not Path(str(payload.get("native_sax_root", ""))).is_dir(): raise FileNotFoundError(payload.get("native_sax_root"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "INPUT_PENDING", "reason": str(exc)}
    return {"status": "READY"}
