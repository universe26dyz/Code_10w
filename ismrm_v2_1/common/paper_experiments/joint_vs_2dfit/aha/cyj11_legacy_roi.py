"""CYJ1.1 manual ROI admission on an explicitly declared native SAX lattice."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .cyj11_adapter import apply_roi_transform
from ..phase2_clean.contracts import sha256, write_immutable_json
from ..phase2_clean.final_native_grid_metrics import native_ras_affine
from evaluation_common.myocardium.build_native_bundle import myocardium_core_1px, myocardium_core_legacy
from evaluation_common.myocardium.legacy_masks import exact_native_plane_crop


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
                          target_shape: tuple[int, int], source_shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    x, y, z, evidence = exact_native_plane_crop(
        source_affine=source_affine, target_affine=target_affine,
        source_shape=source_shape, target_shape=target_shape,
    )
    if not np.all(z == int(source_slice)):
        raise ValueError("CYJ1.1 native plane does not map to its declared manual ROI slice")
    rows, columns = target_shape
    steps = []
    if rows > 1: steps.append((int(x[1, 0] - x[0, 0]), int(y[1, 0] - y[0, 0])))
    if columns > 1: steps.append((int(x[0, 1] - x[0, 0]), int(y[0, 1] - y[0, 0])))
    axes = {(-1, 0), (1, 0), (0, -1), (0, 1)}
    if any(step not in axes for step in steps) or len(set(steps)) != len(steps):
        raise ValueError("CYJ1.1 native plane is not a signed-permutation manual ROI lattice")
    return x, y, z, evidence


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


def derive_cyj11_native_roi_config(*, analysis_root: str | Path, subject: str,
                                   source_roi: str | Path, output: str | Path) -> Path:
    """Derive one exact manual-ROI source slice for every native SAX group."""
    import nibabel as nib
    from .cyj11_adapter import cyj11_native_stack_root, load_cyj11_native_sax

    root, source_root, output = Path(analysis_root), Path(source_roi), Path(output)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    prepared, native_root = root / "prepared" / subject / "sax" / "observations.npz", cyj11_native_stack_root(root, subject)
    source_paths = {name: source_root / name for name in ROI_FILENAMES}
    if not prepared.is_file() or not all(path.is_file() for path in source_paths.values()):
        raise FileNotFoundError("CYJ1.1 config derivation requires prepared SAX and canonical four-file ROI source")
    source = {name: nib.load(str(path)) for name, path in source_paths.items()}
    outer = source["Segmentation_1-out-label.nii.gz"]
    triad = ("Segmentation_1-out-label.nii.gz", "Segmentation_2-in-label.nii.gz", "Segmentation_3-label_point.nii.gz")
    if any(image.shape != outer.shape or not np.allclose(image.affine, outer.affine, atol=1e-6, rtol=0)
           for name, image in source.items() if name in triad):
        raise ValueError("CYJ1.1 outer/inner/point manual ROI geometry differs")
    _composite_to_manual_index_transform(outer, source["sax_composite_reference.nii.gz"])
    native = load_cyj11_native_sax(native_root, prepared)
    planes: list[dict[str, int]] = []
    used_slices: set[int] = set()
    for index, group in enumerate(map(int, native["group_idx"])):
        shape = tuple(map(int, native["t1_ms"][index].shape))
        target_affine = native_ras_affine(native["affine_lps_rc"][index])
        x, y, z, _ = exact_native_plane_crop(
            source_affine=outer.affine, target_affine=target_affine,
            source_shape=tuple(map(int, outer.shape)), target_shape=shape,
        )
        slice_ids = np.unique(z)
        if slice_ids.size != 1:
            raise ValueError("CYJ1.1 native plane does not occupy one manual ROI slice")
        source_slice = int(slice_ids[0])
        x, y, z, _ = _native_plane_indices(outer.affine, target_affine, source_slice,
                                             shape, tuple(map(int, outer.shape)))
        if x.min() < 0 or y.min() < 0 or x.max() >= outer.shape[0] or y.max() >= outer.shape[1]:
            raise ValueError("CYJ1.1 native crop lies outside manual ROI bounds")
        if source_slice in used_slices:
            raise ValueError("CYJ1.1 native groups ambiguously map to one manual ROI slice")
        used_slices.add(source_slice)
        planes.append({"native_group_idx": group, "roi_slice_index": source_slice})
    return write_immutable_json(output, {
        "schema": DERIVATION_SCHEMA,
        "subject": subject,
        "prepared_sax": _record(prepared),
        "source_roi": {"path": str(source_root.resolve())},
        "native_sax_root": str(native_root.resolve()),
        "planes": planes,
        "derivation": "exact native-to-manual integer signed-permutation lattice; no interpolation",
        "immutable_output": True,
    })


def derive_cyj11_native_roi_bundle(*, config: str | Path, output: str | Path) -> Path:
    """Copy manual ROI pixels to every explicit native SAX plane with no resampling."""
    import nibabel as nib
    from .cyj11_adapter import load_cyj11_native_sax
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
        copied_by_group: dict[int, dict[str, np.ndarray]] = {}
        for item in planes:
            if not isinstance(item, dict) or not isinstance(item.get("native_group_idx"), int) or not isinstance(item.get("roi_slice_index"), int):
                raise ValueError("CYJ1.1 ROI derivation plane requires native group and ROI slice integers")
            group, source_slice = int(item["native_group_idx"]), int(item["roi_slice_index"])
            if group in seen or group not in index_by_group or not 0 <= source_slice < outer.shape[2]:
                raise ValueError("CYJ1.1 ROI derivation plane identity is duplicate or out of range")
            index = index_by_group[group]; shape = tuple(map(int, native["t1_ms"][index].shape))
            x, y, z, lattice_evidence = _native_plane_indices(
                outer.affine, native_ras_affine(native["affine_lps_rc"][index]), source_slice,
                shape, tuple(map(int, outer.shape)),
            )
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
                                    "plane": {"path": str(final_plane_path.resolve()), "sha256": sha256(plane_path)},
                                    "lattice_evidence": lattice_evidence})
            copied_by_group[group] = copied
            seen.add(group)
        if seen != set(index_by_group):
            raise ValueError("CYJ1.1 ROI derivation does not cover each native SAX group")
        group_order = [int(group) for group in native["group_idx"]]
        outer_stack = np.stack([copied_by_group[group]["outer"] > 0 for group in group_order])
        inner_stack = np.stack([copied_by_group[group]["inner"] > 0 for group in group_order])
        points_stack = np.stack([copied_by_group[group]["points"] for group in group_order])
        spacing = np.asarray(native["pixel_spacing_rc_mm"], float)
        if not np.allclose(spacing, spacing[0], atol=1e-6, rtol=0):
            raise ValueError("CYJ1.1 SCMR myocardium bundle requires one shared SAX in-plane spacing")
        blood = inner_stack & outer_stack
        full = outer_stack & ~blood
        core_legacy = myocardium_core_legacy(full, tuple(map(float, spacing[0][:2])))
        core_1px = myocardium_core_1px(full)
        np.savez_compressed(
            staging / "sax_myocardium_masks.npz",
            group_idx=np.asarray(group_order, dtype=np.int64),
            outer_filled=outer_stack.astype(np.uint8), inner_filled=inner_stack.astype(np.uint8),
            blood_pool=blood.astype(np.uint8), myocardium_full=full.astype(np.uint8),
            myocardium_core_legacy=core_legacy.astype(np.uint8), myocardium_core_1px=core_1px.astype(np.uint8),
            landmark_labels=points_stack,
        )
        write_immutable_json(staging / "roi_derivation_manifest.json", {
            "schema": ROI_BUNDLE_SCHEMA, "subject": payload["subject"], "resampling": False,
            "derivation_config": _record(config), "prepared_sax": _record(prepared),
            "native_sax": {name: _record(native_root / name) for name in ("native_maps.npz", "geometry.npz", "manifest.json")},
            "source_roi": {name: _record(path) for name, path in source_paths.items()}, "planes": manifest_planes,
            "composite_index_to_manual_index": composite_transform.tolist(),
            "native_mapping": "all target pixels must map to an integer signed-permutation source ROI lattice; no interpolation",
            "immutable_output": True,
        })
        write_immutable_json(staging / "manifest.json", {
            "schema": "exact_native_myocardium_bundle/v2", "status": "PASS", "subject_id": payload["subject"],
            "method": "SCMR signed axis permutation/reversal plus integer crop; no interpolation",
            "source_roi_derivation_manifest": {"path": str((output / "roi_derivation_manifest.json").resolve()),
                                                 "sha256": sha256(staging / "roi_derivation_manifest.json")},
            "sax_myocardium_masks": {"path": str((output / "sax_myocardium_masks.npz").resolve()),
                                       "sha256": sha256(staging / "sax_myocardium_masks.npz")},
            "shape_group_row_col": list(full.shape), "group_idx": group_order,
            "spacing_rc_mm": list(map(float, spacing[0][:2])),
            "core_definitions": {
                "myocardium_core_legacy": "distance_transform_edt(myocardium_full, sampling=row/col spacing) > 1.0 mm",
                "myocardium_core_1px": "binary_erosion(myocardium_full, structure=ones((3,3)), iterations=1, border_value=0), independently per SAX slice",
            },
            "inner_outside_outer_voxels_trimmed": int(np.count_nonzero(inner_stack & ~outer_stack)),
            "per_slice_counts": [{"group_idx": group, "myocardium_full": int(full[index].sum()),
                                  "myocardium_core_legacy": int(core_legacy[index].sum()),
                                  "myocardium_core_1px": int(core_1px[index].sum())}
                                 for index, group in enumerate(group_order)],
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
