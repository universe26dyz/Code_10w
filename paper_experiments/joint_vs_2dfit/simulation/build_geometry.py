"""Choose actual VPS87 geometry templates from the prepared real cohort."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np


def _mode(values):
    counts = Counter(values)
    return sorted(counts, key=lambda x: (-counts[x], str(x)))[0]


def select_modal_geometry_templates(prepared_root: str | Path, subjects: tuple[str, ...] = ("CYJ", "DYZ", "HHZ", "HJL")) -> dict:
    root = Path(prepared_root)
    templates = {}
    for stack in ("sax", "2ch", "4ch"):
        candidates = []
        for sid in subjects:
            path = root / sid / stack / "manifest.json"
            data = json.loads(path.read_text())
            groups = [g for g in data["groups"] if float(g["tr_ms"]) == 2.61 and int(g["vps"]) == 87]
            if not groups:
                continue
            g0 = groups[0]
            # Modal acquisition geometry means the repeated acquisition
            # dimensions, FOV, spacing and thickness.  Orientation is retained
            # from the deterministic source tie-break below; it is not used to
            # split the cohort into singleton modes.
            key = (len(groups), tuple(g0["image_shape_rows_cols"]), tuple(g0["pixel_spacing_rc_mm"]), float(g0["slice_thickness_mm"]))
            candidates.append((sid, path, data, groups, key))
        if not candidates:
            raise FileNotFoundError(f"no compatible {stack} geometry manifests")
        key = _mode([c[-1] for c in candidates])
        chosen = sorted((c for c in candidates if c[-1] == key), key=lambda c: c[0])[0]
        sid, path, data, groups, _ = chosen
        templates[stack] = {
            "selection_rule": "modal group count/shape/spacing/thickness; orientation retained from lexicographic-source tie-break",
            "source_subject": sid,
            "source_manifest": str(path),
            "source_manifest_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
            "stack_name": stack,
            "group_count": len(groups),
            "groups": groups,
        }
    return templates


def recenter_geometry(template: dict, center_mm: np.ndarray) -> list[dict]:
    """Recenter a real orientation/FOV template on the synthetic anatomy.

    The row/column directions, FOV, spacing, slice thickness and group offsets
    are copied from the real manifest.  Only the global translation is changed
    to place the independent MRXCAT anatomy in its own world frame.
    """

    center_mm = np.asarray(center_mm, dtype=float)
    groups = template["groups"]
    first = groups[0]
    first_affine = np.asarray(first["affine_lps_rc"], dtype=float)
    row_axis = first_affine[:3, 0] / np.linalg.norm(first_affine[:3, 0])
    col_axis = first_affine[:3, 1] / np.linalg.norm(first_affine[:3, 1])
    normal = np.cross(row_axis, col_axis)
    normal /= np.linalg.norm(normal)
    rows, cols = first["image_shape_rows_cols"]
    base_center = first_affine[:3, 3] + first_affine[:3, 0] * (rows - 1) / 2 + first_affine[:3, 1] * (cols - 1) / 2
    out = []
    for idx, group in enumerate(groups):
        source_affine = np.asarray(group["affine_lps_rc"], dtype=float)
        source_center = source_affine[:3, 3] + source_affine[:3, 0] * (rows - 1) / 2 + source_affine[:3, 1] * (cols - 1) / 2
        offset = float(np.dot(source_center - base_center, normal))
        affine = np.eye(4, dtype=float)
        affine[:3, 0] = row_axis * float(group["pixel_spacing_rc_mm"][0])
        affine[:3, 1] = col_axis * float(group["pixel_spacing_rc_mm"][1])
        affine[:3, 2] = normal * float(group["slice_thickness_mm"])
        affine[:3, 3] = center_mm + normal * offset - affine[:3, 0] * (rows - 1) / 2 - affine[:3, 1] * (cols - 1) / 2
        out.append({
            "group_idx": idx,
            "image_shape_rows_cols": [int(rows), int(cols)],
            "affine_lps_rc": affine.tolist(),
            "pixel_spacing_rc_mm": list(group["pixel_spacing_rc_mm"]),
            "slice_thickness_mm": float(group["slice_thickness_mm"]),
            "source_group_idx": int(group["group_idx"]),
        })
    return out


def _normal(affine: np.ndarray) -> np.ndarray:
    normal = np.cross(affine[:3, 0], affine[:3, 1])
    return normal / np.linalg.norm(normal)


def _group_center(group: dict) -> np.ndarray:
    affine = np.asarray(group["affine_lps_rc"], dtype=float)
    rows, cols = (int(v) for v in group["image_shape_rows_cols"])
    return affine[:3, 3] + affine[:3, 0] * (rows - 1) / 2 + affine[:3, 1] * (cols - 1) / 2


def _rotation_from_to(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the proper rotation that maps one unit vector onto another."""

    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    source /= np.linalg.norm(source)
    target /= np.linalg.norm(target)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.dot(source, target))
    if sine < 1e-12:
        if cosine > 0:
            return np.eye(3)
        axis = np.zeros(3)
        axis[np.argmin(np.abs(source))] = 1.0
        axis -= source * np.dot(axis, source)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    skew = np.array([[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]])
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine * sine))


def apply_global_rigid_alignment(
    templates: dict[str, dict],
    anatomy_center_mm: np.ndarray,
    anatomy_long_axis: np.ndarray,
) -> tuple[dict[str, list[dict]], dict]:
    """Apply one rigid transform to every real stack, preserving inter-stack geometry.

    The source SAX plane normal is aligned to the mesh/PCA LV long axis.  No
    stack receives an independent rotation or translation.
    """

    if "sax" not in templates or not templates["sax"].get("groups"):
        raise ValueError("a SAX geometry template is required for global alignment")
    source_sax = np.asarray(templates["sax"]["groups"][0]["affine_lps_rc"], dtype=float)
    source_normal = _normal(source_sax)
    long_axis = np.asarray(anatomy_long_axis, dtype=float)
    long_axis /= np.linalg.norm(long_axis)
    rotation = _rotation_from_to(source_normal, long_axis)
    central_centers = []
    for template in templates.values():
        groups = template["groups"]
        central_centers.append(_group_center(groups[len(groups) // 2]))
    source_anchor = np.mean(central_centers, axis=0)
    target_center = np.asarray(anatomy_center_mm, dtype=float)
    translation = target_center - rotation @ source_anchor

    out: dict[str, list[dict]] = {}
    before_centers, after_centers = [], []
    for stack, template in templates.items():
        groups = []
        for group in template["groups"]:
            affine = np.asarray(group["affine_lps_rc"], dtype=float)
            aligned = np.eye(4, dtype=float)
            aligned[:3, :3] = rotation @ affine[:3, :3]
            aligned[:3, 3] = rotation @ affine[:3, 3] + translation
            rows, cols = (int(v) for v in group["image_shape_rows_cols"])
            groups.append({
                "group_idx": int(group["group_idx"]),
                "image_shape_rows_cols": [rows, cols],
                "affine_lps_rc": aligned.tolist(),
                "pixel_spacing_rc_mm": [float(v) for v in group["pixel_spacing_rc_mm"]],
                "slice_thickness_mm": float(group["slice_thickness_mm"]),
                "source_group_idx": int(group["group_idx"]),
            })
            before_centers.append(_group_center(group))
            after_centers.append(_group_center(groups[-1]))
        out[stack] = groups
    before = np.asarray(before_centers)
    after = np.asarray(after_centers)
    pair_errors = []
    for idx in range(len(before)):
        for jdx in range(idx + 1, len(before)):
            pair_errors.append(abs(np.linalg.norm(before[idx] - before[jdx]) - np.linalg.norm(after[idx] - after[jdx])))
    normals = {stack: _normal(np.asarray(groups[0]["affine_lps_rc"], dtype=float)) for stack, groups in out.items()}
    qc = {
        "method": "single_rigid_rotation_source_sax_normal_to_mesh_lv_long_axis_plus_single_translation",
        "rotation_matrix": rotation.tolist(),
        "translation_mm": translation.tolist(),
        "source_anchor_mm": source_anchor.tolist(),
        "anatomy_center_mm": target_center.tolist(),
        "source_sax_normal": source_normal.tolist(),
        "mesh_lv_long_axis": long_axis.tolist(),
        "sax_normal_dot_lv_long_axis": float(np.dot(normals["sax"], long_axis)),
        "stack_plane_normals": {key: value.tolist() for key, value in normals.items()},
        "max_pairwise_center_distance_error_mm": float(max(pair_errors, default=0.0)),
    }
    if "2ch" in normals and "4ch" in normals:
        qc["2ch_4ch_plane_normal_dot"] = float(np.dot(normals["2ch"], normals["4ch"]))
    return out, qc
