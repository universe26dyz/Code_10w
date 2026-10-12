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
            "selection_rule": "modal group count/shape/spacing/thickness/orientation; lexicographic subject tie-break",
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
