"""Read MRXCAT's supplied tetrahedral LV mesh and voxelize it deterministically."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .convert_labels import _source_bytes


@dataclass(frozen=True)
class TetraMesh:
    points_mm: np.ndarray
    cells: np.ndarray
    point_labels: np.ndarray


def load_tetra_mesh(source: str | Path, git_repo: str | Path | None = None) -> TetraMesh:
    """Load the ASCII VTK unstructured-grid mesh without mutating MRXCAT."""

    tokens = _source_bytes(source, git_repo).decode("utf-8").split()
    point_at = tokens.index("POINTS")
    point_count = int(tokens[point_at + 1])
    start = point_at + 3
    points_m = np.asarray(tokens[start : start + 3 * point_count], dtype=float).reshape(point_count, 3)
    cell_at = tokens.index("CELLS", start + 3 * point_count)
    cell_count = int(tokens[cell_at + 1])
    cursor = cell_at + 3
    cells = np.empty((cell_count, 4), dtype=np.int32)
    for idx in range(cell_count):
        if int(tokens[cursor]) != 4:
            raise ValueError("Mesh.vtk contains a non-tetrahedral cell")
        cells[idx] = np.asarray(tokens[cursor + 1 : cursor + 5], dtype=np.int32)
        cursor += 5
    type_at = tokens.index("CELL_TYPES", cursor)
    cell_types = np.asarray(tokens[type_at + 2 : type_at + 2 + cell_count], dtype=np.int32)
    if not np.all(cell_types == 10):
        raise ValueError("Mesh.vtk must contain only VTK_TETRA cells")
    labels_at = tokens.index("SCALARS", type_at + 2 + cell_count)
    if tokens[labels_at + 1] != "labels":
        raise ValueError("Mesh.vtk has no labelled LV point data")
    lookup_at = tokens.index("LOOKUP_TABLE", labels_at)
    point_labels = np.asarray(tokens[lookup_at + 2 : lookup_at + 2 + point_count], dtype=float)
    if point_labels.shape != (point_count,) or not np.all(np.isfinite(point_labels)):
        raise ValueError("Mesh.vtk has invalid point labels")
    # The supplied mesh is the Patient1 *left-ventricle* finite-element mesh
    # (MRXCAT README), while its point labels are FEM tags 0..5 rather than
    # tissue compartments.  We therefore voxelize the complete tetrahedral
    # volume as myocardium and never infer a blood-pool label from these tags.
    return TetraMesh(points_mm=points_m * 1000.0, cells=cells, point_labels=point_labels)


def mesh_axes(mesh: TetraMesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return LV center and deterministic PCA long/transverse axes in mm."""

    center = mesh.points_mm.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov((mesh.points_mm - center).T))
    order = np.argsort(values)[::-1]
    long_axis = vectors[:, order[0]]
    transverse = vectors[:, order[1]]
    # Eigenvector signs are arbitrary.  Fix them reproducibly without changing
    # their geometry; the third axis is implied by the right-handed pair.
    if long_axis[np.argmax(np.abs(long_axis))] < 0:
        long_axis *= -1
    if transverse[np.argmax(np.abs(transverse))] < 0:
        transverse *= -1
    return center, long_axis / np.linalg.norm(long_axis), transverse / np.linalg.norm(transverse)


def voxelize_tetrahedra(mesh: TetraMesh, spacing_mm: float = 1.0, padding_mm: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voxelize the supplied tetrahedra on an isotropic grid using exact barycentric containment."""

    if spacing_mm <= 0 or padding_mm < 0:
        raise ValueError("spacing_mm must be positive and padding_mm non-negative")
    lower = np.floor((mesh.points_mm.min(axis=0) - padding_mm) / spacing_mm) * spacing_mm
    upper = np.ceil((mesh.points_mm.max(axis=0) + padding_mm) / spacing_mm) * spacing_mm
    shape = np.rint((upper - lower) / spacing_mm).astype(int) + 1
    mask = np.zeros(tuple(shape), dtype=bool)
    for cell in mesh.cells:
        vertices = mesh.points_mm[cell]
        tet_min = np.maximum(np.ceil((vertices.min(axis=0) - lower) / spacing_mm).astype(int), 0)
        tet_max = np.minimum(np.floor((vertices.max(axis=0) - lower) / spacing_mm).astype(int), shape - 1)
        if np.any(tet_min > tet_max):
            continue
        xx, yy, zz = np.meshgrid(
            np.arange(tet_min[0], tet_max[0] + 1),
            np.arange(tet_min[1], tet_max[1] + 1),
            np.arange(tet_min[2], tet_max[2] + 1),
            indexing="ij",
        )
        idx = np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=1)
        points = lower[None, :] + idx * spacing_mm
        basis = (vertices[1:] - vertices[0]).T
        try:
            bary = np.linalg.solve(basis, (points - vertices[0]).T).T
        except np.linalg.LinAlgError as exc:
            raise ValueError("degenerate tetrahedron in Mesh.vtk") from exc
        inside = np.all(bary >= -1e-8, axis=1) & (bary.sum(axis=1) <= 1.0 + 1e-8)
        mask[idx[inside, 0], idx[inside, 1], idx[inside, 2]] = True
    if not np.any(mask):
        raise ValueError("Mesh.vtk voxelization produced an empty myocardium")
    return mask, lower, np.repeat(float(spacing_mm), 3)
