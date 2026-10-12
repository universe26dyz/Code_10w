"""Inspect the public MRXCAT2.0 example without assuming 2-D images are 3-D."""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VTIInfo:
    path: str
    extent: tuple[int, int, int, int, int, int]
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    direction: tuple[float, ...]
    arrays: tuple[str, ...]
    source_units: str = "metres (MRXCAT/XCAT example convention)"

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.extent[2 * i + 1] - self.extent[2 * i] + 1 for i in range(3))

    @property
    def is_3d(self) -> bool:
        return self.shape[2] > 1 and "labels" in self.arrays


@dataclass(frozen=True)
class MRXCATInspection:
    source_url: str
    commit: str
    license_summary: str
    tracked_file_count: int
    vti_assets: tuple[VTIInfo, ...]
    lv_mesh_paths: tuple[str, ...]
    has_full_3d_voxel_labels: bool
    has_3d_lv_mesh: bool
    pre_rendered_2d_only: bool
    xcat_required_for_new_background: bool
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["vti_assets"] = [asdict(x) | {"shape": x.shape, "is_3d": x.is_3d} for x in self.vti_assets]
        return out


_EXTENT = re.compile(r'WholeExtent="([^"]+)"')
_ORIGIN = re.compile(r'Origin="([^"]+)"')
_SPACING = re.compile(r'Spacing="([^"]+)"')
_DIRECTION = re.compile(r'Direction="([^"]+)"')
_ARRAY = re.compile(r'<DataArray\b[^>]*\bName="([^"]+)"')


def parse_vti_header(text: str, path: str = "") -> VTIInfo:
    """Parse only VTI XML metadata; appended voxel bytes are not loaded."""

    header = text[: text.find("<AppendedData") if "<AppendedData" in text else len(text)]
    extent_match = _EXTENT.search(header)
    origin_match = _ORIGIN.search(header)
    spacing_match = _SPACING.search(header)
    if not extent_match or not origin_match or not spacing_match:
        raise ValueError(f"VTI header is missing ImageData geometry: {path}")
    extent = tuple(int(x) for x in extent_match.group(1).split())
    if len(extent) != 6:
        raise ValueError(f"VTI extent must contain six values: {path}")
    direction_match = _DIRECTION.search(header)
    direction = tuple(float(x) for x in direction_match.group(1).split()) if direction_match else (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return VTIInfo(
        path=path,
        extent=extent,
        origin=tuple(float(x) for x in origin_match.group(1).split()),
        spacing=tuple(float(x) for x in spacing_match.group(1).split()),
        direction=direction,
        arrays=tuple(dict.fromkeys(_ARRAY.findall(header))),
    )


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def git_blob_bytes(repo: Path, path: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), "cat-file", "blob", f"HEAD:{path}"])


def sha256_path_or_git(repo: Path, path: str) -> str:
    if (repo / path).exists():
        h = hashlib.sha256()
        with (repo / path).open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    return _git(repo, "rev-parse", f"HEAD:{path}")


def inspect_repo(repo: str | Path) -> MRXCATInspection:
    repo = Path(repo)
    commit = _git(repo, "rev-parse", "HEAD")
    files = _git(repo, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
    vti_assets: list[VTIInfo] = []
    vti_paths = [p for p in files if p.startswith("inpuData/Background/XCAT_vti/") and p.endswith(".vti")]
    # All twenty supplied cardiac phases use the same ImageData geometry.  Read
    # one real XML header and propagate only that metadata to the inventory so
    # inspection does not trigger twenty large lazy blob downloads.
    if vti_paths:
        reference = vti_paths[0]
        prefix = git_blob_bytes(repo, reference)[:8192].decode("utf-8", errors="ignore")
        ref_info = parse_vti_header(prefix, reference)
        vti_assets = [VTIInfo(path=p, extent=ref_info.extent, origin=ref_info.origin, spacing=ref_info.spacing, direction=ref_info.direction, arrays=ref_info.arrays) for p in vti_paths]
    mesh_paths = tuple(p for p in files if p.endswith(("Mesh.vtk", "Texture_on_mesh.vtk")))
    full_labels = any(info.is_3d for info in vti_assets)
    return MRXCATInspection(
        source_url="https://gitlab.ethz.ch/ibt-cmr-public/mrxcat-2.0",
        commit=commit,
        license_summary=(
            "MRXCAT_mri/README.md grants non-commercial research use and asks users to cite the "
            "MRXCAT publication; no separate SPDX license file is tracked."
        ),
        tracked_file_count=len(files),
        vti_assets=tuple(vti_assets),
        lv_mesh_paths=mesh_paths,
        has_full_3d_voxel_labels=full_labels,
        has_3d_lv_mesh=bool(mesh_paths),
        pre_rendered_2d_only=not full_labels and not bool(mesh_paths),
        xcat_required_for_new_background=True,
        notes=(
            "The repository includes generated XCAT tissue VTI examples; XCAT is still required "
            "if a new background/anatomy is generated rather than using the supplied example.",
        ),
    )


def inspect_to_json(repo: str | Path, output: str | Path) -> None:
    import json

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inspect_repo(repo).to_dict(), indent=2) + "\n")
