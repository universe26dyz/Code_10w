"""Convert a real MRXCAT/XCAT VTI label volume without changing its semantics."""

from __future__ import annotations

import base64
import re
import struct
import subprocess
import zlib
from pathlib import Path

import numpy as np

from .inspect_case import VTIInfo, parse_vti_header


def _source_bytes(source: str | Path, git_repo: str | Path | None = None) -> bytes:
    path = Path(source)
    if path.exists():
        return path.read_bytes()
    if git_repo is None:
        raise FileNotFoundError(path)
    return subprocess.check_output(["git", "-C", str(git_repo), "cat-file", "blob", f"HEAD:{source}"])


def _decode_array(raw: bytes, offset: int, compressed: bool) -> bytes:
    if not compressed:
        n = struct.unpack_from("<I", raw, offset)[0]
        return raw[offset + 4 : offset + 4 + n]
    nblocks, block_size, last_block_size = struct.unpack_from("<3I", raw, offset)
    sizes = struct.unpack_from(f"<{nblocks}I", raw, offset + 12)
    pos = offset + 12 + 4 * nblocks
    chunks = []
    for size in sizes:
        chunks.append(zlib.decompress(raw[pos : pos + size]))
        pos += size
    decoded = b"".join(chunks)
    expected = block_size * max(0, nblocks - 1) + last_block_size
    if len(decoded) != expected:
        raise ValueError(f"VTI compressed array length {len(decoded)} != expected {expected}")
    return decoded


def load_vti_array(source: str | Path, array_name: str = "labels", git_repo: str | Path | None = None) -> tuple[np.ndarray, VTIInfo]:
    blob = _source_bytes(source, git_repo)
    marker = blob.find(b"<AppendedData")
    if marker < 0:
        raise ValueError(f"VTI has no AppendedData section: {source}")
    prefix = blob[:marker].decode("utf-8", errors="ignore")
    info = parse_vti_header(prefix, str(source))
    data_start = blob.find(b"_", marker) + 1
    data_end = blob.find(b"</AppendedData>", data_start)
    if data_start <= 0 or data_end < 0:
        raise ValueError(f"malformed VTI appended data: {source}")
    appended = base64.b64decode(re.sub(rb"\s+", b"", blob[data_start:data_end]))
    array_re = re.compile(rb'<DataArray\b[^>]*\bName="([^"]+)"[^>]*?(?:offset="([0-9]+)")?[^>]*/?>')
    attrs = {}
    for match in array_re.finditer(blob[:marker]):
        name = match.group(1).decode()
        offset = int(match.group(2) or 0)
        attrs[name] = offset
    if array_name not in attrs:
        raise KeyError(f"VTI array {array_name!r} not found in {source}")
    decoded = _decode_array(appended, attrs[array_name], b"compressor=\"vtkZLibDataCompressor\"" in blob[:marker])
    # MRXCAT stores PointData arrays as Float64 and VTK's x-fastest order.
    values = np.frombuffer(decoded, dtype="<f8")
    expected = int(np.prod(info.shape))
    if values.size != expected:
        raise ValueError(f"VTI array {array_name} has {values.size} values, expected {expected}")
    return values.reshape(info.shape, order="F"), info


def convert_labels(source: str | Path, output: str | Path, git_repo: str | Path | None = None) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    labels, info = load_vti_array(source, "labels", git_repo)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        labels=labels.astype(np.int16, copy=False),
        origin=np.asarray(info.origin, dtype=np.float64),
        spacing=np.asarray(info.spacing, dtype=np.float64),
        direction=np.asarray(info.direction, dtype=np.float64).reshape(3, 3),
    )
    return {"output": str(output), "shape": list(labels.shape), "origin": list(info.origin), "spacing": list(info.spacing)}
