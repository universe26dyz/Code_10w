"""Read-only contract verifier for the Q001 full-FOV native reference bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import REQUIRED_SEMANTICS, STACK_CONTRACT, sha256, verify_q001_input_bundle


def verify_full_fov_reference(reference_root: str | Path, input_root: str | Path) -> dict[str, Any]:
    """Verify package bytes, source-MAT provenance, and unmodified native grids."""

    root = Path(reference_root).resolve()
    manifest_path = root / "native_reference_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Q001 native-reference manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "schema": "q001_full_fov_native_reference/v1",
        "subject_id": "CYJ",
        "spatial_mode": "full_fov",
        "preprocessing_semantics": REQUIRED_SEMANTICS,
        "map_units": "ms",
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Q001 reference manifest {key} must be {expected!r}.")
    inputs = verify_q001_input_bundle(input_root)
    report: dict[str, Any] = {"schema": manifest["schema"], "reference_root": str(root), "manifest_sha256": sha256(manifest_path), "stacks": {}}
    for stack, (groups, _observations, rows_cols) in STACK_CONTRACT.items():
        record = manifest.get("stacks", {}).get(stack)
        if not isinstance(record, dict):
            raise ValueError(f"Reference manifest lacks stack {stack}.")
        source = root / str(record.get("map", ""))
        if not source.is_file():
            raise FileNotFoundError(f"Reference map missing: {source}")
        if sha256(source) != record.get("map_sha256"):
            raise ValueError(f"{stack} reference map SHA256 mismatch.")
        expected_mat_hash = inputs["stacks"][stack]["preprocessed_mat"]["sha256"]
        if record.get("source_preprocessed_mat_sha256") != expected_mat_hash:
            raise ValueError(f"{stack} source preprocessed MAT SHA256 does not match Q001 input manifest.")
        expected_shape = (groups, *rows_cols)
        if tuple(record.get("shape_group_row_col", ())) != expected_shape:
            raise ValueError(f"{stack} reference manifest shape differs from Q001 full-FOV contract.")
        with np.load(source, allow_pickle=False) as data:
            required_npz = {"t1_ms", "t2_ms", "valid_mask", "group_idx"}
            missing = required_npz.difference(data.files)
            if missing:
                raise ValueError(f"{stack} reference NPZ lacks fields {sorted(missing)}.")
            t1, t2 = np.asarray(data["t1_ms"]), np.asarray(data["t2_ms"])
            valid, group_idx = np.asarray(data["valid_mask"], dtype=bool), np.asarray(data["group_idx"], dtype=np.int64).reshape(-1)
        if t1.shape != expected_shape or t2.shape != expected_shape or valid.shape != expected_shape:
            raise ValueError(f"{stack} reference t1/t2/valid native shapes must be {expected_shape}.")
        if not np.isfinite(t1).all() or not np.isfinite(t2).all():
            raise ValueError(f"{stack} reference T1/T2 must be finite over the packaged native grid.")
        if not np.array_equal(group_idx, np.arange(groups, dtype=np.int64)):
            raise ValueError(f"{stack} reference group_idx must be contiguous zero-based.")
        report["stacks"][stack] = {"shape_group_row_col": list(expected_shape), "map_sha256": record["map_sha256"], "source_preprocessed_mat_sha256": expected_mat_hash, "valid_fraction": float(valid.mean())}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--input-root", required=True)
    args = parser.parse_args()
    print(json.dumps(verify_full_fov_reference(args.reference_root, args.input_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
