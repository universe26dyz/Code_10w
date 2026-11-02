"""Exact-original-MultiMap full-FOV native reference generation for Q001A."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from .contracts import REQUIRED_SEMANTICS, sha256, verify_q001_input_bundle


STACKS = ("sax", "2ch", "4ch")


def _quote(value: Path) -> str: return str(value).replace("'", "''")


def generate_mapping(input_root: str | Path, mapping_root: str | Path, *, matlab: str = "matlab") -> None:
    """Use the existing original matcher after a metadata-only compatibility copy.

    The source full-FOV MAT remains read-only; only its semantic label is adapted
    in a disposable mapping-local copy because the legacy wrapper predates Q001.
    """
    verify_q001_input_bundle(input_root); root, mapping = Path(input_root), Path(mapping_root)
    if mapping.exists() and any(mapping.iterdir()): raise FileExistsError(f"Refusing non-empty Q001 mapping root: {mapping}")
    mapping.mkdir(parents=True, exist_ok=True)
    wrapper = Path(__file__).resolve().parents[2] / "trad/evaluation/scmr/matlab"
    for stack in STACKS:
        source, target, compat = root / "full_fov_preprocessed/CYJ" / stack / "preprocessed.mat", mapping / f"{stack}_mapping.mat", mapping / f"{stack}_matcher_compat.mat"
        command = f"S=load('{_quote(source)}'); assert(strcmp(char(S.final_data_semantics),'{REQUIRED_SEMANTICS}')); S.final_data_semantics='MP-PCA(MIND_mag_reg)'; save('{_quote(compat)}','-struct','S','-v7.3'); addpath('{_quote(wrapper)}'); build_native_reference_maps('{_quote(compat)}','{_quote(target)}');"
        subprocess.run([matlab, "-batch", command], check=True)
        if not target.is_file(): raise RuntimeError(f"Original matcher did not create {target}")


def package_reference(input_root: str | Path, mapping_root: str | Path, output: str | Path) -> dict:
    manifest = verify_q001_input_bundle(input_root); mapping, output = Path(mapping_root), Path(output)
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"Refusing non-empty Q001 native reference root: {output}")
    output.mkdir(parents=True, exist_ok=True); stacks = {}
    try: from scipy.io import loadmat
    except ImportError as exc: raise RuntimeError("scipy is required to package Q001 native reference.") from exc
    for stack in STACKS:
        source = mapping / f"{stack}_mapping.mat"; data = loadmat(source)
        required = {"t1_ms", "t2_ms", "valid_mask", "group_idx"}
        if not required.issubset(data): raise ValueError(f"{source} lacks original matcher outputs.")
        t1, t2, valid, groups = np.asarray(data["t1_ms"], dtype=np.float32), np.asarray(data["t2_ms"], dtype=np.float32), np.asarray(data["valid_mask"], dtype=bool), np.asarray(data["group_idx"], dtype=np.int64).reshape(-1)
        expected = manifest["stacks"][stack]
        if t1.shape != (expected["group_count"], *expected["full_image_shape_rows_cols"]) or t2.shape != t1.shape or valid.shape != t1.shape: raise ValueError(f"{stack} full-FOV mapping shape/provenance mismatch.")
        target = output / f"{stack}.npz"; np.savez_compressed(target, t1_ms=t1, t2_ms=t2, valid_mask=valid, group_idx=groups)
        stacks[stack] = {"map": target.name, "map_sha256": sha256(target), "source_preprocessed_mat_sha256": expected["preprocessed_mat"]["sha256"], "shape_group_row_col": list(t1.shape)}
    record = {"schema": "q001_full_fov_native_reference/v1", "subject_id": "CYJ", "spatial_mode": "full_fov", "preprocessing_semantics": REQUIRED_SEMANTICS, "map_units": "ms", "dictionary_provenance": "original MultiMap function_T1T2_10HB_bssfp; authoritative grid/protocol", "stacks": stacks}
    (output / "native_reference_manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--input-root", required=True); parser.add_argument("--mapping-root", required=True); parser.add_argument("--output", required=True); parser.add_argument("--skip-matlab", action="store_true")
    args = parser.parse_args()
    if not args.skip_matlab: generate_mapping(args.input_root, args.mapping_root)
    print(json.dumps(package_reference(args.input_root, args.mapping_root, args.output), indent=2))


if __name__ == "__main__": main()
