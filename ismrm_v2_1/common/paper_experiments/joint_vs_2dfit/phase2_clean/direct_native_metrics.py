"""Build immutable native-plane metric inputs from direct-native prediction bundles."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import sha256, write_immutable_json
from .final_cyj_reporting import PARAMETERS, _load_all_verified_native_cache_groups
from .mask_sensitivity_audit import MASKS, run_mask_sensitivity_audit


def build_direct_native_metrics(*, bundle_root: str | Path, r0_native_maps_root: str | Path, sparse_manifest: str | Path, mask_root: str | Path, output: str | Path) -> Path:
    """Evaluate a bundle only on explicitly supplied current-native masks."""
    bundle_root, mask_root, output = Path(bundle_root), Path(mask_root), Path(output)
    manifest_path = bundle_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "direct_native_prediction_bundle/v1":
        raise ValueError("bundle_root is not a direct-native prediction bundle")
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        records: list[dict[str, Any]] = []
        by_stack = {stack: _load_all_verified_native_cache_groups(r0_native_maps_root, sparse_manifest, stack) for stack in ("sax", "2ch", "4ch")}
        for plane in manifest["planes"]:
            stack, group = plane["stack"], int(plane["original_group_idx"])
            cached = by_stack[stack]
            indices = np.flatnonzero(np.asarray(cached["group_idx"], int) == group)
            if len(indices) != 1:
                raise ValueError(f"{stack} group {group}: unavailable or ambiguous R0 native reference")
            index = int(indices[0])
            with np.load(bundle_root / plane["path"], allow_pickle=False) as prediction:
                for parameter in PARAMETERS:
                    support_path = mask_root / "planes" / parameter / stack / f"group{group:03d}.npz"
                    with np.load(support_path, allow_pickle=False) as masks:
                        missing = [name for name in MASKS if name not in masks.files]
                        if missing:
                            raise ValueError(f"{support_path}: missing masks {missing}")
                        values = {name: np.asarray(masks[name], bool) for name in MASKS}
                    reference = np.asarray(cached[f"{parameter}_ms"][index], float)
                    predicted = np.asarray(prediction[f"{parameter}_ms"], float)
                    if reference.shape != predicted.shape or any(value.shape != reference.shape for value in values.values()):
                        raise ValueError(f"{stack} group {group} {parameter}: prediction/reference/mask geometry mismatch")
                    target = staging / "metric_inputs" / "planes" / parameter / stack / f"group{group:03d}.npz"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(target, reference=reference, prediction=predicted, source_support=np.asarray(cached["source_support"][index], bool), **values)
                    records.append({"parameter": parameter, "stack": stack, "original_group_idx": group, "prediction_plane": {"path": str((bundle_root / plane["path"]).resolve()), "sha256": sha256(bundle_root / plane["path"])}, "mask_plane": {"path": str(support_path.resolve()), "sha256": sha256(support_path)}, "metric_input": str(target.relative_to(staging))})
        sensitivity = run_mask_sensitivity_audit(staging / "metric_inputs", staging / "mask_sensitivity")
        write_immutable_json(staging / "manifest.json", {"schema": "direct_native_bundle_metrics/v1", "bundle_manifest": {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)}, "r0_native_maps_root": str(Path(r0_native_maps_root).resolve()), "sparse_manifest": {"path": str(Path(sparse_manifest).resolve()), "sha256": sha256(Path(sparse_manifest))}, "mask_root": str(mask_root.resolve()), "metric_domains": list(MASKS), "display_mask_separate_from_metric_mask": True, "planes": records, "mask_sensitivity_output": str(sensitivity.relative_to(staging)), "no_training": True, "no_reconstruction": True, "no_formal_export_modified": True})
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True); parser.add_argument("--r0-native-maps-root", required=True)
    parser.add_argument("--sparse-manifest", required=True); parser.add_argument("--mask-root", required=True); parser.add_argument("--output", required=True)
    print(build_direct_native_metrics(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
