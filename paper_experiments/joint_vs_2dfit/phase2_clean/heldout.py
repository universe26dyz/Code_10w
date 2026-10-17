"""Read-only R1/L100 held-out native-plane evaluation from the immutable R0 cache."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from .contracts import sha256, write_immutable_json
from .evaluation import heldout_quantitative_metrics, reproject_volume_to_native_plane
from .fit_first import select_verified_native_cache_groups


def _heldout_ids(manifest: dict[str, object], stack: str) -> list[int]:
    spec = manifest["stacks"][stack]
    if manifest.get("condition") != "L100":
        return [int(value) for value in spec["heldout_original_group_ids"]]
    source_path = Path(spec["source_observation"]["path"])
    with np.load(source_path, allow_pickle=False) as source:
        return [int(value) for value in np.unique(source["group_idx"])]


def evaluate_heldout(
    sparse_manifest: str | Path, proposed_t1: str | Path, proposed_t2: str | Path,
    fit_first_t1: str | Path, fit_first_t2: str | Path, output: str | Path,
    r0_native_maps_root: str | Path,
) -> Path:
    """Compare reconstruction volumes on held-out cached native maps; never refit."""
    import nibabel as nib

    manifest_path, output, cache_root = Path(sparse_manifest), Path(output), Path(r0_native_maps_root)
    if output.exists():
        raise FileExistsError(f"refusing overwrite: {output}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "ELIGIBLE":
        raise ValueError("held-out evaluation requires an ELIGIBLE sparse manifest")
    proposed_images = {"t1_ms": nib.load(str(proposed_t1)), "t2_ms": nib.load(str(proposed_t2))}
    fit_images = {"t1_ms": nib.load(str(fit_first_t1)), "t2_ms": nib.load(str(fit_first_t2))}
    l100 = manifest.get("condition") == "L100"
    selected = {stack: select_verified_native_cache_groups(cache_root, manifest_path, stack, _heldout_ids(manifest, stack)) for stack in ("2ch", "4ch")}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        result: dict[str, object] = {
            "subject": manifest["subject"], "condition": manifest["condition"], "phase": manifest["phase"],
            "endpoint": "full-data LAX agreement reference point; not held-out generalization" if l100 else "held-out native-plane quantitative agreement/generalization, not true 3-D accuracy",
            "l100_reconstruction_source": "R0_full" if l100 else None,
            "r1_signal_domain_evaluation": "DEFERRED_BY_DESIGN", "stacks": {},
        }
        for stack in ("2ch", "4ch"):
            spec, cache = manifest["stacks"][stack], selected[stack]
            records = []
            for index, group in enumerate(cache["group_idx"]):
                affine_ras = np.diag([-1.0, -1.0, 1.0, 1.0]) @ np.asarray(cache["affine_lps_rc"][index], float)
                plane = {"original_group_idx": int(group), "parameters": {}}
                for parameter in ("t1_ms", "t2_ms"):
                    proposed, p_valid = reproject_volume_to_native_plane(proposed_images[parameter].get_fdata(), proposed_images[parameter].affine, affine_ras, cache[parameter][index].shape)
                    baseline, b_valid = reproject_volume_to_native_plane(fit_images[parameter].get_fdata(), fit_images[parameter].affine, affine_ras, cache[parameter][index].shape)
                    plane["parameters"][parameter] = heldout_quantitative_metrics(cache[parameter][index], proposed, baseline, cache["source_support"][index], p_valid & b_valid)
                records.append(plane)
            result["stacks"][stack] = {
                "heldout_original_group_ids": _heldout_ids(manifest, stack),
                "source_observation_sha256": spec["source_observation"]["sha256"], "planes": records,
                "native_reference": cache["provenance"],
            }
        result["provenance"] = {
            "sparse_manifest": {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)},
            "native_reference_source": "R0_native_cache", "native_reference_cache_root": str(cache_root.resolve()),
            "evaluation_only_native_reference": True, "heldout_never_visible_to_reconstruction": True,
            "native_fit_recomputed": False,
        }
        write_immutable_json(staging / "heldout_quantitative_metrics.json", result)
        if output.exists():
            raise FileExistsError(f"refusing overwrite: {output}")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output / "heldout_quantitative_metrics.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sparse-manifest", "proposed-t1", "proposed-t2", "fit-first-t1", "fit-first-t2", "r0-native-maps-root", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(evaluate_heldout(**vars(args)))


if __name__ == "__main__":
    main()
