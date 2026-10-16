"""Method-neutral held-out 2CH/4CH quantitative-map evaluation for Phase-4R."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluation import heldout_quantitative_metrics, reproject_volume_to_native_plane
from .fit_first import fit_native
from .contracts import sha256, write_immutable_json


def _subset_observations(source: Path, ids: list[int], output: Path) -> Path:
    with np.load(source, allow_pickle=False) as data:
        selected = np.isin(data["group_idx"], ids)
        if not selected.any(): raise ValueError(f"{source}: no requested held-out groups")
        output.parent.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(output, **{key: np.asarray(data[key])[selected] if np.asarray(data[key]).shape[:1] == selected.shape else np.asarray(data[key]) for key in data.files})
    return output


def evaluate_heldout(
    sparse_manifest: str | Path, proposed_t1: str | Path, proposed_t2: str | Path,
    fit_first_t1: str | Path, fit_first_t2: str | Path, output: str | Path,
) -> Path:
    """Fit held-out native maps after reconstruction, then compare on common support."""
    import nibabel as nib

    manifest_path, output = Path(sparse_manifest), Path(output)
    if output.exists(): raise FileExistsError(f"refusing overwrite: {output}")
    manifest = json.loads(manifest_path.read_text())
    output.mkdir(parents=True)
    proposed_images = {"t1_ms": nib.load(str(proposed_t1)), "t2_ms": nib.load(str(proposed_t2))}
    fit_images = {"t1_ms": nib.load(str(fit_first_t1)), "t2_ms": nib.load(str(fit_first_t2))}
    result: dict[str, object] = {"subject": manifest["subject"], "condition": manifest["condition"], "phase": manifest["phase"], "endpoint": "held-out native-plane quantitative agreement/generalization, not true 3-D accuracy", "r1_signal_domain_evaluation": "DEFERRED_BY_DESIGN", "stacks": {}}
    for stack in ("2ch", "4ch"):
        spec = manifest["stacks"][stack]
        if manifest.get("condition") == "L100":
            with np.load(spec["source_observation"]["path"], allow_pickle=False) as source:
                heldout = [int(value) for value in np.unique(source["group_idx"])]
        else:
            heldout = [int(value) for value in spec["heldout_original_group_ids"]]
        selected = _subset_observations(Path(spec["source_observation"]["path"]), heldout, output / "evaluation_only" / stack / "observations.npz")
        maps_root = fit_native(selected, output / "evaluation_only" / stack / "dictionary_fit")
        maps, geometry = np.load(maps_root / "native_maps.npz"), np.load(maps_root / "geometry.npz")
        records = []
        for index, group in enumerate(maps["group_idx"]):
            original_group = int(group)
            rows = np.flatnonzero(np.load(selected, allow_pickle=False)["group_idx"] == original_group)
            with np.load(selected, allow_pickle=False) as source:
                source_support = np.all(source["masks"][rows].astype(bool), axis=0)
            affine_ras = np.diag([-1.0, -1.0, 1.0, 1.0]) @ np.asarray(geometry["affine_lps_rc"][index], float)
            plane = {"original_group_idx": original_group, "parameters": {}}
            for parameter in ("t1_ms", "t2_ms"):
                proposed, p_valid = reproject_volume_to_native_plane(proposed_images[parameter].get_fdata(), proposed_images[parameter].affine, affine_ras, maps[parameter][index].shape)
                baseline, b_valid = reproject_volume_to_native_plane(fit_images[parameter].get_fdata(), fit_images[parameter].affine, affine_ras, maps[parameter][index].shape)
                plane["parameters"][parameter] = heldout_quantitative_metrics(maps[parameter][index], proposed, baseline, source_support, p_valid & b_valid)
            records.append(plane)
        result["stacks"][stack] = {"heldout_original_group_ids": heldout, "source_observation_sha256": spec["source_observation"]["sha256"], "planes": records}
    result["provenance"] = {"sparse_manifest": {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)}, "evaluation_only_native_maps": True, "heldout_never_visible_to_reconstruction": True}
    write_immutable_json(output / "heldout_quantitative_metrics.json", result)
    return output / "heldout_quantitative_metrics.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sparse-manifest", "proposed-t1", "proposed-t2", "fit-first-t1", "fit-first-t2", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(); print(evaluate_heldout(**vars(args)))


if __name__ == "__main__":
    main()
