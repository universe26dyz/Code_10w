"""CPU contract for frozen Fit-first final-NeSVoR native-pose provenance."""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from paper_experiments.joint_vs_2dfit.phase2_clean import fit_first_native_pose_audit as audit


def _volume(path: Path, affine: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.ones((2, 3, 1), dtype=np.float32), affine), path)


def test_audit_maps_each_original_group_to_final_output_slice_and_rejects_r0_fallback(tmp_path: Path) -> None:
    baseline, r0 = tmp_path / "baseline", tmp_path / "r0"
    group_ids = {"sax": [5], "2ch": [2], "4ch": [8]}
    for stack, groups in group_ids.items():
        native = baseline / "native_maps" / stack; native.mkdir(parents=True)
        np.savez_compressed(native / "native_maps.npz", group_idx=np.asarray(groups, dtype=np.int64), t1_ms=np.ones((len(groups), 2, 3), dtype=np.float32))
        inputs = baseline / "nesvor_inputs" / stack; inputs.mkdir(parents=True)
        (inputs / "manifest.json").write_text(json.dumps({"group_idx": groups}), encoding="utf-8")
        cache = r0 / stack; cache.mkdir(parents=True)
        np.savez_compressed(cache / "geometry.npz", affine_lps_rc=np.eye(4)[None])
        np.savez_compressed(cache / "native_maps.npz", group_idx=np.asarray(groups, dtype=np.int64), t1_ms=np.ones((len(groups), 2, 3), dtype=np.float32))
    for parameter in ("t1", "t2"):
        root = baseline / "nesvor" / parameter; (root / "slices").mkdir(parents=True)
        (root / "result.json").write_text(json.dumps({"registration": "stack", "no_transformation_optimization": False}), encoding="utf-8")
        (root / "model.pt").write_bytes(b"checkpoint")
        for index in range(3):
            affine = np.eye(4); affine[0, 3] = 10.0 + index
            _volume(root / "slices" / f"{index}.nii.gz", affine)

    output = audit.audit_fit_first_native_poses(baseline_root=baseline, r0_native_maps_root=r0, output=tmp_path / "audit")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["parameters"]["t1"]["groups"]
    assert [(row["stack"], row["original_group_idx"], row["output_slice_index"]) for row in records] == [("sax", 5, 0), ("2ch", 2, 1), ("4ch", 8, 2)]
    assert all(row["authoritative_geometry"] == "nesvor_output_slices_final_optimized_affine" for row in records)
    assert all(row["r0_affine_mathematically_equivalent"] is False for row in records)
    assert manifest["fit_first_endpoint_policy"] == "must_use_final_nesvor_output_slices_unless_equivalence_is_proven"
    assert not (output / "metrics.json").exists()
