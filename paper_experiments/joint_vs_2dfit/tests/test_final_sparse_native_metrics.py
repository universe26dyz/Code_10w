"""CPU contracts for held-out sparse formal native endpoint metrics."""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256


def _volume(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.full((4, 4, 1), value, np.float32), np.eye(4)), str(path))


def _endpoint(root: Path, method: str, endpoint: str) -> Path:
    root.mkdir(parents=True)
    for parameter, value in (("t1", 900.0), ("t2", 70.0)):
        target = root / (f"{parameter.upper()}_3D.nii.gz" if method == "Proposed" else f"{parameter}/volume_ms.nii.gz")
        _volume(target, value)
    (root / "manifest.json").write_text(json.dumps({"schema": "phase4r_formal_proposed_export/v1" if method == "Proposed" else "phase4r_fixed_l100_fit_first_export_bundle/v1", "subject": "CYJ", "condition": "L50_fold0", "method": method, "endpoint": endpoint}), encoding="utf-8")
    return root


def _mask_root(root: Path) -> Path:
    planes = []
    for parameter in ("t1", "t2"):
        for stack in ("2ch", "4ch"):
            target = root / "planes" / parameter / stack / "group003.npz"; target.parent.mkdir(parents=True, exist_ok=True)
            support = np.ones((2, 2), bool)
            np.savez_compressed(target, source_support=support, phase4r_l100=support, base_native_support=support, primary_native_mask=support, reference=np.ones((2, 2)))
            planes.append({"parameter": parameter, "stack": stack, "original_group_idx": 3, "output": {"path": str(target.relative_to(root)), "sha256": sha256(target)}})
    (root / "manifest.json").write_text(json.dumps({"schema": "phase4r_current_native_central_component_mask/v1", "planes": planes, "selection_rule": "test"}), encoding="utf-8")
    return root


def test_sparse_evaluator_uses_only_heldout_original_geometry_and_current_native_mask(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_sparse_native_metrics as sparse

    proposed, fit = _endpoint(tmp_path / "proposed", "Proposed", "central"), _endpoint(tmp_path / "fit", "Fit-first", "central")
    manifest = tmp_path / "sparse.json"
    manifest.write_text(json.dumps({"subject": "CYJ", "condition": "L50_fold0", "status": "ELIGIBLE", "stacks": {stack: {"heldout_original_group_ids": [3]} for stack in ("2ch", "4ch")}}), encoding="utf-8")
    cache = {"group_idx": np.array([3, 9]), "source_support": np.ones((2, 2, 2), bool), "affine_lps_rc": np.repeat(np.eye(4)[None], 2, axis=0), "t1_ms": np.full((2, 2, 2), 1000.0), "t2_ms": np.full((2, 2, 2), 80.0)}
    calls = []
    monkeypatch.setattr(sparse, "select_verified_native_cache_groups", lambda root, source, stack, groups: calls.append((stack, groups)) or {key: (value[[0]] if isinstance(value, np.ndarray) and value.shape[0] == 2 else value) for key, value in cache.items()})
    output = sparse.build_final_sparse_native_metrics(proposed_endpoint_root=proposed, fit_first_endpoint_root=fit, native_mask_root=_mask_root(tmp_path / "masks"), r0_native_maps_root=tmp_path / "r0", sparse_manifest=manifest, output=tmp_path / "out")
    payload = json.loads((output / "aggregate_metrics.json").read_text(encoding="utf-8"))
    assert calls == [("2ch", [3]), ("4ch", [3])]
    assert payload["primary_mask"] == "primary_native_mask"
    assert payload["regions"]["t1"]["all_lax"]["proposed"]["support_voxels"] == 2
    assert "learned pose" not in json.dumps(payload).lower()


def test_sparse_evaluator_rejects_endpoint_mismatch(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_sparse_native_metrics as sparse

    with pytest.raises(ValueError, match="endpoint mismatch"):
        sparse.validate_sparse_endpoint_roots(_endpoint(tmp_path / "proposed", "Proposed", "central"), _endpoint(tmp_path / "fit", "Fit-first", "psf128"), "L50_fold0")
