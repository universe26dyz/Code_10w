"""CPU contracts for the explicit-fold three-method selection bundle."""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256


CASE = "S2CH100_4CH50_fold0"


def _endpoint(root: Path, *, method: str, endpoint: str = "central", condition: str = CASE, samples: int | None = None) -> Path:
    root.mkdir(parents=True)
    for parameter, value in (("t1", 900.0), ("t2", 70.0)):
        path = root / (f"{parameter.upper()}_3D.nii.gz" if method != "Fit-first" else f"{parameter}/volume_ms.nii.gz")
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.full((4, 4, 4), value, np.float32), np.eye(4)), str(path))
    if method == "Fit-first":
        manifest = {"schema": "phase4r_fixed_l100_fit_first_export_bundle/v1", "subject": "CYJ", "condition": condition, "endpoint": endpoint, "method": "Fit-first", "training_route": "Fit-first sparse checkpoint query"}
    else:
        route = "Q002S640_B6_roi_same_anchor_mse" if method == "Q002S640" else "Q005S640_K4_B6_roi_same_anchor_mse"
        manifest = {"schema": "phase4r_formal_proposed_export/v1", "subject": "CYJ", "condition": condition, "endpoint": endpoint, "method": "Proposed", "training_route": route, "training_psf": {"training_psf_samples": samples}}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _mask_root(root: Path, groups: dict[str, list[int]]) -> Path:
    planes = []
    for parameter in ("t1", "t2"):
        for stack, ids in groups.items():
            for group in ids:
                path = root / "planes" / parameter / stack / f"group{group:03d}.npz"; path.parent.mkdir(parents=True, exist_ok=True)
                support = np.ones((2, 2), bool)
                np.savez_compressed(path, source_support=support, phase4r_l100=support, base_native_support=support, primary_native_mask=support, reference=np.ones((2, 2)))
                planes.append({"parameter": parameter, "stack": stack, "original_group_idx": group, "shape_rc": [2, 2], "output": {"path": str(path.relative_to(root)), "sha256": sha256(path)}})
    (root / "manifest.json").write_text(json.dumps({"schema": "phase4r_current_native_central_component_mask/v1", "planes": planes, "selection_rule": "test"}), encoding="utf-8")
    return root


def _sparse(path: Path, *, fold_id: int = 0) -> Path:
    path.write_text(json.dumps({"schema_version": "phase4r_sparse_profile_v2", "status": "ELIGIBLE", "subject": "CYJ", "profile_id": "S2CH100_4CH50", "case_id": f"S2CH100_4CH50_fold{fold_id}", "fold_id": fold_id, "fold_count": 2, "stacks": {"sax": {}, "2ch": {"retained_original_group_ids": [0], "heldout_original_group_ids": [1]}, "4ch": {"retained_original_group_ids": [0], "heldout_original_group_ids": [1]}}}), encoding="utf-8")
    return path


def _cache(groups: dict[str, list[int]], stack: str) -> dict[str, np.ndarray]:
    ids = groups[stack]
    return {"group_idx": np.asarray(ids), "source_support": np.ones((len(ids), 2, 2), bool), "affine_lps_rc": np.repeat(np.eye(4)[None], len(ids), axis=0), "t1_ms": np.full((len(ids), 2, 2), 1000.0), "t2_ms": np.full((len(ids), 2, 2), 80.0)}


def test_explicit_fold_bundle_uses_one_three_method_support_and_preserves_provenance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_method_selection_metrics as selection

    groups = {"sax": [0], "2ch": [0, 1], "4ch": [0, 1]}
    sparse = _sparse(tmp_path / "sparse.json")
    monkeypatch.setattr(selection, "validate_sparse_profile_manifest", lambda path, subject: json.loads(Path(path).read_text(encoding="utf-8")))
    monkeypatch.setattr(selection, "_load_all_verified_native_cache_groups", lambda _root, _manifest, stack: _cache(groups, stack))
    def reproject(values: np.ndarray, _affine: np.ndarray, _plane: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        result = np.full(shape, float(np.nanmean(values)))
        if np.isclose(np.nanmean(values), 70.0):
            result.flat[-1] = np.nan
        return result, np.ones(shape, bool)
    monkeypatch.setattr(selection, "reproject_volume_to_native_plane", reproject)
    output = selection.build_final_method_selection_bundle(
        q002_endpoint_root=_endpoint(tmp_path / "q002", method="Q002S640", samples=8),
        q005_endpoint_root=_endpoint(tmp_path / "q005", method="Q005S640_K4", samples=4),
        fit_first_endpoint_root=_endpoint(tmp_path / "fit", method="Fit-first"),
        native_mask_root=_mask_root(tmp_path / "masks", groups), r0_native_maps_root=tmp_path / "r0", sparse_manifest=sparse, output=tmp_path / "out",
    )
    metrics = json.loads((output / "aggregate_metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert metrics["schema"] == "phase4r_method_selection_stratified_common_support_metrics/v1"
    assert metrics["regions"]["t1"]["all_source"]["voxel_count"] == 20
    assert metrics["regions"]["t1"]["lax_observed"]["voxel_count"] == 8
    assert metrics["regions"]["t1"]["lax_heldout"]["voxel_count"] == 8
    assert {(record["stack"], record["original_group_idx"], record["membership"]) for record in metrics["per_plane_records"]} == {("sax", 0, "observed"), ("2ch", 0, "observed"), ("2ch", 1, "held-out"), ("4ch", 0, "observed"), ("4ch", 1, "held-out")}
    assert all(record["parameters"]["t2"]["common_support_voxels"] == 3 for record in metrics["per_plane_records"])
    assert {name for name in ("Q002S640", "Q005S640_K4", "Fit-first") if metrics["regions"]["t1"]["all_source"][name] is not None} == {"Q002S640", "Q005S640_K4", "Fit-first"}
    assert manifest["case_id"] == CASE and manifest["fold_id"] == 0 and manifest["immutable"] is True
    assert manifest["q002_endpoint_manifest"]["sha256"] == sha256(tmp_path / "q002" / "manifest.json")
    with pytest.raises(FileExistsError):
        selection.build_final_method_selection_bundle(q002_endpoint_root=tmp_path / "q002", q005_endpoint_root=tmp_path / "q005", fit_first_endpoint_root=tmp_path / "fit", native_mask_root=tmp_path / "masks", r0_native_maps_root=tmp_path / "r0", sparse_manifest=sparse, output=output)


@pytest.mark.parametrize(("target", "value", "error"), (("q002", 4, "Q002"), ("q005", 8, "Q005"), ("fit_endpoint", "psf128", "endpoint mismatch"), ("q005_condition", "other", "condition")))
def test_bundle_rejects_route_and_identity_mismatches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, target: str, value: object, error: str) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_method_selection_metrics as selection

    groups = {"sax": [0], "2ch": [0, 1], "4ch": [0, 1]}; sparse = _sparse(tmp_path / "sparse.json")
    monkeypatch.setattr(selection, "validate_sparse_profile_manifest", lambda path, subject: json.loads(Path(path).read_text(encoding="utf-8")))
    q002_samples, q005_samples, fit_endpoint, q005_condition = 8, 4, "central", CASE
    if target == "q002": q002_samples = int(value)
    if target == "q005": q005_samples = int(value)
    if target == "fit_endpoint": fit_endpoint = str(value)
    if target == "q005_condition": q005_condition = str(value)
    with pytest.raises(ValueError, match=error):
        selection.build_final_method_selection_bundle(q002_endpoint_root=_endpoint(tmp_path / "q002", method="Q002S640", samples=q002_samples), q005_endpoint_root=_endpoint(tmp_path / "q005", method="Q005S640_K4", samples=q005_samples, condition=q005_condition), fit_first_endpoint_root=_endpoint(tmp_path / "fit", method="Fit-first", endpoint=fit_endpoint), native_mask_root=_mask_root(tmp_path / "masks", groups), r0_native_maps_root=tmp_path / "r0", sparse_manifest=sparse, output=tmp_path / "out")


def test_bundle_keeps_fold0_and_fold1_as_explicit_separate_cases(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_method_selection_metrics as selection

    sparse0, sparse1 = _sparse(tmp_path / "sparse0.json", fold_id=0), _sparse(tmp_path / "sparse1.json", fold_id=1)
    monkeypatch.setattr(selection, "validate_sparse_profile_manifest", lambda path, subject: json.loads(Path(path).read_text(encoding="utf-8")))
    assert selection.validate_method_selection_sparse_manifest(sparse0)["case_id"] != selection.validate_method_selection_sparse_manifest(sparse1)["case_id"]


def test_low_level_empty_common_support_fails_closed(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.final_method_selection_metrics import evaluate_method_selection_common_support

    with pytest.raises(ValueError, match="common support is empty"):
        evaluate_method_selection_common_support(reference=np.array([1.0]), q002=np.array([np.nan]), q005=np.array([1.0]), fit_first=np.array([1.0]), primary_native_mask=np.ones(1, bool), q002_endpoint_valid=np.ones(1, bool), q005_endpoint_valid=np.ones(1, bool), fit_first_endpoint_valid=np.ones(1, bool), endpoint="central", parameter="t1_ms", output=tmp_path / "empty.json")
