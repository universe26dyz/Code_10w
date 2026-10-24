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


def _endpoint(root: Path, method: str, endpoint: str, *, subject: str = "CYJ", route_schema: str = "phase4r_exp1_1_fit_first/v1") -> Path:
    root.mkdir(parents=True)
    for parameter, value in (("t1", 900.0), ("t2", 70.0)):
        target = root / (f"{parameter.upper()}_3D.nii.gz" if method == "Proposed" else f"{parameter}/volume_ms.nii.gz")
        _volume(target, value)
        if method == "Fit-first":
            mask = root / parameter / "query_effective_mask.nii.gz"
            _volume(mask, 1.0)
    parameters = {} if method == "Proposed" else {parameter: {"effective_query_mask": {"path": str((root / parameter / "query_effective_mask.nii.gz").resolve()), "sha256": sha256(root / parameter / "query_effective_mask.nii.gz")}} for parameter in ("t1", "t2")}
    payload = {"schema": "phase4r_formal_proposed_export/v1" if method == "Proposed" else "phase4r_fixed_l100_fit_first_export_bundle/v2", "subject": subject, "condition": "L50_fold0", "method": method, "endpoint": endpoint, "parameters": parameters}
    if method == "Fit-first":
        payload["fit_first_routes"] = {name: {"route_schema": route_schema} for name in ("l100", "sparse")}
    (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
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
    manifest.write_text(json.dumps({"subject": "CYJ", "condition": "L50", "phase": 0, "status": "ELIGIBLE", "stacks": {stack: {"heldout_original_group_ids": [3]} for stack in ("2ch", "4ch")}}), encoding="utf-8")
    cache = {"group_idx": np.array([3, 9]), "source_support": np.ones((2, 2, 2), bool), "affine_lps_rc": np.repeat(np.eye(4)[None], 2, axis=0), "t1_ms": np.full((2, 2, 2), 1000.0), "t2_ms": np.full((2, 2, 2), 80.0)}
    calls = []
    monkeypatch.setattr(sparse, "select_verified_native_cache_groups", lambda root, source, stack, groups: calls.append((stack, groups)) or {key: (value[[0]] if isinstance(value, np.ndarray) and value.shape[0] == 2 else value) for key, value in cache.items()})
    output = sparse.build_final_sparse_native_metrics(proposed_endpoint_root=proposed, fit_first_endpoint_root=fit, native_mask_root=_mask_root(tmp_path / "masks"), r0_native_maps_root=tmp_path / "r0", sparse_manifest=manifest, output=tmp_path / "out")
    payload = json.loads((output / "aggregate_metrics.json").read_text(encoding="utf-8"))
    assert calls == [("2ch", [3]), ("4ch", [3])]
    assert payload["primary_mask"] == "primary_native_mask"
    assert payload["regions"]["t1"]["all_lax"]["proposed"]["support_voxels"] == 2
    assert "learned pose" not in json.dumps(payload).lower()
    published = json.loads((output / "manifest.json").read_text(encoding="utf-8"))["aggregate_metrics"]
    assert published == {"path": str((output / "aggregate_metrics.json").resolve()), "sha256": sha256(output / "aggregate_metrics.json")}
    assert ".staging-" not in published["path"]


def test_sparse_evaluator_rejects_endpoint_mismatch(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_sparse_native_metrics as sparse

    with pytest.raises(ValueError, match="endpoint mismatch"):
        sparse.validate_sparse_endpoint_roots(_endpoint(tmp_path / "proposed", "Proposed", "central"), _endpoint(tmp_path / "fit", "Fit-first", "psf128"), "L50_fold0")


def test_sparse_evaluator_rejects_old_two_stage_fit_first_route(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_sparse_native_metrics as sparse

    with pytest.raises(ValueError, match="CYJ Experiment 1.1"):
        sparse.validate_sparse_endpoint_roots(_endpoint(tmp_path / "proposed", "Proposed", "central"), _endpoint(tmp_path / "fit", "Fit-first", "central", route_schema="phase4r_intensity_preserving_fit_first/v2"), "L50_fold0")


def test_sparse_evaluator_excludes_effective_mask_background_from_both_methods(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_sparse_native_metrics as sparse

    proposed, fit = _endpoint(tmp_path / "proposed", "Proposed", "central"), _endpoint(tmp_path / "fit", "Fit-first", "central")
    manifest = tmp_path / "sparse.json"
    manifest.write_text(json.dumps({"subject": "CYJ", "condition": "L50", "phase": 0, "status": "ELIGIBLE", "stacks": {stack: {"heldout_original_group_ids": [3]} for stack in ("2ch", "4ch")}}), encoding="utf-8")
    cache = {"group_idx": np.array([3]), "source_support": np.ones((1, 2, 2), bool), "affine_lps_rc": np.eye(4)[None], "t1_ms": np.full((1, 2, 2), 1000.0), "t2_ms": np.full((1, 2, 2), 80.0)}
    monkeypatch.setattr(sparse, "select_verified_native_cache_groups", lambda *_args: cache)
    monkeypatch.setattr(sparse, "reproject_volume_to_native_plane", lambda *_args: (np.full((2, 2), 900.0), np.ones((2, 2), bool)))
    monkeypatch.setattr(sparse, "reproject_binary_mask_to_native_plane", lambda *_args: np.array([[True, False], [True, False]]))
    output = sparse.build_final_sparse_native_metrics(proposed_endpoint_root=proposed, fit_first_endpoint_root=fit, native_mask_root=_mask_root(tmp_path / "masks"), r0_native_maps_root=tmp_path / "r0", sparse_manifest=manifest, output=tmp_path / "out")
    payload = json.loads((output / "aggregate_metrics.json").read_text(encoding="utf-8"))
    assert payload["regions"]["t1"]["all_lax"]["proposed"]["support_voxels"] == 4
    assert "fit_first_query_effective_mask" in payload["primary_support_equation"]


def test_sparse_endpoint_validation_accepts_the_explicit_dyl2_subject(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_sparse_native_metrics as sparse

    proposed = _endpoint(tmp_path / "proposed", "Proposed", "central", subject="DYL_2")
    fit = _endpoint(tmp_path / "fit", "Fit-first", "central", subject="DYL_2")
    _, provenance = sparse.validate_sparse_endpoint_roots(proposed, fit, "L50_fold0", subject="DYL_2")
    assert provenance["endpoint"] == "central"


@pytest.mark.parametrize(("level", "phase", "case"), (("L50", 0, "L50_fold0"), ("L50", 1, "L50_fold1"), ("L25", 0, "L25_fold0"), ("L25", 3, "L25_fold3")))
def test_sparse_manifest_level_phase_builds_formal_case(level: str, phase: int, case: str) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_sparse_native_metrics as sparse

    assert sparse.sparse_manifest_case({"condition": level, "phase": phase}) == case


@pytest.mark.parametrize("payload", ({"condition": "L50", "phase": 2}, {"condition": "L25", "phase": 4}, {"condition": "L50"}, {"condition": "L25", "phase": "0"}))
def test_sparse_manifest_invalid_phase_fails_closed(payload: dict[str, object]) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_sparse_native_metrics as sparse

    with pytest.raises(ValueError, match="L50/L25"):
        sparse.sparse_manifest_case(payload)


def test_sparse_manifest_case_uses_explicit_custom_case_id_without_string_splitting() -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_sparse_native_metrics as sparse

    assert sparse.sparse_manifest_case({"schema_version": "phase4r_sparse_profile_v2", "case_id": "SAX100_2CH0_4CH0", "profile_id": "SAX100_2CH0_4CH0"}) == "SAX100_2CH0_4CH0"


@pytest.mark.parametrize("case_id", ("SAX100_2CH0_4CH0", "SAX100_2CH0_4CH100", "SAX100_2CH100_4CH0", "S2CH50_4CH100_fold0", "S2CH50_4CH100_fold1", "S2CH100_4CH50_fold0", "S2CH100_4CH50_fold1"))
def test_custom_v2_cases_admit_one_empty_heldout_stack(case_id: str) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_sparse_native_metrics as sparse

    heldout = {"2ch": [3], "4ch": []} if "2CH0_4CH100" in case_id or "S2CH50" in case_id else {"2ch": [], "4ch": [4]}
    result = sparse.validate_sparse_heldout_admission({"schema_version": "phase4r_sparse_profile_v2", "profile_id": case_id.split("_fold")[0], "case_id": case_id, "stacks": {stack: {"heldout_original_group_ids": ids} for stack, ids in heldout.items()}})
    assert result["total_heldout_lax"] == 1


def test_sparse_case_with_no_heldout_lax_is_rejected() -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_sparse_native_metrics as sparse

    with pytest.raises(ValueError, match="at least one held-out LAX"):
        sparse.validate_sparse_heldout_admission({"schema_version": "phase4r_sparse_profile_v2", "profile_id": "SAX100_2CH100_4CH100", "case_id": "SAX100_2CH100_4CH100", "stacks": {stack: {"heldout_original_group_ids": []} for stack in ("2ch", "4ch")}})
