"""Fail-closed formal downstream admission for both Fit-first routes."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_experiments.joint_vs_2dfit.phase2_clean.fit_first_route_contract import resolve_fit_first_route
from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256


def _result(input_slices: Path, *, frozen: bool = True, scaled: bool = True) -> dict:
    return {"input_stacks": None, "input_slices": str(input_slices.resolve()), "no_transformation_optimization": frozen, "no_slice_scale": scaled}


def _new_route(root: Path, *, gate: str = "REGISTERED_QUANTITATIVE_INPUTS_VALID", audit: str = "POSE_STAGE1_AND_QUANTITATIVE_STAGE2_VALID", frozen: bool = True, scaled: bool = True) -> None:
    root.mkdir()
    active = ["sax"]
    (root / "registered_quantitative_inputs_gate.json").write_text(json.dumps({"schema": "phase4r_fit_first_registered_quantitative_inputs/v2", "status": gate}))
    masks = {}
    for parameter in ("t1", "t2"):
        target = root / "stage2_reconstruction" / parameter; target.mkdir(parents=True)
        (target / "model.pt").write_bytes(parameter.encode())
        (target / "result.json").write_text(json.dumps(_result(root / f"registered_quantitative_inputs/{parameter}/slices", frozen=frozen, scaled=scaled)))
        (target / "slices").mkdir()
        stack = root / f"nesvor_inputs/sax/{parameter}/stack.nii.gz"; stack.parent.mkdir(parents=True); stack.write_bytes(b"stack")
        mask = root / f"nesvor_inputs/sax/{parameter}/stack_mask.nii.gz"; mask.write_bytes(b"mask")
        stage1 = root / f"stage1_registration/{parameter}"; stage1.mkdir(parents=True)
        (stage1 / "result.json").write_text(json.dumps({"registration": "stack", "input_stacks": [str(stack.resolve())], "stack_masks": [str(mask.resolve())]}))
        masks[parameter] = {"sax": {"path": str(mask.resolve()), "sha256": sha256(mask)}}
    stage1_manifest = root / "stage1_registration_manifest.json"
    stage1_manifest.write_text(json.dumps({"schema": "phase4r_fit_first_registration_stage/v2", "active_stacks": active, "stack_masks": masks}))
    audit_payload = {"schema": "fit_first_intensity_preserving_final_pose_audit/v2", "status": audit, "parameters": {parameter: {"stage1_stack_masks": [masks[parameter]["sax"]]} for parameter in ("t1", "t2")}}
    (root / "final_pose_audit.json").write_text(json.dumps(audit_payload))
    (root / "manifest.json").write_text(json.dumps({"schema": "phase4r_intensity_preserving_fit_first/v2", "no_dictionary_refit": True, "no_pose_copying": True, "active_stacks": active, "stage1": {"path": str(stage1_manifest.resolve()), "sha256": sha256(stage1_manifest)}}))


def test_resolves_legacy_models_without_new_route_admission(tmp_path: Path) -> None:
    root = tmp_path / "legacy"; root.mkdir()
    (root / "input_manifest.json").write_text(json.dumps({"method": "fit_first"}))
    for parameter in ("t1", "t2"):
        target = root / "nesvor" / parameter; target.mkdir(parents=True)
        (target / "model.pt").write_bytes(b"legacy")
        (target / "result.json").write_text("{}")
        (target / "slices").mkdir()
    route = resolve_fit_first_route(root)
    assert route.route_kind == "legacy_fit_first"
    assert route.models["t1"] == root / "nesvor/t1/model.pt"


def test_resolves_intensity_preserving_stage2_models(tmp_path: Path) -> None:
    root = tmp_path / "new"; _new_route(root)
    route = resolve_fit_first_route(root)
    assert route.route_kind == "intensity_preserving_fit_first"
    assert route.models["t1"] == root / "stage2_reconstruction/t1/model.pt"
    assert "stage1_registration" not in str(route.models["t1"])


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"gate": "BAD"}, "registered quantitative"),
    ({"audit": "BAD"}, "final pose audit"),
    ({"frozen": False}, "no_transformation_optimization"),
    ({"scaled": False}, "no_slice_scale"),
])
def test_new_route_rejects_missing_or_bad_formal_admission(tmp_path: Path, kwargs: dict, message: str) -> None:
    root = tmp_path / "new"; _new_route(root, **kwargs)
    with pytest.raises(ValueError, match=message):
        resolve_fit_first_route(root)


def test_new_route_rejects_missing_gate_and_unknown_schema(tmp_path: Path) -> None:
    root = tmp_path / "missing"; _new_route(root)
    (root / "registered_quantitative_inputs_gate.json").unlink()
    with pytest.raises(FileNotFoundError):
        resolve_fit_first_route(root)
    unknown = tmp_path / "unknown"; unknown.mkdir(); (unknown / "manifest.json").write_text(json.dumps({"schema": "unknown/v1"}))
    with pytest.raises(ValueError, match="unknown"):
        resolve_fit_first_route(unknown)


def test_new_route_rejects_maskless_v1_route_schema(tmp_path: Path) -> None:
    root = tmp_path / "old"; _new_route(root)
    manifest = json.loads((root / "manifest.json").read_text()); manifest["schema"] = "phase4r_intensity_preserving_fit_first/v1"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unknown"):
        resolve_fit_first_route(root)


def test_new_route_rejects_missing_stage1_stack_masks(tmp_path: Path) -> None:
    root = tmp_path / "new"; _new_route(root)
    result = root / "stage1_registration/t1/result.json"
    payload = json.loads(result.read_text()); payload.pop("stack_masks")
    result.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="stack mask"):
        resolve_fit_first_route(root)
