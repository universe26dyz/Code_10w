"""Fail-closed formal downstream admission for both Fit-first routes."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_experiments.joint_vs_2dfit.phase2_clean.fit_first_route_contract import resolve_fit_first_route


def _result(input_slices: Path, *, frozen: bool = True, scaled: bool = True) -> dict:
    return {"input_stacks": None, "input_slices": str(input_slices.resolve()), "no_transformation_optimization": frozen, "no_slice_scale": scaled}


def _new_route(root: Path, *, gate: str = "REGISTERED_QUANTITATIVE_INPUTS_VALID", audit: str = "POSE_STAGE1_AND_QUANTITATIVE_STAGE2_VALID", frozen: bool = True, scaled: bool = True) -> None:
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"schema": "phase4r_intensity_preserving_fit_first/v1", "no_dictionary_refit": True, "no_pose_copying": True}))
    (root / "registered_quantitative_inputs_gate.json").write_text(json.dumps({"status": gate}))
    (root / "final_pose_audit.json").write_text(json.dumps({"schema": "fit_first_intensity_preserving_final_pose_audit/v1", "status": audit}))
    for parameter in ("t1", "t2"):
        target = root / "stage2_reconstruction" / parameter; target.mkdir(parents=True)
        (target / "model.pt").write_bytes(parameter.encode())
        (target / "result.json").write_text(json.dumps(_result(root / f"registered_quantitative_inputs/{parameter}/slices", frozen=frozen, scaled=scaled)))
        (target / "slices").mkdir()


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
