"""Contracts for the read-only SCMR/Phase4R support sensitivity audit."""
from __future__ import annotations

import json

import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.phase2_clean import mask_sensitivity_audit as audit
from paper_experiments.joint_vs_2dfit.phase2_clean import scmr_central_component_mask as scmr


def test_strict_common_and_component_selection_match_legacy_algorithm() -> None:
    original = np.ones((5, 5)); registered = np.ones((5, 5)); synthetic = np.ones((5, 5))
    strict = scmr.build_legacy_strict_common(original, registered, synthetic, np.ones((5, 5), bool), np.ones((5, 5), bool), np.ones((5, 5), bool))
    strict[:] = False
    strict[1, 1] = strict[2, 2] = strict[3, 3] = True
    strict[0, 4] = True
    selected, record = scmr.select_scmr_central_component(strict)
    assert selected.sum() == 3
    assert record["selection"] == "contains_image_center" and record["connectivity"] == 8


def test_phase4r_control_isolates_component_rule_from_six_input_construction() -> None:
    payload = {name: np.ones((5, 5), bool) for name in ("training_mask", "synthetic_support", "fitting_valid_mask", "phase4r_l100")}
    payload.update({name: np.ones((5, 5), float) for name in ("original", "registered", "synthetic")})
    payload["phase4r_l100"] = np.eye(5, dtype=bool)
    masks, _ = scmr.build_sensitivity_masks(payload)
    assert masks["scmr_style_central_component"].all()
    assert np.array_equal(masks["phase4r_l100"], masks["phase4r_l100_central_component"])


def test_component_rule_falls_back_to_largest_without_morphology() -> None:
    support = np.zeros((7, 7), bool)
    support[3, 3] = True  # centre component is <25% of the largest component.
    support[0, :3] = True; support[1, :3] = True
    selected, record = scmr.select_scmr_central_component(support)
    assert record["selection"] == "largest_component"
    assert selected.sum() == 6 and not selected[3, 3]


def test_strict_common_requires_each_finite_input_and_each_explicit_mask() -> None:
    values = np.ones((2, 2)); registered = values.copy(); registered[0, 1] = np.nan
    mask = np.ones((2, 2), bool); synthetic_support = mask.copy(); synthetic_support[1, 0] = False
    support = scmr.build_legacy_strict_common(values, registered, values, mask, synthetic_support, mask)
    assert support.tolist() == [[True, False], [False, True]]


def test_same_prediction_has_different_metrics_when_masks_change() -> None:
    reference = np.array([[0.0, 0.0, 0.0, 0.0]]); prediction = np.array([[0.0, 0.0, 10.0, 10.0]])
    result = audit.compare_mask_sensitivity(reference, prediction, np.array([[True, True, False, False]]), np.array([[False, False, True, True]]), np.array([[False, False, True, True]]))
    assert result["scmr_style_central_component"]["metrics"]["rmse"] == pytest.approx(0.0)
    assert result["phase4r_l100"]["metrics"]["rmse"] == pytest.approx(10.0)
    assert result["scmr_style_central_component_vs_phase4r_l100_intersection"]["metrics"] is None


def test_rejects_misaligned_support_arrays() -> None:
    with pytest.raises(ValueError, match="share one shape"):
        audit.compare_mask_sensitivity(np.zeros((2, 2)), np.zeros((2, 2)), np.ones((2, 2), bool), np.ones((2, 3), bool), np.ones((2, 2), bool))


def test_generator_outputs_sensitivity_ready_bundles_and_audit_is_immutable(tmp_path) -> None:
    inputs = tmp_path / "inputs"; target = inputs / "planes" / "t1" / "2ch" / "group000.npz"; target.parent.mkdir(parents=True)
    np.savez_compressed(target, reference=np.array([[0.0, 0.0]]), prediction=np.array([[0.0, 10.0]]), original=np.array([[1.0, 1.0]]), registered=np.array([[1.0, 1.0]]), synthetic=np.array([[1.0, 1.0]]), training_mask=np.array([[True, False]]), synthetic_support=np.array([[True, True]]), fitting_valid_mask=np.array([[True, True]]), phase4r_l100=np.array([[False, True]]), source_support=np.array([[True, True]]))
    masks = scmr.run_scmr_central_component_mask(inputs, tmp_path / "masks")
    output = audit.run_mask_sensitivity_audit(masks, tmp_path / "audit")
    manifest = json.loads((output / "manifest.json").read_text())
    aggregate = json.loads((output / "aggregate_metrics.json").read_text())
    assert manifest["no_reconstruction"] is True
    assert aggregate["domains"]["scmr_style_central_component"]["metrics"]["rmse"] == pytest.approx(0.0)
    assert aggregate["domains"]["phase4r_l100"]["metrics"]["rmse"] == pytest.approx(10.0)
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        audit.run_mask_sensitivity_audit(masks, output)
