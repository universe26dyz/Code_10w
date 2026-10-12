from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.mrxcat.define_gt_tissue_maps import build_gt_maps
from paper_experiments.joint_vs_2dfit.simulation.render_observations import (
    integrate_signal_volume,
    lps_to_ras_affine,
    ras_to_lps_affine,
)
from paper_experiments.joint_vs_2dfit.simulation.simulate_hhz_signals import generate_signal_field
from paper_experiments.joint_vs_2dfit.simulation.select_timing_template import (
    choose_timing_template,
)
from paper_experiments.joint_vs_2dfit.simulation.write_observations_npz import (
    write_observations,
)


def test_gt_values_within_contract_and_primary_myocardium() -> None:
    labels = np.array([[0, 1, 5, 2]], dtype=np.int16)
    maps = build_gt_maps(labels)
    assert maps.t1_ms[0, 1] == pytest.approx(1410.0)
    assert maps.t2_ms[0, 1] == pytest.approx(50.0)
    assert maps.b1[0, 1] == pytest.approx(1.0)
    assert maps.myocardium_mask[0, 1]
    assert maps.signal_support_mask.tolist() == [[False, True, True, True]]
    assert np.all((maps.t1_ms[maps.signal_support_mask] > 0) & (maps.t1_ms[maps.signal_support_mask] <= 2500))
    assert np.all((maps.t2_ms[maps.signal_support_mask] >= 5) & (maps.t2_ms[maps.signal_support_mask] <= 200))
    assert np.all((maps.b1 >= 0.1) & (maps.b1 <= 1.2))


def test_timing_selection_is_actual_vps87_group_and_decoder_compatible() -> None:
    candidates = [
        {"subject_id": "CYJ", "stack_name": "sax", "group_idx": 2,
         "tr_ms": 2.61, "vps": 87, "timing9_ms": [1, 2, 3, 4, 5, 6, 7, 8, 9]},
        {"subject_id": "DYZ", "stack_name": "sax", "group_idx": 1,
         "tr_ms": 2.61, "vps": 87, "timing9_ms": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1]},
        {"subject_id": "DYL", "stack_name": "sax", "group_idx": 1,
         "tr_ms": 2.61, "vps": 85, "timing9_ms": [1, 2, 3, 4, 5, 6, 7, 8, 9]},
    ]
    approved = {"tr_ms": 2.61, "vps": 87,
                "train_timing9_min_ms": [0] * 9,
                "train_timing9_max_ms": [10] * 9}
    selected = choose_timing_template(candidates, approved)
    assert selected["source"]["subject_id"] == "CYJ"
    assert selected["source"]["vps"] == 87
    assert selected["decoder_domain_check"] is True


def test_signal_space_thickness_integration_differs_from_centre_plane() -> None:
    # Two samples across the acquired slice have different signals.  The
    # renderer must average signals, not average T1/T2 before simulation.
    volume = np.zeros((1, 1, 3, 1), dtype=np.float32)
    volume[0, 0, :, 0] = [0.0, 1.0, 2.0]
    affine = np.eye(4, dtype=float)
    affine[2, 3] = 1.0
    out = integrate_signal_volume(
        volume,
        volume_origin_mm=np.zeros(3),
        volume_spacing_mm=np.ones(3),
        affine_lps_rc=affine,
        rows=1,
        cols=1,
        inplane_samples=1,
        thickness_samples=3,
    )
    assert out.shape == (1, 1, 1)
    assert out[0, 0, 0] == pytest.approx(1.0)


def test_lps_ras_round_trip() -> None:
    affine = np.array([[1, 0, 0, 4], [0, -1, 0, 5], [0, 0, -1, 6], [0, 0, 0, 1]], float)
    np.testing.assert_allclose(ras_to_lps_affine(lps_to_ras_affine(affine)), affine)


def test_writer_schema_shared_group_and_overwrite_protection(tmp_path: Path) -> None:
    images = np.ones((10, 2, 2), dtype=np.float32)
    masks = np.ones((10, 2, 2), dtype=bool)
    metadata = {
        "group_idx": np.repeat([0], 10),
        "weight_idx": np.arange(10),
        "stack_idx": np.zeros(10, dtype=np.int64),
        "acquisition_time_ms": np.arange(10, dtype=float),
        "timing9_ms": np.repeat(np.arange(9, dtype=float)[None, :], 10, axis=0),
        "affine_lps_rc": np.repeat(np.eye(4)[None, ...], 10, axis=0),
        "pixel_spacing_rc_mm": np.repeat([[1.25, 1.25]], 10, axis=0),
        "slice_thickness_mm": np.repeat([8.0], 10),
        "tr_ms": np.repeat([2.61], 10),
        "vps": np.repeat([87], 10),
    }
    out = tmp_path / "observations"
    result = write_observations(out, images, masks, metadata, provenance={"source": "test"})
    assert (out / "observations.npz").exists()
    assert set(result["schema_fields"]) == {
        "images", "masks", "group_idx", "weight_idx", "stack_idx",
        "acquisition_time_ms", "timing9_ms", "affine_lps_rc",
        "pixel_spacing_rc_mm", "slice_thickness_mm", "tr_ms", "vps",
    }
    with pytest.raises(FileExistsError):
        write_observations(out, images, masks, metadata, provenance={"source": "test"})


def test_signal_generation_is_finite_and_deterministic() -> None:
    t1 = np.array([1410.0, 1000.0], dtype=np.float32)
    t2 = np.array([50.0, 50.0], dtype=np.float32)
    b1 = np.ones(2, dtype=np.float32)
    timing = np.array([642.93, 568.93, 558.93, 406.93, 590.93, 565.93, 580.93, 606.93, 591.93], dtype=np.float32)
    first = generate_signal_field(t1, t2, b1, timing)
    second = generate_signal_field(t1, t2, b1, timing)
    assert first.shape == (2, 10)
    assert np.all(np.isfinite(first))
    np.testing.assert_array_equal(first, second)


def test_generated_clean_dataset_contract_if_present() -> None:
    root = Path(__file__).resolve().parents[1] / "simulation/outputs/phase1_clean_mrxcat"
    if not root.exists():
        pytest.skip("formal Phase 1 output is generated after the source-only test stage")
    data = np.load(root / "sax/observations.npz")
    assert data["images"].shape[0] % 10 == 0
    assert np.all(data["vps"] == 87)
    assert np.allclose(data["tr_ms"], 2.61)
    assert np.all(np.isfinite(data["images"]))
    source = json.loads((root / "synthetic_source_manifest.json").read_text())
    assert source["gt_contract"]["m0"] == 1.0
    assert source["forbidden_routes"]["frozen_mlp"] is False
    assert source["forbidden_routes"]["reconstruction"] is False
