from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.xcat3d.build_quantitative_gt import build_quantitative_gt
from paper_experiments.joint_vs_2dfit.xcat3d.read_xcat_volume import load_xcat_inputs
from paper_experiments.joint_vs_2dfit.xcat3d.validate_quantitative_gt import validate_quantitative_gt


def _manifest(tmp_path: Path, *, shape: tuple[int, int, int] = (8, 9, 10), t2_axis_order: str = "xyz") -> Path:
    arrays = {
        "t1_ms": np.full(shape, 1000.0, dtype=np.float32),
        "t2_ms": np.full(shape, 50.0, dtype=np.float32),
        "signal_support_mask": np.ones(shape, dtype=bool),
        "myocardium_mask": np.zeros(shape, dtype=bool),
    }
    arrays["myocardium_mask"][2:5, 3:6, 4:7] = True
    arrays["t1_ms"][arrays["myocardium_mask"]] = 1410.0
    for name, array in arrays.items():
        np.save(tmp_path / f"{name}.npy", array)
    geometry = {"shape": list(shape), "spacing_mm": [1.0, 1.0, 1.0], "origin_mm": [-3.0, 2.0, 5.0], "axis_order": "xyz", "world_convention": "LPS"}
    volumes = {
        name: {"path": f"{name}.npy", "dtype": str(array.dtype), "sha256": hashlib.sha256((tmp_path / f"{name}.npy").read_bytes()).hexdigest(), **geometry}
        for name, array in arrays.items()
    }
    volumes["t2_ms"]["axis_order"] = t2_axis_order
    payload = {
        "phase": "1.2",
        "source_family": "single_source_xcat_3d_quantitative",
        "xcat": {"version_build": "fixture", "source": "external XCAT", "parameter_files": "parameters/", "configuration_hashes": {"shared": "fixture"}, "cardiac_phase": "0", "respiratory_phase": "0"},
        "m0_policy": "controlled_binary_signal_support_m0_1",
        "volumes": volumes,
    }
    path = tmp_path / "xcat3d_inputs.json"
    path.write_text(json.dumps(payload))
    return path


def test_full_3d_single_geometry_fixture_builds_and_validates(tmp_path: Path) -> None:
    inputs = load_xcat_inputs(_manifest(tmp_path))
    gt = build_quantitative_gt(inputs)
    qc = validate_quantitative_gt(gt)
    assert qc["scientific_gt_geometry_gate"] is True
    assert np.all(gt.t1_ms[gt.myocardium_mask] == 1410.0)
    assert np.all(gt.t2_ms[gt.myocardium_mask] == 50.0)
    assert np.all(gt.b1[gt.myocardium_mask] == 1.0)
    assert np.array_equal(gt.m0[~gt.signal_support_mask], np.zeros((~gt.signal_support_mask).sum(), dtype=np.float32))


def test_rejects_effectively_2d_xcat_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="full 3-D"):
        load_xcat_inputs(_manifest(tmp_path, shape=(920, 920, 1)))


def test_rejects_axis_order_geometry_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="geometry mismatch"):
        load_xcat_inputs(_manifest(tmp_path, t2_axis_order="zyx"))


def test_build_requires_frozen_myocardium_values(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    t1 = np.load(tmp_path / "t1_ms.npy")
    myo = np.load(tmp_path / "myocardium_mask.npy")
    t1[myo] = 1200.0
    np.save(tmp_path / "t1_ms.npy", t1)
    payload = json.loads(path.read_text()); payload["volumes"]["t1_ms"]["sha256"] = hashlib.sha256((tmp_path / "t1_ms.npy").read_bytes()).hexdigest(); path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="1410"):
        build_quantitative_gt(load_xcat_inputs(path))
