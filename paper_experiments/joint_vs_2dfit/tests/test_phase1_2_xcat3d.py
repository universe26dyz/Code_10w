from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.xcat3d.build_quantitative_gt import build_quantitative_gt
from paper_experiments.joint_vs_2dfit.xcat3d.read_xcat_volume import load_xcat_inputs
from paper_experiments.joint_vs_2dfit.xcat3d.validate_quantitative_gt import validate_quantitative_gt


def _manifest(tmp_path: Path, *, shape: tuple[int, int, int] = (8, 9, 10), spacing: tuple[float, float, float] = (1.0, 1.0, 1.0), t2_axis_order: str = "xyz") -> Path:
    arrays = {
        "t1_ms": np.full(shape, 1000.0, dtype=np.float32),
        "t2_ms": np.full(shape, 50.0, dtype=np.float32),
        "pd_m0": np.ones(shape, dtype=np.float32),
        "myocardium_mask": np.zeros(shape, dtype=bool),
    }
    arrays["myocardium_mask"][2:5, 3:6, 4:7] = True
    arrays["t1_ms"][arrays["myocardium_mask"]] = 1410.0
    for name, array in arrays.items():
        np.save(tmp_path / f"{name}.npy", array)
    geometry = {"shape": list(shape), "spacing_mm": list(spacing), "origin_mm": [-3.0, 2.0, 5.0], "axis_order": "xyz", "world_convention": "LPS"}
    parameters = {}
    for filename in ("t1_phase1_2.par", "t2_phase1_2.par", "pd_phase1_2.par", "lv_mask_phase1_2.par"):
        path = tmp_path / filename
        path.write_text(filename)
        parameters[filename] = {"path": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    phantom, geometry_spec = tmp_path / "cardiac_roi_phantom_spec.json", tmp_path / "cardiac_roi_geometry.json"
    phantom.write_text(json.dumps({"name": "fixture"})); geometry_spec.write_text(json.dumps({"minimum_lv_boundary_margin_mm": 1.0}))
    parameter_for = {"t1_ms": "t1_phase1_2.par", "t2_ms": "t2_phase1_2.par", "pd_m0": "pd_phase1_2.par", "myocardium_mask": "lv_mask_phase1_2.par"}
    volumes = {
        name: {"path": f"{name}.npy", "dtype": str(array.dtype), "sha256": hashlib.sha256((tmp_path / f"{name}.npy").read_bytes()).hexdigest(), "parameter_file_sha256": parameters[parameter_for[name]]["sha256"], "shared_geometry_spec_sha256": hashlib.sha256(geometry_spec.read_bytes()).hexdigest(), "cardiac_phase": "0", "respiratory_phase": "0", **geometry}
        for name, array in arrays.items()
    }
    volumes["t2_ms"]["axis_order"] = t2_axis_order
    payload = {
        "phase": "1.2",
        "source_family": "single_source_xcat_3d_quantitative",
        "xcat": {"version_build": "fixture", "source": "external XCAT", "parameter_files": parameters, "phantom_spec": {"path": phantom.name, "sha256": hashlib.sha256(phantom.read_bytes()).hexdigest()}, "geometry_spec": {"path": geometry_spec.name, "sha256": hashlib.sha256(geometry_spec.read_bytes()).hexdigest()}, "cardiac_phase": "0", "respiratory_phase": "0"},
        "support_policy": "native_xcat_pd_positive",
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


def test_rejects_tampered_parameter_file_against_declared_sha256(tmp_path: Path) -> None:
    """A real raw volume cannot be accepted if its generating parameter file drifted."""
    path = _manifest(tmp_path)
    (tmp_path / "t1_phase1_2.par").write_text("tampered")
    with pytest.raises(ValueError, match="parameter-file .*SHA256 mismatch"):
        load_xcat_inputs(path)


def test_rejects_placeholder_in_frozen_geometry_spec(tmp_path: Path) -> None:
    """A hashed but unfinished geometry file is not a formal XCAT input."""
    path = _manifest(tmp_path)
    geometry = tmp_path / "cardiac_roi_geometry.json"
    geometry.write_text(json.dumps({"minimum_lv_boundary_margin_mm": 1.0, "array_size": "REQUIRED"}))
    payload = json.loads(path.read_text())
    digest = hashlib.sha256(geometry.read_bytes()).hexdigest()
    payload["xcat"]["geometry_spec"]["sha256"] = digest
    for record in payload["volumes"].values():
        record["shared_geometry_spec_sha256"] = digest
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="placeholder"):
        load_xcat_inputs(path)


def test_tissue_property_canonicalization_does_not_create_boundary_values(tmp_path: Path) -> None:
    """Linear T1/T2 resampling would silently invent non-tissue relaxation values."""
    gt = build_quantitative_gt(load_xcat_inputs(_manifest(tmp_path, spacing=(2.0, 1.0, 1.0))))
    assert set(np.unique(gt.t1_ms)) <= {np.float32(1000.0), np.float32(1410.0)}
    assert set(np.unique(gt.t2_ms)) == {np.float32(50.0)}


def test_rejects_lv_mask_that_touches_the_cardiac_roi_boundary(tmp_path: Path) -> None:
    """A cropped LV must fail rather than becoming an unlabelled partial phantom."""
    path = _manifest(tmp_path)
    mask = np.load(tmp_path / "myocardium_mask.npy")
    mask[0, 3:6, 4:7] = True
    t1 = np.load(tmp_path / "t1_ms.npy")
    t1[mask] = 1410.0
    np.save(tmp_path / "t1_ms.npy", t1)
    np.save(tmp_path / "myocardium_mask.npy", mask)
    payload = json.loads(path.read_text())
    payload["volumes"]["t1_ms"]["sha256"] = hashlib.sha256((tmp_path / "t1_ms.npy").read_bytes()).hexdigest()
    payload["volumes"]["myocardium_mask"]["sha256"] = hashlib.sha256((tmp_path / "myocardium_mask.npy").read_bytes()).hexdigest()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="LV myocardium boundary margin"):
        build_quantitative_gt(load_xcat_inputs(path))
