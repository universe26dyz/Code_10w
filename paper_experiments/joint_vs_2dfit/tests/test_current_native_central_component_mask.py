"""Current-native formal-mask contract: no historical reconstruction support."""
from __future__ import annotations

import json
import inspect
from pathlib import Path

import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.phase2_clean import current_native_central_component_mask as masks


COUNTS = {"sax": 14, "2ch": 15, "4ch": 12}


def _cache(stack: str) -> dict[str, object]:
    count = COUNTS[stack]
    return {"group_idx": np.arange(count), "t1_ms": np.full((count, 3, 3), 1000.0, np.float32), "t2_ms": np.full((count, 3, 3), 50.0, np.float32), "source_support": np.array([[[True, True, True], [True, False, True], [True, True, True]]] * count), "provenance": {"stack": stack, "r0": "frozen"}}


def _inputs(tmp_path: Path, *, bad_l100_shape: bool = False) -> tuple[Path, Path]:
    root = tmp_path / "q004"; planes = []
    for parameter in ("t1", "t2"):
        for stack, count in COUNTS.items():
            for group in range(count):
                relative = Path("planes") / parameter / stack / f"group{group:03d}.npz"
                target = root / relative; target.parent.mkdir(parents=True, exist_ok=True)
                shape = (2, 2) if bad_l100_shape and parameter == "t1" and stack == "sax" and group == 0 else (3, 3)
                np.savez_compressed(target, l100_native=np.array([[True, True, False], [True, True, False], [False, False, True]]) if shape == (3, 3) else np.ones(shape, bool))
                planes.append({"parameter": parameter, "stack": stack, "group_idx": group, "output": str(relative)})
    (root / "manifest.json").write_text(json.dumps({"schema": "q004_direct_native_endpoint_audit/v2", "planes": planes}))
    sparse = tmp_path / "sparse.json"; sparse.write_text(json.dumps({"subject": "CYJ", "condition": "L100"}))
    return root, sparse


def test_primary_mask_is_current_source_and_frozen_l100_only() -> None:
    source = np.array([[True, True, False], [True, False, True], [False, True, True]])
    l100 = np.array([[True, False, True], [True, True, True], [True, False, False]])
    values, record = masks.build_current_native_masks(source, l100)
    assert np.array_equal(values["base_native_support"], source & l100)
    assert np.array_equal(values["primary_native_mask"], np.array([[True, False, False], [True, False, False], [False, False, False]]))
    assert record["connectivity"] == 8 and record["minimum_ratio"] == pytest.approx(.25)


def test_selection_uses_center_when_large_enough_else_largest_and_handles_empty() -> None:
    center = np.array([[True, False, False, False, False], [True, False, False, False, False], [True, False, True, True, False], [True, False, False, False, False], [True, False, False, False, False]])
    selected, record = masks.select_current_native_central_component(center)
    assert record["selection"] == "contains_image_center" and selected[2, 2] and selected.sum() == 2
    large = np.array([[True, False, False, False, False], [True, False, False, False, False], [True, False, True, False, False], [True, False, False, False, False], [True, False, False, False, False]])
    selected, record = masks.select_current_native_central_component(large)
    assert record["selection"] == "largest_component" and selected.sum() == 5
    empty, record = masks.select_current_native_central_component(np.zeros((3, 3), bool))
    assert not empty.any() and record["selection"] == "empty_support"


def test_builds_immutable_complete_current_native_artifact_and_validates_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(masks, "_load_all_verified_native_cache_groups", lambda _root, _sparse, stack: _cache(stack))
    l100, sparse = _inputs(tmp_path)
    output = masks.build_current_native_central_component_mask(r0_native_maps_root=tmp_path / "r0", sparse_manifest=sparse, l100_native_support_root=l100, output=tmp_path / "masks")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema"] == "phase4r_current_native_central_component_mask/v1"
    assert len(manifest["planes"]) == 82
    with np.load(output / "planes/t1/sax/group000.npz") as plane:
        assert set(("source_support", "phase4r_l100", "base_native_support", "primary_native_mask", "reference")).issubset(plane.files)
        assert np.array_equal(plane["base_native_support"], plane["source_support"] & plane["phase4r_l100"])
    loaded, _ = masks.load_validated_current_native_mask_plane(output, parameter="t1", stack="sax", original_group_idx=0, shape_rc=(3, 3), source_support=_cache("sax")["source_support"][0])
    assert set(loaded) == {"source_support", "phase4r_l100", "base_native_support", "primary_native_mask", "reference"}
    assert loaded["reference"].dtype == np.dtype(float)
    assert loaded["primary_native_mask"].shape == (3, 3)
    with pytest.raises(ValueError, match="source_support"):
        masks.load_validated_current_native_mask_plane(output, parameter="t1", stack="sax", original_group_idx=0, shape_rc=(3, 3), source_support=np.ones((3, 3), bool))
    with pytest.raises(FileExistsError):
        masks.build_current_native_central_component_mask(r0_native_maps_root=tmp_path / "r0", sparse_manifest=sparse, l100_native_support_root=l100, output=output)


def test_loader_uses_python38_compatible_dictionary_assembly() -> None:
    assert "} | {" not in inspect.getsource(masks.load_validated_current_native_mask_plane)


def test_l100_shape_mismatch_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(masks, "_load_all_verified_native_cache_groups", lambda _root, _sparse, stack: _cache(stack))
    l100, sparse = _inputs(tmp_path, bad_l100_shape=True)
    with pytest.raises(ValueError, match="current native lattice"):
        masks.build_current_native_central_component_mask(r0_native_maps_root=tmp_path / "r0", sparse_manifest=sparse, l100_native_support_root=l100, output=tmp_path / "masks")
