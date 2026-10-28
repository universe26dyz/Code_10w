"""Current-native formal-mask contract: no historical reconstruction support."""
from __future__ import annotations

import json
import inspect
from pathlib import Path

import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.phase2_clean import current_native_central_component_mask as masks
from paper_experiments.joint_vs_2dfit.phase2_clean import formal_l100_sparse_manifest as l100_manifests


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
    sparse = tmp_path / "sparse.json"; sparse.write_text(json.dumps({"schema_version": "phase4r_sparse_v1", "subject": "CYJ", "condition": "L100"}))
    return root, sparse


def test_primary_mask_is_current_source_and_frozen_l100_only() -> None:
    source = np.array([[True, True, False], [True, False, True], [False, True, True]])
    l100 = np.array([[True, False, True], [True, True, True], [True, False, False]])
    values, record = masks.build_current_native_masks(source, l100)
    assert np.array_equal(values["base_native_support"], source & l100)
    assert np.array_equal(values["primary_native_mask"], np.array([[True, False, False], [True, False, False], [False, False, False]]))
    assert record["connectivity"] == 8 and record["minimum_ratio"] == pytest.approx(.25)


def test_manual_outer_segment_replaces_central_component_for_slice_metrics() -> None:
    source = np.ones((5, 5), bool)
    l100 = np.ones((5, 5), bool)
    outer = np.zeros((5, 5), bool)
    outer[1:4, 1:4] = True
    values, record = masks.build_current_native_masks(source, l100, manual_outer_region=outer)
    assert np.array_equal(values["primary_native_mask"], outer)
    assert np.array_equal(values["manual_outer_region"], outer)
    assert record["selection"] == "manual_outer_segment"


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


def test_current_native_mask_uses_the_manifest_subject_and_observed_group_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    counts = {"sax": 1, "2ch": 2, "4ch": 1}

    def cache(stack: str) -> dict[str, object]:
        count = counts[stack]
        return {"group_idx": np.arange(count), "t1_ms": np.full((count, 3, 3), 1000.0, np.float32), "t2_ms": np.full((count, 3, 3), 50.0, np.float32), "source_support": np.ones((count, 3, 3), bool), "provenance": {"stack": stack, "r0": "frozen"}}

    monkeypatch.setattr(masks, "_load_all_verified_native_cache_groups", lambda _root, _sparse, stack: cache(stack))
    root = tmp_path / "l100"; planes = []
    for parameter in ("t1", "t2"):
        for stack, count in counts.items():
            for group in range(count):
                relative = Path("planes") / parameter / stack / f"group{group:03d}.npz"
                target = root / relative; target.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target, l100_native=np.ones((3, 3), bool))
                planes.append({"parameter": parameter, "stack": stack, "group_idx": group, "output": str(relative)})
    (root / "manifest.json").write_text(json.dumps({"schema": "q004_direct_native_endpoint_audit/v2", "planes": planes}))
    sparse = tmp_path / "sparse.json"; sparse.write_text(json.dumps({"schema_version": "phase4r_sparse_v1", "subject": "DYL_2", "condition": "L100"}))
    output = masks.build_current_native_central_component_mask(r0_native_maps_root=tmp_path / "r0", sparse_manifest=sparse, l100_native_support_root=root, output=tmp_path / "masks")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["subject"] == "DYL_2"
    assert len(manifest["planes"]) == 2 * sum(counts.values())


def test_loader_uses_python38_compatible_dictionary_assembly() -> None:
    assert "} | {" not in inspect.getsource(masks.load_validated_current_native_mask_plane)


def test_l100_shape_mismatch_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(masks, "_load_all_verified_native_cache_groups", lambda _root, _sparse, stack: _cache(stack))
    l100, sparse = _inputs(tmp_path, bad_l100_shape=True)
    with pytest.raises(ValueError, match="current native lattice"):
        masks.build_current_native_central_component_mask(r0_native_maps_root=tmp_path / "r0", sparse_manifest=sparse, l100_native_support_root=l100, output=tmp_path / "masks")


def _formal_v2_l100(*, profile_id: str = "L100", case_id: str = "L100", heldout: bool = False, stride_2ch: bool = False) -> dict[str, object]:
    stacks = {}
    for stack, count in COUNTS.items():
        groups = list(range(count))
        stacks[stack] = {
            "mode": "stride" if stack == "2ch" and stride_2ch else "all",
            "ordered_source_group_ids": groups,
            "retained_original_group_ids": groups,
            "heldout_original_group_ids": [groups[-1]] if heldout else [],
        }
        if heldout:
            stacks[stack]["retained_original_group_ids"] = groups[:-1]
    return {
        "schema_version": "phase4r_sparse_profile_v2", "status": "ELIGIBLE",
        "subject": "DYL_2", "profile_id": profile_id, "case_id": case_id,
        "fold_count": 1, "fold_id": None, "stacks": stacks,
    }


def test_formal_v2_l100_is_admitted_by_mask_builder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(masks, "_load_all_verified_native_cache_groups", lambda _root, _sparse, stack: _cache(stack))
    payload = _formal_v2_l100()
    validated: list[Path] = []
    monkeypatch.setattr(l100_manifests, "validate_sparse_profile_manifest", lambda path: validated.append(Path(path)) or payload)
    l100, _legacy_sparse = _inputs(tmp_path)
    sparse = tmp_path / "formal_l100.json"; sparse.write_text(json.dumps(payload))
    output = masks.build_current_native_central_component_mask(r0_native_maps_root=tmp_path / "r0", sparse_manifest=sparse, l100_native_support_root=l100, output=tmp_path / "masks")
    manifest = json.loads((output / "manifest.json").read_text())
    assert validated == [sparse]
    assert manifest["sparse_manifest_schema"] == "phase4r_sparse_profile_v2"
    assert manifest["sparse_identity"] == {"profile_id": "L100", "case_id": "L100", "fold_id": None, "fold_count": 1}


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (_formal_v2_l100(profile_id="L50", case_id="L50_fold0"), "profile_id"),
        (_formal_v2_l100(heldout=True), "full retained source lattice"),
        (_formal_v2_l100(stride_2ch=True), "mode"),
    ],
)
def test_formal_v2_l100_rejects_nonidentity_profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: dict[str, object], match: str) -> None:
    path = tmp_path / "sparse.json"; path.write_text(json.dumps(payload))
    monkeypatch.setattr(l100_manifests, "validate_sparse_profile_manifest", lambda _path: payload)
    with pytest.raises(ValueError, match=match):
        l100_manifests.validate_formal_l100_sparse_manifest(path)


def test_legacy_v1_l100_remains_admitted_and_unknown_schema_fails_closed(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.json"; legacy.write_text(json.dumps({"schema_version": "phase4r_sparse_v1", "subject": "CYJ", "condition": "L100"}))
    result = l100_manifests.validate_formal_l100_sparse_manifest(legacy)
    assert result["subject"] == "CYJ" and result["sparse_manifest_schema"] == "phase4r_sparse_v1"
    unknown = tmp_path / "unknown.json"; unknown.write_text(json.dumps({"schema_version": "unknown", "subject": "CYJ", "condition": "L100"}))
    with pytest.raises(ValueError, match="unsupported sparse manifest schema"):
        l100_manifests.validate_formal_l100_sparse_manifest(unknown)
