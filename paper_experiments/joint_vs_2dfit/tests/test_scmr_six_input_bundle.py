"""Exact-grid safety contracts for historical SCMR six-input production."""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.phase2_clean import scmr_six_input_bundle as producer
from paper_experiments.joint_vs_2dfit.phase2_clean.scmr_central_component_mask import run_scmr_central_component_mask
from paper_experiments.joint_vs_2dfit.phase2_clean.scmr_six_input_bundle import apply_exact_relation, exact_grid_relation


def _target_from_relation(relation: np.ndarray) -> np.ndarray:
    affine = np.eye(4); affine[:2, :2] = relation[:2, :2]; affine[:2, 3] = relation[:2, 3]
    return affine


@pytest.mark.parametrize("relation,expected", [
    (np.eye(4), "IDENTICAL_GRID"),
    (np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float), "EXACT_REORIENTABLE_GRID"),
    (np.array([[-1, 0, 0, 2], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float), "EXACT_REORIENTABLE_GRID"),
    (np.array([[0, -1, 0, 2], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float), "EXACT_REORIENTABLE_GRID"),
])
def test_exact_grid_proof_accepts_identity_transpose_flips_and_integer_offset(relation: np.ndarray, expected: str) -> None:
    proof = exact_grid_relation(np.eye(4), (3, 3), _target_from_relation(relation), (3, 3))
    source = np.arange(9).reshape(3, 3)
    assert proof["relation_type"] == expected and proof["no_interpolation"] is True
    assert np.array_equal(apply_exact_relation(source, proof, (3, 3)), source[np.asarray(proof["source_indices"])[0], np.asarray(proof["source_indices"])[1]].reshape(3, 3))


def test_exact_grid_proof_accepts_affine_proved_integer_translation() -> None:
    target = np.eye(4); target[0, 3] = 1.0
    proof = exact_grid_relation(np.eye(4), (3, 2), target, (2, 2))
    source = np.arange(6).reshape(3, 2)
    assert proof["integer_offset"] == [1, 0]
    assert np.array_equal(apply_exact_relation(source, proof, (2, 2)), source[1:, :])


@pytest.mark.parametrize("relation", [
    np.array([[1, 0, 0, .5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float),
    np.diag([2., 1., 1., 1.]),
    np.array([[1, .1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float),
    np.array([[0.6, -0.8, 0, 0], [0.8, 0.6, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float),
    np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], float),
])
def test_exact_grid_proof_rejects_fractional_scale_shear_and_arbitrary_rotation(relation: np.ndarray) -> None:
    with pytest.raises(ValueError, match="SCMR_SIX_INPUT_GEOMETRY_NOT_EXACTLY_REINDEXABLE"):
        exact_grid_relation(np.eye(4), (3, 3), relation, (3, 3))


def _write_nifti(path: Path, values: np.ndarray, affine: np.ndarray) -> str:
    nib.save(nib.Nifti1Image(np.asarray(values, np.float32), affine), str(path))
    return str(path)


def _current_cache(stack: str) -> dict[str, object]:
    count = {"sax": 14, "2ch": 15, "4ch": 12}[stack]
    shape = (2, 2)
    return {
        "group_idx": np.arange(count),
        "t1_ms": np.full((count, *shape), 1000.0, dtype=np.float32),
        "t2_ms": np.full((count, *shape), 50.0, dtype=np.float32),
        "source_support": np.array([[[True, True], [True, False]]] * count),
        "affine_lps_rc": np.array([np.diag([-1.0, -1.0, 1.0, 1.0])] * count),
        "provenance": {"stack": stack, "current_r0_sha": f"{stack}-sha"},
    }


def test_builds_complete_bundle_from_q004_l100_and_preserves_independent_parameter_geometries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The existing frozen Q004 native L100 artifact is the direct source contract."""
    source = np.array([[10.0, 20.0], [30.0, 40.0]])
    source_t2 = np.array([[11.0, 21.0], [31.0, 41.0]])
    flip_row = np.array([[-1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    fields = ("original_ms", "registered_ms", "synthetic_ms", "training_mask", "synthetic_support", "fitting_valid_mask")
    historical_files: dict[tuple[str, str], str] = {}
    for parameter, values, affine in (("t1", source, np.eye(4)), ("t2", source_t2, flip_row)):
        for field in fields:
            historical_files[parameter, field] = _write_nifti(tmp_path / f"{parameter}_{field}.nii.gz", values if field.endswith("_ms") else values > 0, affine)

    historical = {"subject_id": "CYJ", "slices": []}
    mappings = []
    q004_planes = []
    l100_root = tmp_path / "q004"
    for parameter in ("t1", "t2"):
        for stack, count in (("sax", 14), ("2ch", 15), ("4ch", 12)):
            for group in range(count):
                historical["slices"].append({"parameter": parameter.upper(), "stack": stack, "native_slice_id": group, **{field: historical_files[parameter, field] for field in fields}})
                mappings.append({"parameter": parameter, "stack": stack, "historical_native_slice_id": group, "original_group_idx": group})
                relative = Path("planes") / parameter / stack / f"group{group:03d}.npz"
                target = l100_root / relative; target.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target, l100_native=np.array([[True, True], [False, True]]))
                q004_planes.append({"parameter": parameter, "stack": stack, "group_idx": group, "output": str(relative)})
    historical_path = tmp_path / "common_mask_manifest.json"; historical_path.write_text(json.dumps(historical))
    group_map = tmp_path / "group_map.json"; group_map.write_text(json.dumps({"schema": "scmr_historical_group_mapping/v1", "mappings": mappings}))
    (l100_root / "manifest.json").write_text(json.dumps({"schema": "q004_direct_native_endpoint_audit/v2", "subject": "CYJ", "planes": q004_planes, "coordinate_contract": "final learned native lattice"}))
    sparse = tmp_path / "sparse.json"; sparse.write_text(json.dumps({"subject": "CYJ", "condition": "L100"}))
    monkeypatch.setattr(producer, "_load_all_verified_native_cache_groups", lambda _root, _manifest, stack: _current_cache(stack))

    output = producer.build_scmr_six_input_bundle(historical_common_mask_manifest=historical_path, group_map_manifest=group_map, r0_native_maps_root=tmp_path / "r0", sparse_manifest=sparse, phase4r_l100_root=l100_root, output=tmp_path / "six_input")

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema"] == "scmr_six_input_native_bundle/v1"
    assert len(manifest["planes"]) == 82
    with np.load(output / "planes/t1/sax/group000.npz") as payload:
        assert np.array_equal(payload["source_support"], _current_cache("sax")["source_support"][0])
        assert np.array_equal(payload["registered"], source)
        assert "phase4r_l100" in payload.files
    with np.load(output / "planes/t2/sax/group000.npz") as payload:
        assert np.array_equal(payload["registered"], np.flipud(source_t2))
    masks = run_scmr_central_component_mask(output, tmp_path / "scmr_masks")
    assert (masks / "planes/t2/4ch/group011.npz").is_file()
    with pytest.raises(FileExistsError):
        producer.build_scmr_six_input_bundle(historical_common_mask_manifest=historical_path, group_map_manifest=group_map, r0_native_maps_root=tmp_path / "r0", sparse_manifest=sparse, phase4r_l100_root=l100_root, output=output)


def test_discovery_fails_closed_for_mismatched_group_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(producer, "_load_all_verified_native_cache_groups", lambda _root, _manifest, stack: _current_cache(stack))
    historical = tmp_path / "historical.json"; historical.write_text(json.dumps({"slices": []}))
    groups = tmp_path / "groups.json"; groups.write_text(json.dumps({"schema": "scmr_historical_group_mapping/v1", "mappings": []}))
    l100 = tmp_path / "l100"; l100.mkdir(); (l100 / "manifest.json").write_text(json.dumps({"schema": "q004_direct_native_endpoint_audit/v2", "planes": []}))
    sparse = tmp_path / "sparse.json"; sparse.write_text(json.dumps({"subject": "CYJ", "condition": "L100"}))
    report = producer.discover_scmr_six_input_sources(historical_common_mask_manifest=historical, group_map_manifest=groups, r0_native_maps_root=tmp_path / "r0", sparse_manifest=sparse, phase4r_l100_root=l100)
    assert report["complete"] is False
    assert {item["missing"] for item in report["missing"]} == {"explicit_historical_group_mapping"}
    assert report["coverage"]["t2"]["4ch"] == {"required": 12, "found": 0, "missing": 12}
