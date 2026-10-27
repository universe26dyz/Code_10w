from __future__ import annotations

from pathlib import Path

import numpy as np


STACKS = ("sax", "2ch", "4ch")


def _write_source(path: Path, group_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = np.repeat(np.arange(group_count, dtype=np.int64), 10)
    weights = np.tile(np.arange(10, dtype=np.int64), group_count)
    rows = len(groups)
    affine = np.repeat(np.eye(4, dtype=np.float64)[None], rows, axis=0)
    affine[:, :3, 0] = [1.0, 0.0, 0.0]
    affine[:, :3, 1] = [0.0, 1.0, 0.0]
    affine[:, :3, 2] = [0.0, 0.0, 5.0]
    affine[:, 2, 3] = groups.astype(float)
    np.savez_compressed(
        path,
        images=np.ones((rows, 2, 2), dtype=np.float32),
        masks=np.ones((rows, 2, 2), dtype=bool),
        group_idx=groups,
        weight_idx=weights,
        stack_idx=np.zeros(rows, dtype=np.int64),
        acquisition_time_ms=np.zeros(rows, dtype=np.float64),
        timing9_ms=np.repeat(np.arange(9, dtype=np.float64)[None], rows, axis=0),
        affine_lps_rc=affine,
        pixel_spacing_rc_mm=np.repeat([[1.0, 1.0]], rows, axis=0),
        slice_thickness_mm=np.full(rows, 5.0),
        tr_ms=np.full(rows, 2.61),
        vps=np.full(rows, 87, dtype=np.int64),
    )


def _audit(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        groups = np.unique(data["group_idx"])
        positions = [float(data["affine_lps_rc"][np.flatnonzero(data["group_idx"] == group)[0], 2, 3]) for group in groups]
    gap = abs(positions[1] - positions[0]) if len(positions) > 1 else 0.0
    return {"regular_gap": True, "gap_mm": gap, "group_count": len(groups)}


def test_legacy_l50_manifest_is_formally_validated_without_changing_group_id_selection(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import sha256
    from paper_experiments.joint_vs_2dfit.phase2_clean.sparse import build_sparse_condition
    from paper_experiments.joint_vs_2dfit.phase2_clean.sparse_profile_manifest import validate_sparse_profile_manifest
    from quality_experiments.Q002_B6_roi_same_anchor_mse.contracts import profiled_observation_paths

    source_root = tmp_path / "prepared"
    for stack in STACKS:
        _write_source(source_root / "HHZ" / stack / "observations.npz", group_count=4)
    output_root = tmp_path / "sparse"
    generated = build_sparse_condition(
        source_root,
        "HHZ",
        "L50",
        0,
        output_root,
        git_sha="a" * 40,
        audit_stack=_audit,
    )
    manifest = Path(str(generated["manifest_path"]))

    validated = validate_sparse_profile_manifest(
        manifest,
        subject="HHZ",
        strict=True,
        expected_generator_git_sha="a" * 40,
        expected_manifest_sha256=sha256(manifest),
    )

    assert validated["case_id"] == "L50_fold0"
    assert validated["profile_id"] == "legacy_group_id_modulo_L50"
    assert validated["selection_policy"] == "group_idx_modulo"
    assert validated["stacks"]["2ch"]["retained_original_group_ids"] == [0, 2]
    assert validated["stacks"]["4ch"]["retained_original_group_ids"] == [0, 2]
    assert profiled_observation_paths(validated["prepared_root"], "HHZ", manifest) == [
        Path(validated["prepared_root"]) / stack / "observations.npz" for stack in STACKS
    ]
