"""CPU contracts for immutable direct-native central and PSF128 bundles."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from paper_experiments.joint_vs_2dfit.phase2_clean import direct_native_bundles as bundles


def _prepared(path: Path, groups: list[int], shape: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = np.zeros((len(groups), *shape), dtype=np.float32)
    np.savez_compressed(
        path,
        images=images,
        group_idx=np.asarray(groups, dtype=np.int64),
        pixel_spacing_rc_mm=np.tile(np.array([[1.5, 2.0]], dtype=np.float32), (len(groups), 1)),
    )


def test_canonical_l100_lattice_assigns_stable_global_indices_independent_of_requested_groups(tmp_path: Path) -> None:
    roots = []
    for stack, groups in (("sax", [2, 1]), ("2ch", [4]), ("4ch", [3])):
        path = tmp_path / stack / "observations.npz"
        _prepared(path, groups, (2, 3))
        roots.append(path)

    lattice = bundles.canonical_l100_native_lattice(roots)
    assert [(item.stack, item.original_group_idx, item.global_query_start) for item in lattice] == [
        ("sax", 1, 0), ("sax", 2, 6), ("2ch", 4, 12), ("4ch", 3, 18),
    ]
    assert lattice[1].global_query_indices.reshape(2, 3).tolist() == [[6, 7, 8], [9, 10, 11]]


def test_psf128_offsets_are_identical_when_lattice_is_chunked() -> None:
    whole_rng = np.random.default_rng(bundles.FORMAL_PSF128_SEED)
    whole = bundles.draw_psf128_physical_offsets(whole_rng, 4)
    chunked_rng = np.random.default_rng(bundles.FORMAL_PSF128_SEED)
    chunked = np.concatenate((bundles.draw_psf128_physical_offsets(chunked_rng, 1), bundles.draw_psf128_physical_offsets(chunked_rng, 3)))
    np.testing.assert_array_equal(chunked, whole)
