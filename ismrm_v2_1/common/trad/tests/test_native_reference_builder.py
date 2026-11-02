from __future__ import annotations

import json

import numpy as np
import pytest


def _write_inputs(root, subject: str, stack: str, groups: int, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h5py = pytest.importorskip("h5py")
    from scipy.io import savemat

    preprocessed = root / "preprocessed" / subject / stack / "preprocessed.mat"
    preprocessed.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(preprocessed, "w") as handle:
        handle.create_dataset("current_preprocessing_proof", data=np.array([groups, *shape]))
    observations = root / "prepared" / subject / stack / "observations.npz"
    observations.parent.mkdir(parents=True, exist_ok=True)
    group_idx = np.repeat(np.arange(groups, dtype=np.int64), 10)
    weight_idx = np.tile(np.arange(10, dtype=np.int64), groups)
    affine = np.repeat(np.eye(4, dtype=np.float64)[None], groups * 10, axis=0)
    affine[:, :3, 0] = [0.0, 1.25, 0.0]  # row direction, LPS
    affine[:, :3, 1] = [1.25, 0.0, 0.0]  # column direction, LPS
    affine[:, :3, 2] = [0.0, 0.0, 8.0]
    for group in range(groups):
        affine[group_idx == group, :3, 3] = [5.0, 10.0, 40.0 - 8.0 * group]
    np.savez_compressed(observations, images=np.zeros((groups * 10, *shape), dtype=np.float32), masks=np.ones((groups * 10, *shape), dtype=bool), group_idx=group_idx, weight_idx=weight_idx, affine_lps_rc=affine)
    rows, cols = np.mgrid[:shape[0], :shape[1]]
    t1 = np.stack([1000.0 + group * 100 + rows * 10 + cols for group in range(groups)]).astype(np.float32)
    t2 = (t1 / 20).astype(np.float32)
    mapping = root / "mapping" / f"{stack}_mapping.mat"
    mapping.parent.mkdir(parents=True, exist_ok=True)
    savemat(mapping, {"t1_ms": t1, "t2_ms": t2, "valid_mask": np.ones_like(t1, dtype=bool), "group_idx": np.arange(groups, dtype=np.int32)})
    return t1, t2


def test_package_native_reference_uses_prepared_group_order_and_current_sax_geometry(tmp_path) -> None:
    nib = pytest.importorskip("nibabel")
    from trad.evaluation.scmr.build_native_reference import package_native_reference
    from trad.evaluation.scmr.reference_2d import load_verified_reference

    subject, shape = "TEST", (3, 4)
    expected = {stack: _write_inputs(tmp_path, subject, stack, groups, shape) for stack, groups in (("sax", 2), ("2ch", 1), ("4ch", 1))}
    output = tmp_path / "bundle"
    manifest = package_native_reference(subject, tmp_path / "prepared", tmp_path / "preprocessed", tmp_path / "mapping", output)

    loaded = load_verified_reference(output, subject, tmp_path / "preprocessed")
    assert loaded.manifest["preprocessing_semantics"] == "MP-PCA(MIND_mag_reg)"
    assert loaded.manifest["map_units"] == "ms"
    assert np.array_equal(loaded.stacks["sax"].t1_ms, expected["sax"][0])
    assert np.array_equal(loaded.stacks["sax"].group_idx, np.array([0, 1]))
    image = nib.load(str(output / "sax_T1_stack_ms.nii.gz"))
    assert image.shape == (3, 4, 2)
    assert np.allclose(image.affine @ [0, 0, 0, 1], [-5, -10, 40, 1])
    assert np.allclose(image.affine @ [0, 0, 1, 1], [-5, -10, 32, 1])
    assert manifest["group_ordering_verified"] is True
    assert json.loads((output / "native_reference_manifest.json").read_text())["maps"]["sax"] == "sax.npz"
