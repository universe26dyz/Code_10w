from __future__ import annotations

import numpy as np
import pytest

from trad.evaluation.scmr.metrics import agreement_metrics


def test_agreement_metrics_excludes_invalid_and_background() -> None:
    reference = np.array([[1000.0, 1100.0], [0.0, np.nan]])
    prediction = np.array([[1010.0, 1080.0], [9999.0, 1200.0]])
    mask = np.array([[True, True], [False, True]])
    values = agreement_metrics(reference, prediction, mask, min_pixels=2)
    assert values["N"] == 2
    assert values["bias_ms"] == -5.0
    assert values["MAE_ms"] == 15.0
    assert np.isclose(values["RMSE_ms"], np.sqrt(250.0))


def test_agreement_metrics_preserves_ms_units_and_boolean_masking() -> None:
    reference = np.array([[1000.0, 1200.0], [1400.0, 1600.0]])
    prediction = reference + 50.0
    values = agreement_metrics(reference, prediction, np.array([[True, False], [False, True]]), min_pixels=2)
    assert values["N"] == 2
    assert values["bias_ms"] == 50.0
    assert values["RMSE_ms"] == 50.0
    assert np.isclose(values["NRMSE"], 50.0 / 600.0)


def test_scmr_runner_validates_alignment_and_writes_figures(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    nib = pytest.importorskip("nibabel")
    from argparse import Namespace

    from trad.evaluation.scmr.reference_2d import sha256
    from trad.evaluation.scmr.run_scmr_fig12 import run

    subject = "TEST"
    prepared_root, preprocessed_root, run_root, reference_root = [tmp_path / name for name in ("prepared", "preprocessed", "run", "reference")]
    reference_root.mkdir(); run_root.mkdir()
    ref_hashes, maps = {}, {}
    rows, cols = np.mgrid[:9, :9]
    for stack_offset, stack in enumerate(("sax", "2ch", "4ch")):
        mat = preprocessed_root / subject / stack / "preprocessed.mat"
        mat.parent.mkdir(parents=True)
        with h5py.File(mat, "w") as handle:
            handle.create_dataset("proof", data=np.array([stack_offset]))
        ref_hashes[stack] = sha256(mat)
        base = 1000.0 + stack_offset * 100 + rows * 10 + cols
        reference_file = reference_root / f"{stack}.npz"
        np.savez_compressed(reference_file, t1_ms=base[None], t2_ms=(base / 20)[None], valid_mask=np.ones((1, 9, 9), dtype=bool), group_idx=np.array([0]))
        maps[stack] = reference_file.name
        prepared = prepared_root / subject / stack / "observations.npz"
        prepared.parent.mkdir(parents=True)
        np.savez_compressed(prepared, placeholder=np.array([1]))
        weights = np.arange(10, dtype=np.int64)
        groups = np.zeros(10, dtype=np.int64)
        masks = np.ones((10, 9, 9), dtype=bool)
        np.savez_compressed(run_root / f"t1_t2_native_plane_{stack}.npz", t1_ms=np.repeat((base + 5)[None], 10, axis=0), t2_ms=np.repeat((base / 20 + 1)[None], 10, axis=0), group_idx=groups, weight_idx=weights, masks=masks)
        observed = np.repeat((rows + cols + stack_offset)[None].astype(np.float32), 10, axis=0)
        np.savez_compressed(run_root / f"signal_reprojection_{stack}.npz", observed=observed, predicted=observed + 0.5, residual=np.full_like(observed, 0.5), group_idx=groups, weight_idx=weights, masks=masks)
    affine = np.eye(4)
    native_t1, native_t2 = reference_root / "sax_T1_stack_ms.nii.gz", reference_root / "sax_T2_stack_ms.nii.gz"
    nib.save(nib.Nifti1Image((1000 + rows * 10 + cols)[..., None].astype(np.float32), affine), native_t1)
    nib.save(nib.Nifti1Image((50 + rows / 2 + cols / 20)[..., None].astype(np.float32), affine), native_t2)
    nib.save(nib.Nifti1Image((1100 + rows * 8 + cols)[..., None].astype(np.float32), affine), run_root / "T1_3D.nii.gz")
    nib.save(nib.Nifti1Image((55 + rows / 3 + cols / 30)[..., None].astype(np.float32), affine), run_root / "T2_3D.nii.gz")
    (run_root / "experiment_manifest.json").write_text("{}")
    (reference_root / "native_reference_manifest.json").write_text(__import__("json").dumps({"subject_id": subject, "preprocessing_semantics": "MP-PCA(MIND_mag_reg)", "map_units": "ms", "preprocessed_mat_sha256": ref_hashes, "maps": maps, "figure1_native_stacks": {"t1": native_t1.name, "t2": native_t2.name}}))
    output = tmp_path / "out"
    result = run(Namespace(subject_id=subject, prepared_root=str(prepared_root), preprocessed_root=str(preprocessed_root), trad_run=str(run_root), native_reference_root=str(reference_root), output=str(output), legacy_source=".", plane_axis="x", plane_index=4, figure1_reference_group=0, selected_group=["sax=0", "2ch=0", "4ch=0"], dry_run=False, overwrite=False))
    assert result == output
    assert (output / "figure1" / "Figure1_native_vs_trad_through_plane.png").is_file()
    assert (output / "figure1" / "candidates" / "Figure1_T1_x_candidates.png").is_file()
    assert (output / "figure2" / "Figure2_T1_native_vs_reprojection.pdf").is_file()
    assert (output / "metrics" / "quantitative_agreement_per_slice.csv").is_file()
    manifest = __import__("json").loads((output / "evaluation_manifest.json").read_text())
    assert manifest["figure1"]["same_world_plane_check"] == "PASS"
    assert manifest["figure3"].startswith("SKIPPED / ROI_PENDING")
