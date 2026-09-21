import numpy as np

from evaluation.scmr.map_domain_psf import (
    NativeGrid,
    grids_from_trad_prepared,
    reproject_2dfit_exported_maps,
    reproject_trad_exported_maps,
    reproject_volume_to_native_map_psf,
)
from evaluation.scmr.run_map_domain_psf_comparison import run


def _grid(*, translation=(12.0, 12.0, 12.0), rotation=None):
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    affine = np.eye(4)
    affine[:3, 0] = rotation[:, 1] * 1.0  # row/local y
    affine[:3, 1] = rotation[:, 0] * 1.0  # col/local x
    affine[:3, 2] = rotation[:, 2] * 4.0  # local z/thickness
    affine[:3, 3] = np.asarray(translation) - affine[:3, 0] - affine[:3, 1]
    return NativeGrid((3, 3), affine, np.array([1.0, 1.0, 4.0]))


def test_constant_volume_k32_and_k128_are_preserved():
    volume = np.full((31, 31, 31), 777.0)
    affine = np.eye(4)
    grid = _grid()
    for k in (32, 128):
        result = reproject_volume_to_native_map_psf(volume, affine, grid, n_samples=k, seed=20260921)
        assert result.support.all()
        assert np.array_equal(result.values, np.full((3, 3), 777.0, dtype=np.float32))


def test_sampling_is_exactly_seeded_and_deterministic():
    volume = np.indices((31, 31, 31))[2].astype(float)
    grid = _grid()
    first = reproject_volume_to_native_map_psf(volume, np.eye(4), grid, n_samples=32, seed=8)
    second = reproject_volume_to_native_map_psf(volume, np.eye(4), grid, n_samples=32, seed=8)
    third = reproject_volume_to_native_map_psf(volume, np.eye(4), grid, n_samples=32, seed=9)
    assert np.array_equal(first.values, second.values, equal_nan=True)
    assert not np.array_equal(first.values, third.values, equal_nan=True)


def test_linear_gradient_uses_final_pose_and_slice_normal_in_ras():
    # V(x,y,z)=z.  A 90-degree rotation about y makes local through-plane +z
    # point toward world +x; the symmetric PSF mean must stay at x=12.
    volume = np.indices((31, 31, 31))[0].astype(float)
    rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    result = reproject_volume_to_native_map_psf(volume, np.eye(4), _grid(rotation=rotation), n_samples=128, seed=20260921)
    assert result.support.all()
    assert abs(float(result.values[1, 1]) - 12.0) < 0.35


def test_trad_and_2dfit_wrappers_call_the_same_map_domain_operator(monkeypatch):
    import evaluation.scmr.map_domain_psf as module

    calls = []
    original = module.reproject_volume_to_native_map_psf

    def wrapped(*args, **kwargs):
        calls.append(kwargs["n_samples"])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "reproject_volume_to_native_map_psf", wrapped)
    volumes = {parameter: (np.full((31, 31, 31), 1000.0), np.eye(4)) for parameter in ("T1", "T2")}
    grid = _grid()
    trad = reproject_trad_exported_maps(volumes, [grid], n_samples=32, seed=1)
    fit = reproject_2dfit_exported_maps(volumes, {"T1": [grid], "T2": [grid]}, n_samples=128, seed=1)
    assert calls == [32, 32, 128, 128]
    assert trad["T1"][0].support.all() and fit["T2"][0].support.all()


def test_trad_prepared_geometry_uses_exported_final_trans_first_pose(tmp_path):
    source = tmp_path / "observations.npz"
    affine_lps = np.eye(4)
    affine_lps[:3, 0] = [0.0, 1.0, 0.0]
    affine_lps[:3, 1] = [1.0, 0.0, 0.0]
    affine_lps[:3, 2] = [0.0, 0.0, 4.0]
    np.savez_compressed(
        source,
        images=np.zeros((10, 3, 3), dtype=np.float32),
        group_idx=np.zeros(10, dtype=np.int64),
        weight_idx=np.arange(10, dtype=np.int64),
        affine_lps_rc=np.repeat(affine_lps[None], 10, axis=0),
        pixel_spacing_rc_mm=np.ones((10, 2)),
        slice_thickness_mm=np.full(10, 4.0),
    )
    poses = tmp_path / "final_rigid_poses.json"
    poses.write_text('{"coordinate_convention":"physical RAS mm; trans_first=true; world=R@(local+T)","matrix_physical":[[[1,0,0,12],[0,1,0,12],[0,0,1,12]]]}')
    grid = grids_from_trad_prepared([source], poses)[0]
    assert np.allclose(grid.affine_ras_rc[:3, 3] + grid.affine_ras_rc[:3, 0] + grid.affine_ras_rc[:3, 1], [12.0, 12.0, 12.0])


def test_2dfit_cli_writes_nonoverwriting_machine_outputs_and_manifest(tmp_path):
    import argparse
    import json
    import nibabel as nib

    volume = np.full((31, 31, 31), 1000.0, dtype=np.float32)
    t1, t2 = tmp_path / "T1_3D.nii.gz", tmp_path / "T2_3D.nii.gz"
    nib.save(nib.Nifti1Image(volume, np.eye(4)), t1); nib.save(nib.Nifti1Image(volume, np.eye(4)), t2)
    folders = {}
    affine = np.eye(4); affine[:3, 2] *= 4.0; affine[:3, 3] = [11.0, 11.0, 12.0]
    for parameter in ("T1", "T2"):
        folder = tmp_path / parameter / "registered_slices"; folder.mkdir(parents=True)
        nib.save(nib.Nifti1Image(np.zeros((3, 3, 1), dtype=np.float32), affine), folder / "0.nii.gz")
        folders[parameter] = folder
    output = tmp_path / "output"
    manifest_path = run(argparse.Namespace(method="2dfit", subject_id="SYN", t1_volume=t1, t2_volume=t2, n_samples=32, seed=7, output=output, prepared_root=None, final_poses=None, t1_registered_slices=folders["T1"], t2_registered_slices=folders["T2"], native_reference=None, preprocessed_root=None, mask_bundle=None))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["domain"] == "map" and not manifest["bloch_used"] and not manifest["dictionary_used"]
    assert manifest["n_samples"] == 32
    assert (output / "T1_native_map_domain_psf_000.npz").is_file()
