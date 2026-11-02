from pathlib import Path

import pytest

from scripts.reconstruct_subject import resolve_subject_observations
from scripts.run_training import load_training_config, reject_blocked_experiment


def test_reconstruct_subject_resolves_exact_three_stacks_and_never_lax(tmp_path):
    for stack in ("sax", "2ch", "4ch"):
        path = tmp_path / "CYJ" / stack / "observations.npz"; path.parent.mkdir(parents=True); path.write_bytes(b"x")
    paths = resolve_subject_observations(tmp_path, "CYJ")
    assert [path.parent.name for path in paths] == ["sax", "2ch", "4ch"]
    assert all("lax" not in str(path) for path in paths)


def test_reg_full_fails_before_any_reconstruction_work():
    config = load_training_config(Path("configs/experiments_6k/REG_FULL.yaml"))
    with pytest.raises(ValueError, match="REG_FULL.*blocked"):
        reject_blocked_experiment(config)


@pytest.mark.parametrize("name,k,tv", [("B6", 8, 1e-4), ("TV2", 8, 2e-4), ("TV3", 8, 3e-4), ("PSF4", 4, 1e-4), ("PSF16", 16, 1e-4)])
def test_first_wave_configs_resolve_to_explicit_6k_contract(name, k, tv):
    config = load_training_config(Path(f"configs/experiments_6k/{name}.yaml"))
    assert config["training"]["stage_a_iterations"] == 2000
    assert config["training"]["stage_b_iterations"] == 4000
    assert config["training"]["seed"] == 20260911 and config["training"]["batch_size"] == 640
    assert config["training"]["psf_samples"] == k and config["psf"]["training"]["n_samples"] == k
    assert config["spatial_regularization"]["t1"]["weight"] == tv
    assert config["spatial_regularization"]["t2"]["weight"] == tv
    assert config["spatial_regularization"]["b1"]["weight"] == 1e-3
