from pathlib import Path

from scripts.run_training import load_training_config


def test_b6_config_inherits_only_the_6k_stage_b_change():
    config = load_training_config(Path("configs/experiments_6k/B6.yaml"))
    assert config["training"]["stage_a_iterations"] == 2000
    assert config["training"]["stage_b_iterations"] == 4000
    assert config["training"]["seed"] == 20260911
    assert config["psf"]["training"]["n_samples"] == 8
