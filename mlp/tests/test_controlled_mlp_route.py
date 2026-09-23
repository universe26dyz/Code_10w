"""Controlled MLP route configuration must differ from B6 only operationally."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlp.scripts.run_training import controlled_config_without_decoder, load_controlled_config
from trad.scripts.run_training import load_training_config


ROOT = Path(__file__).resolve().parents[2]
TRAD_B6 = ROOT / "trad/configs/experiments_6k/B6.yaml"
MLP_B6 = ROOT / "mlp_surrogate_v1/CYJ_B6/reconstruction.yaml"


def test_controlled_b6_matches_trad_settings_when_decoder_and_run_metadata_are_removed():
    trad = controlled_config_without_decoder(load_training_config(TRAD_B6))
    mlp = controlled_config_without_decoder(load_training_config(MLP_B6))
    assert mlp == trad


def test_blocked_template_requires_an_explicit_checkpoint_before_it_can_be_resolved():
    with pytest.raises(ValueError, match="blocked"):
        load_controlled_config(MLP_B6, None)
    resolved = load_controlled_config(MLP_B6, "/tmp/approved_mlp.pth")
    assert resolved["decoder"]["checkpoint"] == "/tmp/approved_mlp.pth"
    assert resolved["experiment"].get("blocked") is False
