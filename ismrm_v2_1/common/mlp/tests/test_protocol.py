from pathlib import Path

import yaml


def test_hhz_v1_protocol_is_fixed_and_records_dyz_difference():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "protocol_hhz_v1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["num_images"] == 10
    assert config["flip_angle_degrees"] == [45, 45, 45]
    assert config["inversion_times_ms"] == [50, 150]
    assert config["t2prep_ms"] == [35, 45, 55]
    assert config["ramp_up_pulses"] == 10
    assert config["known_difference"]["dyz_legacy_inversion_times_ms"] == [10, 100]
