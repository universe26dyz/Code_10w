from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.deployment_paths import resolve_data_paths


def test_data_roots_resolve_from_one_environment_specific_root(tmp_path):
    config = tmp_path / "paths.yaml"
    config.write_text(yaml.safe_dump({"data": {"local_data_root": "/local/Code_10w_v1", "server_data_root": "/server/Code_10w_v1", "derived": {"prepared_root": "Code_10w_prepared", "runs_root": "Code_10w_runs", "runtime_root": "Code_10w_runtime", "preprocessed_root": "Code_10w_preprocessed"}}}), encoding="utf-8")
    assert resolve_data_paths(config, "local")["prepared_root"] == Path("/local/Code_10w_v1/Code_10w_prepared")
    assert resolve_data_paths(config, "server")["runs_root"] == Path("/server/Code_10w_v1/Code_10w_runs")


def test_data_roots_reject_an_absolute_derived_path(tmp_path):
    config = tmp_path / "paths.yaml"
    config.write_text(yaml.safe_dump({"data": {"local_data_root": "/local", "server_data_root": "/server", "derived": {"prepared_root": "/wrong"}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="relative"):
        resolve_data_paths(config, "local")
