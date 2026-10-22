"""Read-only status wrappers must validate their supplied manifests."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "phase2_clean" / "server_commands"


def test_profile_status_script_rejects_invalid_manifest_and_reads_valid_custom_case(tmp_path: Path) -> None:
    script = SCRIPTS / "status_cyj_experiment_1_1_v2.sh"
    invalid = tmp_path / "invalid.json"; invalid.write_text("{}", encoding="utf-8")
    assert subprocess.run(["bash", str(script), "--dry-run", str(invalid)], text=True, capture_output=True).returncode != 0
    valid = tmp_path / "profile.json"; valid.write_text(json.dumps({"schema_version": "phase4r_sparse_profile_v2", "subject": "CYJ", "profile_id": "CYJ_S2CH50_4CH100", "case_id": "CYJ_S2CH50_4CH100_fold0", "stacks": {stack: {"training_stack_present": stack != "2ch", "source_observation": {"path": "/not-inspected", "sha256": "declared"}} for stack in ("sax", "2ch", "4ch")}}), encoding="utf-8")
    result = subprocess.run(["bash", str(script), "--dry-run", str(valid)], text=True, capture_output=True)
    assert result.returncode == 0 and "CYJ_S2CH50_4CH100_fold0" in result.stdout


def test_k4_status_script_rejects_invalid_manifest_and_reads_controls(tmp_path: Path) -> None:
    script = SCRIPTS / "status_cyj_experiment_1_2_v2.sh"
    invalid = tmp_path / "invalid.json"; invalid.write_text("{}", encoding="utf-8")
    assert subprocess.run(["bash", str(script), "--dry-run", str(invalid)], text=True, capture_output=True).returncode != 0
    valid = tmp_path / "k4.json"; valid.write_text(json.dumps({"experiment_id": "Q005S640_K4_B6_roi_same_anchor_mse", "training_psf_samples": 4, "data_psf_inr_location_count": 2560, "psf_forward_chunk_size": 4, "endpoint_semantics": {"central": "central", "psf128": "post-hoc"}}), encoding="utf-8")
    result = subprocess.run(["bash", str(script), "--dry-run", str(valid)], text=True, capture_output=True)
    assert result.returncode == 0 and "2560" in result.stdout
