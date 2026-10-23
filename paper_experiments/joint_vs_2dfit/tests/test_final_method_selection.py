from __future__ import annotations

import json
from pathlib import Path

import pytest


def _evidence(tmp_path: Path) -> dict[str, Path]:
    return {name: (tmp_path / f"{name}.json") for name in ("experiment_1_1", "experiment_1_2", "metrics")}


@pytest.mark.parametrize(("method", "samples"), (("Q002S640", 8), ("Q005S640_K4", 4)))
def test_final_method_selection_requires_explicit_evidence_and_method_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, samples: int) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import final_method_selection as selection

    evidence = _evidence(tmp_path)
    for path in evidence.values(): path.write_text("{}", encoding="utf-8")
    protocol = tmp_path / "protocol.yaml"; protocol.write_text("TR: 2.61\n", encoding="utf-8")
    monkeypatch.setattr(selection, "B6_CONFIG_SOURCE_SHA256", "b" * 64)
    monkeypatch.setattr(selection, "APPROVED_DECODER_SHA256", "d" * 64)
    profile = tmp_path / "profile.json"; profile.write_text('{"schema":"phase4r_sparse_profile/v2","profile_id":"SAX100_2CH0_4CH0","stacks":{}}', encoding="utf-8")
    manifest = selection.create_final_method_selection(output=tmp_path / "selection.json", selected_profile={"profile_id": "SAX100_2CH0_4CH0", "profile_path": str(profile), "profile_sha256": selection.sha256(profile)}, selected_proposed_method=method, profile_replication_policy={"mode": "one_preregistered_fold", "fold_id": 0}, experiment_1_1_evidence={"selected_case": evidence["experiment_1_1"]}, experiment_1_2_evidence={"selected_method": evidence["experiment_1_2"]}, metric_figure_summary_evidence={"selection_metrics": evidence["metrics"]}, protocol=protocol)
    payload = selection.validate_final_method_selection(manifest)
    assert payload["training_psf_samples"] == samples
    broken = json.loads(manifest.read_text()); broken["experiment_1_1_evidence"]["selected_case"]["sha256"] = "0" * 64; manifest.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        selection.validate_final_method_selection(manifest)
