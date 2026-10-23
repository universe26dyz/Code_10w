from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest


def test_method_selection_uses_three_method_common_support_and_keeps_endpoints_separate(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.final_method_selection_metrics import evaluate_method_selection_common_support
    output = evaluate_method_selection_common_support(reference=np.array([1., 2., 3.]), q002=np.array([1., 3., 3.]), q005=np.array([2., 2., np.nan]), fit_first=np.array([1., 2., 4.]), primary_native_mask=np.ones(3, bool), q002_endpoint_valid=np.ones(3, bool), q005_endpoint_valid=np.ones(3, bool), fit_first_endpoint_valid=np.ones(3, bool), endpoint="central", parameter="t1_ms", output=tmp_path / "metrics.json")
    payload = json.loads(output.read_text())
    assert payload["support_voxel_count"] == 2 and set(payload["methods"]) == {"Q002S640", "Q005S640_K4", "Fit-first"}


def test_population_aggregator_uses_subjects_and_rejects_support_mismatch(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.aggregate_final_multi_subject import aggregate_final_multi_subject, SUBJECTS
    paths = []
    for subject in SUBJECTS:
        path = tmp_path / f"{subject}.json"; path.write_text(json.dumps({"subject": subject, "paired_subject_metrics": [{"endpoint": "central", "parameter": "t1_ms", "metric": "rmse", "region": "all", "proposed": 1.0, "fit_first": 2.0, "support_identity": "same", "paired_support_status": "AUDITED_IDENTICAL"}]})); paths.append(path)
    output = aggregate_final_multi_subject(subject_summaries=paths, output=tmp_path / "cohort")
    assert json.loads((output / "cohort_summary.json").read_text())["groups"][0]["subject_count"] == 5
    broken = json.loads(paths[0].read_text()); broken["paired_subject_metrics"][0]["paired_support_status"] = "DIFFERENT"; paths[0].write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="PAIRED_SUPPORT_MISMATCH"):
        aggregate_final_multi_subject(subject_summaries=paths, output=tmp_path / "broken")
