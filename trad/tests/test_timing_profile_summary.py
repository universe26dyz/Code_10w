import csv
import json

from evaluation.profiling.summarize_timing_profile import summarize_timing_profile


def test_summary_reports_stage_section_statistics_and_estimate(tmp_path):
    profile = tmp_path / "timing_profile.csv"
    with profile.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("iteration", "stage", "section", "milliseconds"))
        writer.writeheader()
        writer.writerows(
            [
                {"iteration": 1, "stage": "A", "section": "whole_iteration", "milliseconds": 10},
                {"iteration": 1, "stage": "A", "section": "decoder", "milliseconds": 4},
                {"iteration": 2, "stage": "A", "section": "whole_iteration", "milliseconds": 20},
                {"iteration": 2, "stage": "A", "section": "decoder", "milliseconds": 8},
            ]
        )
    result = summarize_timing_profile(profile, stage_iterations={"A": 2000})
    rows = {row["section"]: row for row in result["rows"]}
    assert rows["decoder"]["N"] == 2
    assert rows["decoder"]["median_ms"] == 6.0
    assert rows["decoder"]["median_fraction_of_whole_iteration"] == 0.4
    assert result["estimated_total_training_seconds"] == 30.0
    assert (tmp_path / "timing_profile_summary.csv").is_file()
    assert json.loads((tmp_path / "timing_profile_summary.json").read_text())["estimated_total_training_seconds"] == 30.0
