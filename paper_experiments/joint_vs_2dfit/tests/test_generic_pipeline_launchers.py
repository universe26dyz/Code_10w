from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
COMMANDS = ROOT / "paper_experiments" / "joint_vs_2dfit" / "phase2_clean" / "server_commands"


def test_generic_fit_first_launcher_has_no_hhz_cache_or_delegate() -> None:
    text = (COMMANDS / "run_scmr_v5_fit_first.sh").read_text(encoding="utf-8")

    assert "${SUBJECT}_scmr_v5" in text
    assert "run_hhz_scmr_v5_fit_first.sh" not in text
    assert "HHZ_scmr_v5" not in text


def test_generic_evaluation_launchers_do_not_delegate_to_hhz_scripts() -> None:
    for name in ("paired_final_pose.sh", "common_native_metrics_qc.sh", "common_native_all_stack_roi_metrics.sh"):
        text = (COMMANDS / name).read_text(encoding="utf-8")
        assert "hhz_" not in text
        assert "SUBJECT=${SUBJECT:-HHZ}" not in text
