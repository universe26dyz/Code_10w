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


def test_server_only_preprocessing_launcher_uses_one_server_dicom_manifest() -> None:
    text = (COMMANDS / "server_preprocess_and_prepare.sh").read_text(encoding="utf-8")

    assert "RUN_ROOT SUBJECT DEPLOYMENT_MANIFEST PROTOCOL" in text
    assert "local_preprocess_all" in text
    assert "validate_transferred_preprocessed.py" in text
    assert "prepare_all_observations.py" in text
    assert "local_dicom_dir and server_dicom_dir" in text
    assert "HHZ" not in text


def test_server_sparse_profile_launcher_builds_all_standard_cases_without_subject_names() -> None:
    text = (COMMANDS / "build_sparse_profiles.sh").read_text(encoding="utf-8")

    assert "PREPARED_ROOT SUBJECT OUTPUT_ROOT" in text
    assert "for CONDITION in L100 L25 L50" in text
    assert '--condition "$CONDITION" --phase 0' in text
    assert "SAX100_2CH0_4CH0 SAX100_2CH0_4CH100 SAX100_2CH100_4CH0 S2CH100_4CH50 S2CH50_4CH100" in text
    assert '"$PROFILE_ROOT/$PROFILE.json"' in text
    assert "CYJ_" not in text
    assert "HHZ" not in text
