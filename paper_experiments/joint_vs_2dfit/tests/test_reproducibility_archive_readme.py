from __future__ import annotations

import json


def test_writes_bilingual_archive_readmes_after_materialization(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.generate_reproducibility_archive_readme import build_archive_readmes

    archive = tmp_path / "01_reproducible_pipeline"
    integrity = archive / "99_integrity"; integrity.mkdir(parents=True)
    for required in ("00_live_route_freeze", "01_code_snapshot", "03_subject_examples"):
        (archive / required).mkdir()
    (integrity / "archive_manifest.json").write_text(json.dumps({
        "schema_version": "scmr_v5_multimap_reproducibility_archive/v1",
        "subject_examples": ["SUBJECT_X"],
        "code_snapshot": {"commit": "a" * 40},
        "file_count": 12,
    }), encoding="utf-8")

    zh, en = build_archive_readmes(archive)

    assert "从原始 DICOM 开始" in zh.read_text(encoding="utf-8")
    assert "03_subject_examples/SUBJECT_X/01_inputs/prepared" in zh.read_text(encoding="utf-8")
    assert "generic" in en.read_text(encoding="utf-8")
    assert "01_code_snapshot/source.tar" in en.read_text(encoding="utf-8")
    assert "SERVER_DEPLOYMENT_MANIFEST" in en.read_text(encoding="utf-8")


def test_replace_only_overwrites_existing_readmes(tmp_path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.generate_reproducibility_archive_readme import build_archive_readmes

    archive = tmp_path / "01_reproducible_pipeline"
    for required in ("00_live_route_freeze", "01_code_snapshot", "03_subject_examples", "99_integrity"):
        (archive / required).mkdir(parents=True, exist_ok=True)
    (archive / "99_integrity" / "archive_manifest.json").write_text(json.dumps({
        "schema_version": "scmr_v5_multimap_reproducibility_archive/v1",
        "subject_examples": ["SUBJECT_X"],
        "code_snapshot": {"commit": "a" * 40},
        "file_count": 12,
    }), encoding="utf-8")
    (archive / "README_zh-CN.md").write_text("old", encoding="utf-8")
    (archive / "README_en.md").write_text("old", encoding="utf-8")

    zh, en = build_archive_readmes(archive, replace_existing=True)

    assert zh.read_text(encoding="utf-8").startswith("# SCMR-v5")
    assert en.read_text(encoding="utf-8").startswith("# SCMR-v5")
