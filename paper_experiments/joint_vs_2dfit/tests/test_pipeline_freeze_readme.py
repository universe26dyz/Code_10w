"""Tests for the generic live-route freeze README generator."""
from __future__ import annotations

import json
from pathlib import Path


def test_build_freeze_readmes_writes_bilingual_subject_example_without_subject_hardcoding(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.generate_pipeline_freeze_readme import build_freeze_readmes

    freeze = tmp_path / "00_live_route_freeze"
    freeze.mkdir()
    (freeze / "live_route_freeze.json").write_text(json.dumps({
        "subject": "SUBJECT_X",
        "code": {"git_head": "a" * 40, "path": "/code"},
        "roots": {
            "shared_prepared_observations": "/prepared/SUBJECT_X",
            "shared_manual_roi": "/roi/SUBJECT_X",
            "shared_sparse_inputs": "/sparse/SUBJECT_X",
            "fit_first_matlab_mapping": "/fit/mapping/SUBJECT_X",
            "fit_first_native_cache": "/fit/native_cache/SUBJECT_X_scmr_v5",
            "fit_first_reconstructions": "/fit/reconstruction/SUBJECT_X",
            "proposed_reconstructions": "/proposed/SUBJECT_X",
            "final_pose_evaluation": "/fit/evaluation/final_pose/SUBJECT_X",
            "common_native_evaluation": "/fit/evaluation/common_native/SUBJECT_X",
        },
        "native_cache_manifests": {
            "sax": {"method": "SCMR-v5 MATLAB dictionary matching; TI=[10,100] ms",
                    "source_observations": {"sha256": "b" * 64},
                    "source_preprocessed": {"sha256": "c" * 64}},
        },
        "per_case": {"L100_fold0": {
            "proposed": {"proposed_model": {"path": "/proposed/SUBJECT_X/L100/model.pt", "sha256": "f" * 64}},
            "fit_first": {}, "evaluation": {},
        }},
    }), encoding="utf-8")
    (freeze / "proposed_semantic_verification.json").write_text(json.dumps({
        "protocol": {"path": "/code/mlp/configs/protocol_scmr_v5.yaml", "sha256": "d" * 64,
                     "inversion_times_ms": [10, 100], "tr_ms": 2.61, "vps": 87},
        "decoder_candidates": {"approved.pth": {"sha256": "e" * 64}},
        "cases": {"L100_fold0": {"status": "PASS"}},
    }), encoding="utf-8")

    zh, en = build_freeze_readmes(freeze)

    assert zh.name == "README_zh-CN.md" and en.name == "README_en.md"
    assert "SUBJECT_X" in zh.read_text(encoding="utf-8")
    assert "TI=[10,100]" in zh.read_text(encoding="utf-8")
    assert "/proposed/SUBJECT_X/L100/model.pt" in zh.read_text(encoding="utf-8")
    assert zh.read_text(encoding="utf-8").index("## 操作步骤") < zh.read_text(encoding="utf-8").index("## 冻结现场文件与 SHA 附录")
    assert "SUBJECT_X" in en.read_text(encoding="utf-8")
    assert "Where each stage runs" in en.read_text(encoding="utf-8")
    assert "HHZ" not in zh.read_text(encoding="utf-8")
