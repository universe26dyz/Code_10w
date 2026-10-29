from __future__ import annotations

from pathlib import Path


def test_copy_plan_is_subject_parameterized_and_keeps_inputs_outputs_separate(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.materialize_pipeline_archive import build_copy_plan

    root = tmp_path / "source"
    preprocessed = root / "preprocessed" / "SUBJECT_X" / "sax" / "preprocessed.mat"
    preprocessed.parent.mkdir(parents=True)
    preprocessed.write_bytes(b"mat")
    roots = {
        "shared_prepared_observations": root / "prepared" / "SUBJECT_X",
        "shared_manual_roi": root / "roi" / "SUBJECT_X",
        "shared_sparse_inputs": root / "sparse" / "SUBJECT_X",
        "fit_first_matlab_mapping": root / "mapping" / "SUBJECT_X",
        "fit_first_native_cache": root / "cache" / "SUBJECT_X_scmr_v5",
        "fit_first_reconstructions": root / "fit_first" / "SUBJECT_X",
        "proposed_reconstructions": root / "proposed" / "SUBJECT_X",
        "final_pose_evaluation": root / "final_pose" / "SUBJECT_X",
        "common_native_evaluation": root / "common_native" / "SUBJECT_X",
    }
    for path in roots.values():
        path.mkdir(parents=True)
    raw = root / "dicom" / "SUBJECT_X"; raw.mkdir(parents=True)
    deployment = root / "SUBJECT_X-deployment.json"; deployment.write_text("{}", encoding="utf-8")
    report = {
        "subject": "SUBJECT_X",
        "roots": {key: str(value) for key, value in roots.items()},
        "native_cache_manifests": {"sax": {"source_preprocessed": {"path": str(preprocessed)}}},
    }

    plan = build_copy_plan(report, raw_dicom_subject_root=raw, deployment_manifest=deployment)

    by_label = {entry.label: entry for entry in plan}
    assert by_label["raw_dicom"].destination == Path("03_subject_examples/SUBJECT_X/01_inputs/raw_dicom")
    assert by_label["preprocessed"].source == preprocessed.parents[1]
    assert by_label["prepared"].destination.parts[:4] == ("03_subject_examples", "SUBJECT_X", "01_inputs", "prepared")
    assert by_label["fit_first"].destination.parts[:4] == ("03_subject_examples", "SUBJECT_X", "02_outputs", "fit_first")
    assert by_label["proposed"].destination.parts[:4] == ("03_subject_examples", "SUBJECT_X", "02_outputs", "proposed")
