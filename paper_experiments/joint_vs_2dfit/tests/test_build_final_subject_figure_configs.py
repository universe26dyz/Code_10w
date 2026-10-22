"""Tests for deterministic, subject-general formal figure config generation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_builder_writes_every_formal_case_with_all_source_native_routes(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.build_final_subject_figure_configs import build_final_subject_figure_configs

    output = build_final_subject_figure_configs(
        subject="DYL_2", formal_grid_root=tmp_path / "formal", sparse_root=tmp_path / "sparse",
        r0_native_maps_root=tmp_path / "r0", native_mask_root=tmp_path / "masks",
        method_roots={"Fit-first": tmp_path / "fit", "Q002S640": tmp_path / "q002", "Q004S640": tmp_path / "q004"},
        output=tmp_path / "configs",
    )
    paths = sorted(output.rglob("*.json"))
    assert len(paths) == 14
    config = json.loads((output / "L50_fold1" / "psf128.json").read_text(encoding="utf-8"))
    assert config["subject"] == "DYL_2"
    assert config["native"]["mode"] == "all_source_groups"
    assert config["native"]["stacks"] == ["sax", "2ch", "4ch"]
    assert config["longitudinal"]["common_support_contract"].startswith("frozen_L100_display_mask")
    l100 = json.loads((output / "L100_fold0" / "central.json").read_text(encoding="utf-8"))
    assert "fit_first_native_bundle" in l100["methods"]["Fit-first"]
    assert "native_bundle" in l100["methods"]["Q002S640"]
    with pytest.raises(FileExistsError):
        build_final_subject_figure_configs(subject="DYL_2", formal_grid_root=tmp_path / "formal", sparse_root=tmp_path / "sparse", r0_native_maps_root=tmp_path / "r0", native_mask_root=tmp_path / "masks", method_roots={"Fit-first": tmp_path / "fit", "Q002S640": tmp_path / "q002", "Q004S640": tmp_path / "q004"}, output=output)


def test_builder_accepts_explicit_custom_cases_without_legacy_level_inference(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.build_final_subject_figure_configs import build_final_subject_figure_configs

    output = build_final_subject_figure_configs(subject="CYJ", formal_grid_root=tmp_path / "formal", sparse_root=tmp_path / "sparse", r0_native_maps_root=tmp_path / "r0", native_mask_root=tmp_path / "masks", method_roots={"Fit-first": tmp_path / "fit", "Q002S640": tmp_path / "q002", "Q004S640": tmp_path / "q004"}, custom_cases=[{"profile_id": "CYJ_SAX100_2CH0_4CH0", "case_id": "CYJ_SAX100_2CH0_4CH0"}], output=tmp_path / "configs")
    config = json.loads((output / "CYJ_SAX100_2CH0_4CH0" / "central.json").read_text(encoding="utf-8"))
    assert config["profile_id"] == "CYJ_SAX100_2CH0_4CH0"
    assert config["case_id"] == "CYJ_SAX100_2CH0_4CH0"


def test_cli_accepts_seven_explicit_custom_cases_and_writes_both_endpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import build_final_subject_figure_configs as builder

    cases = [f"P{index}=P{index}_fold0" for index in range(7)]
    monkeypatch.setattr("sys.argv", ["builder", "--subject", "CYJ", "--formal-grid-root", str(tmp_path / "formal"), "--sparse-root", str(tmp_path / "sparse"), "--r0-native-maps-root", str(tmp_path / "r0"), "--native-mask-root", str(tmp_path / "masks"), "--method-root", f"Fit-first={tmp_path / 'fit'}", "--method-root", f"Q002S640={tmp_path / 'q002'}", "--method-root", f"Q004S640={tmp_path / 'q004'}", "--output", str(tmp_path / "configs"), *sum((["--custom-case", item] for item in cases), [])])
    builder.main()
    assert len(list((tmp_path / "configs").rglob("*.json"))) == 14
