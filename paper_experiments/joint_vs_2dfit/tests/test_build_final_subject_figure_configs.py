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


def test_builder_writes_explicit_final5_l100_layout_without_path_discovery(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.build_final_subject_figure_configs import build_final5_subject_figure_config

    sparse = tmp_path / "sparse_manifest.json"
    sparse.write_text(json.dumps({"subject": "DYL_2", "profile_id": "L100", "case_id": "L100", "fold_id": None, "fold_count": 1, "generator_git_sha": "a" * 40, "prepared_root": str(tmp_path / "prepared")}), encoding="utf-8")
    final5 = tmp_path / "final5"
    output = build_final5_subject_figure_config(
        subject="DYL_2", case_id="L100", final5_root=final5, sparse_manifest=sparse,
        r0_native_maps_root=tmp_path / "r0", output=tmp_path / "configs",
    )
    config = json.loads((output / "DYL_2" / "L100" / "central.json").read_text(encoding="utf-8"))
    assert config["final5"]["root"] == str(final5.resolve())
    assert config["methods"]["Q002S640"]["native_bundle"] == str((final5 / "05_formal_endpoints/central/proposed_native_bundle/DYL_2/L100").resolve())
    assert config["methods"]["Fit-first"]["fit_first_native_bundle"] == str((final5 / "05_formal_endpoints/central/fit_first_native_bundle/DYL_2/L100").resolve())
    assert config["native_mask_root"] == str((final5 / "06_metrics/current_native_mask/DYL_2/L100").resolve())
    assert config["native"]["mode"] == "all_source_groups" and config["native"]["stacks"] == ["sax", "2ch", "4ch"]


def test_builder_accepts_fit_first_plus_one_proposed_method_and_future_q005(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.build_final_subject_figure_configs import build_final_subject_figure_configs

    output = build_final_subject_figure_configs(subject="CYJ", formal_grid_root=tmp_path / "formal", sparse_root=tmp_path / "sparse", r0_native_maps_root=tmp_path / "r0", native_mask_root=tmp_path / "masks", method_roots={"Fit-first": tmp_path / "fit", "Q005S640_K4": tmp_path / "q005"}, output=tmp_path / "configs")
    config = json.loads((output / "L100_fold0" / "central.json").read_text(encoding="utf-8"))
    assert list(config["methods"]) == ["Fit-first", "Q005S640_K4"]
    assert config["endpoint"] == "central"
    assert json.loads((output / "L100_fold0" / "psf128.json").read_text(encoding="utf-8"))["endpoint"] == "psf128"
    with pytest.raises(ValueError, match="unsupported"):
        build_final_subject_figure_configs(subject="CYJ", formal_grid_root=tmp_path / "formal2", sparse_root=tmp_path / "sparse2", r0_native_maps_root=tmp_path / "r02", native_mask_root=tmp_path / "masks2", method_roots={"Fit-first": tmp_path / "fit2", "SomeRandomMethod": tmp_path / "bad"}, output=tmp_path / "invalid")


def test_cli_accepts_seven_explicit_custom_cases_and_writes_both_endpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import build_final_subject_figure_configs as builder

    cases = [f"P{index}=P{index}_fold0" for index in range(7)]
    monkeypatch.setattr("sys.argv", ["builder", "--subject", "CYJ", "--formal-grid-root", str(tmp_path / "formal"), "--sparse-root", str(tmp_path / "sparse"), "--r0-native-maps-root", str(tmp_path / "r0"), "--native-mask-root", str(tmp_path / "masks"), "--method-root", f"Fit-first={tmp_path / 'fit'}", "--method-root", f"Q002S640={tmp_path / 'q002'}", "--method-root", f"Q004S640={tmp_path / 'q004'}", "--output", str(tmp_path / "configs"), *sum((["--custom-case", item] for item in cases), [])])
    builder.main()
    assert len(list((tmp_path / "configs").rglob("*.json"))) == 14
