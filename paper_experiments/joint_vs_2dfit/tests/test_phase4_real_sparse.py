from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from paper_experiments.joint_vs_2dfit.phase2_clean.aggregate_real import (
    bland_altman, load_r0_paired_records, paired_r0_comparison, pooled_r1_fold_metrics, summarize_r1_subjects,
)
from paper_experiments.joint_vs_2dfit.phase2_clean.evaluation import heldout_quantitative_metrics, paired_per_plane_records, real_native_agreement, real_native_paired_agreement
from paper_experiments.joint_vs_2dfit.phase2_clean.sparse import build_sparse_condition, build_sparse_profile_condition
from paper_experiments.joint_vs_2dfit.phase2_clean import proposed
from paper_experiments.joint_vs_2dfit.phase2_clean.contracts import APPROVED_DECODER_SHA256
from paper_experiments.joint_vs_2dfit.phase2_clean import fit_first
from paper_experiments.joint_vs_2dfit.phase2_clean import heldout
from paper_experiments.joint_vs_2dfit.phase2_clean.fit_first import build_independent_nesvor_commands, reuse_native_maps, select_verified_native_cache_groups
from paper_experiments.joint_vs_2dfit.phase2_clean.migrate_native_fit_cache import migrate_native_fit_cache


def _observations(path: Path, group_ids: tuple[int, ...], *, missing_weight: bool = False) -> None:
    rows = [(group, weight) for group in group_ids for weight in range(10)]
    if missing_weight:
        rows.pop()
    count = len(rows)
    groups = np.asarray([row[0] for row in rows], dtype=np.int64)
    weights = np.asarray([row[1] for row in rows], dtype=np.int64)
    affine = np.repeat(np.eye(4)[None], count, axis=0)
    for index, group in enumerate(groups):
        affine[index, 2, 3] = float(group * 5)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        images=np.ones((count, 3, 4), np.float32), masks=np.ones((count, 3, 4), bool),
        group_idx=groups, weight_idx=weights, stack_idx=np.zeros(count, np.int64),
        acquisition_time_ms=np.arange(count, dtype=float), timing9_ms=np.tile(np.arange(1.0, 10.0), (count, 1)),
        affine_lps_rc=affine, pixel_spacing_rc_mm=np.ones((count, 2), float),
        slice_thickness_mm=np.full(count, 8.0), tr_ms=np.full(count, 2.61), vps=np.full(count, 87, np.int64),
    )


def _prepared(root: Path, *, missing_weight: bool = False) -> Path:
    for stack in ("sax", "2ch", "4ch"):
        _observations(root / "CYJ" / stack / "observations.npz", (4, 5, 6, 7), missing_weight=missing_weight and stack == "2ch")
    return root


def test_sparse_builder_keeps_atomic_groups_and_original_provenance(tmp_path: Path) -> None:
    source = _prepared(tmp_path / "prepared")
    result = build_sparse_condition(source, "CYJ", "L50", 0, tmp_path / "runs", git_sha="abc", audit_stack=lambda _: {"regular_gap": True, "gap_mm": 10.0, "group_count": 2})
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert manifest["status"] == "ELIGIBLE"
    assert manifest["stacks"]["2ch"]["retained_original_group_ids"] == [4, 6]
    assert manifest["stacks"]["2ch"]["heldout_original_group_ids"] == [5, 7]
    with np.load(result["prepared_root"] / "2ch" / "observations.npz") as data:
        assert np.unique(data["group_idx"]).tolist() == [0, 1]
        for group in (0, 1):
            assert np.sort(data["weight_idx"][data["group_idx"] == group]).tolist() == list(range(10))
    assert manifest["leakage_gate"] is True
    assert manifest["preprocessing_leakage_statement"].startswith("MIND operates")


def test_sparse_builder_rejects_incomplete_fingerprint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="weights 0..9"):
        build_sparse_condition(_prepared(tmp_path / "prepared", missing_weight=True), "CYJ", "L50", 0, tmp_path / "runs")


def test_sparse_builder_marks_too_few_retained_groups_ineligible(tmp_path: Path) -> None:
    source = _prepared(tmp_path / "prepared")
    result = build_sparse_condition(source, "CYJ", "L25", 3, tmp_path / "runs")
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert manifest["status"] == "INELIGIBLE"
    assert "fewer than two retained" in manifest["reason"]


def test_l100_manifest_reuses_original_observations_without_copying_them(tmp_path: Path) -> None:
    source = _prepared(tmp_path / "prepared")
    result = build_sparse_condition(source, "CYJ", "L100", 0, tmp_path / "runs", git_sha="abc")
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert manifest["identity_source"] == "R0_full_prepared_observations"
    assert Path(result["prepared_root"]) == source / "CYJ"
    assert not (tmp_path / "runs" / "CYJ" / "L100_fold0" / "2ch" / "observations.npz").exists()


def test_heldout_metrics_use_common_method_neutral_support() -> None:
    reference = np.array([[1000.0, 1100.0], [np.nan, 1200.0]])
    proposed = np.array([[1010.0, 1110.0], [999.0, np.nan]])
    baseline = np.array([[990.0, 1090.0], [999.0, 1210.0]])
    source_support = np.array([[True, True], [True, True]])
    reprojection = np.array([[True, True], [True, True]])
    report = heldout_quantitative_metrics(reference, proposed, baseline, source_support, reprojection)
    assert report["common_support_voxels"] == 2
    assert report["proposed"]["rmse"] == pytest.approx(10.0)
    assert report["fit_first"]["rmse"] == pytest.approx(10.0)
    shifted = heldout_quantitative_metrics(reference, proposed + 25.0, baseline, source_support, reprojection)
    assert shifted["common_support_voxels"] == report["common_support_voxels"]


def test_r0_metrics_means_bland_altman_and_subject_aggregation() -> None:
    reference = np.array([[1000.0, 1100.0]])
    prediction = np.array([[1010.0, 1110.0]])
    metric = real_native_agreement(reference, prediction, np.ones_like(reference, bool))
    assert metric["reference_mean_ms"] == 1050.0
    assert metric["prediction_mean_ms"] == 1060.0
    assert metric["mean_difference_ms"] == 10.0
    ba = bland_altman([1000.0, 1100.0, 1200.0, 1300.0], [1010.0, 1090.0, 1210.0, 1290.0])
    assert ba["n_subjects"] == 4 and ba["mean_difference_ms"] == 0.0
    summary = summarize_r1_subjects([
        {"subject": "CYJ", "method": "proposed", "level": "L50", "parameter": "t1_ms", "phase": 0, "pooled_rmse": 10.0},
        {"subject": "CYJ", "method": "proposed", "level": "L50", "parameter": "t1_ms", "phase": 1, "pooled_rmse": 14.0},
    ])
    assert summary[0]["rmse"] == 12.0 and summary[0]["fold_count"] == 2


def test_r0_paired_evaluation_uses_identical_common_myocardium_support() -> None:
    reference = np.array([[1000.0, 1100.0], [1200.0, 1300.0]])
    proposed = reference + 10.0
    baseline = reference - 10.0
    myocardium = np.ones_like(reference, bool)
    proposed_valid = np.array([[True, True], [True, False]])
    baseline_valid = np.array([[True, True], [False, True]])
    report = real_native_paired_agreement(reference, proposed, baseline, myocardium, proposed_valid, baseline_valid)
    assert report["common_support_voxels"] == 2
    assert report["proposed"]["support_voxels"] == report["fit_first"]["support_voxels"] == 2
    assert report["proposed"]["reference_mean_ms"] == report["fit_first"]["reference_mean_ms"] == 1050.0


def test_r0_zero_common_support_plane_is_retained_without_metric_key_error() -> None:
    rows = [
        {"group_index": 0, "proposed": {"rmse": 1.0}, "fit_first": {"rmse": 2.0}},
        {"group_index": 1, "status": "no_common_myocardium_support", "support_voxels": 0, "coverage_fraction": 0.0},
    ]
    proposed = paired_per_plane_records(rows, "proposed")
    baseline = paired_per_plane_records(rows, "fit_first")
    assert proposed[1]["status"] == baseline[1]["status"] == "no_common_myocardium_support"
    assert proposed[1]["support_voxels"] == baseline[1]["support_voxels"] == 0


def test_r0_aggregation_rejects_legacy_individual_method_json(tmp_path: Path) -> None:
    (tmp_path / "CYJ_proposed.json").write_text("{}")
    with pytest.raises(ValueError, match="paired common-support"):
        load_r0_paired_records(tmp_path)


def test_proposed_command_uses_current_adapter_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proposed, "sha256", lambda _: APPROVED_DECODER_SHA256)
    command = proposed.build_proposed_command("/prepared", "/decoder.pth", "/b6.pt", "/output")
    assert shlex.split(command)[0] == sys.executable


def test_fit_first_commands_disable_default_700_rescaling_and_save_models() -> None:
    commands = build_independent_nesvor_commands("/inputs", "/output", nesvor_executable="/historical/nesvor")
    assert len(commands) == 2
    for parameter, scale, command in zip(("t1", "t2"), ("2500", "200"), commands):
        assert "--output-intensity-mean 0" in command
        assert f"--output-model /output/{parameter}/model.pt" in command
        assert "--output-intensity-mean 700" not in command
        assert "--sample-mask" not in command
        assert f"--scale {scale}" in command
        assert "--registration stack" in command and "--n-iter 10000" in command


def test_fit_first_explicit_output_sample_masks_fail_closed_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="t1: explicit sample mask is missing"):
        build_independent_nesvor_commands(
            "/inputs",
            tmp_path / "output",
            sample_masks={"t1": tmp_path / "missing_t1.nii.gz", "t2": tmp_path / "missing_t2.nii.gz"},
        )


def _fake_dictionary(*_args: object) -> dict[str, np.ndarray]:
    return {
        "fingerprints": np.ones((1, 10), dtype=float),
        "t1_ms": np.asarray([1111.0]),
        "t2_ms": np.asarray([55.0]),
        "b1": np.asarray([0.9]),
    }


def _full_native_maps(source: Path, destination: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(fit_first, "frozen_dictionary", _fake_dictionary)
    for stack in ("sax", "2ch", "4ch"):
        fit_first.fit_native(source / "CYJ" / stack / "observations.npz", destination / stack)
    return destination


def test_r1_reuses_exact_retained_native_groups_without_heldout_leakage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _prepared(tmp_path / "prepared")
    sparse = build_sparse_condition(source, "CYJ", "L50", 0, tmp_path / "sparse", git_sha="abc", audit_stack=lambda _: {"regular_gap": True, "gap_mm": 10.0, "group_count": 2})
    full = _full_native_maps(source, tmp_path / "r0_native", monkeypatch)
    reused = reuse_native_maps(full, sparse["manifest_path"], tmp_path / "r1_native")
    direct = fit_first.fit_native(Path(sparse["prepared_root"]) / "2ch" / "observations.npz", tmp_path / "direct")
    with np.load(reused / "2ch" / "native_maps.npz") as actual, np.load(direct / "native_maps.npz") as expected:
        assert actual["group_idx"].tolist() == [4, 6]
        for name in ("t1_ms", "t2_ms", "b1"):
            assert np.array_equal(actual[name], expected[name], equal_nan=True)
    with np.load(reused / "sax" / "native_maps.npz") as sax:
        assert sax["group_idx"].tolist() == [4, 5, 6, 7]
    provenance = json.loads((reused / "reuse_manifest.json").read_text())
    assert provenance["native_fit_reuse_is_group_local"] is True
    assert provenance["stacks"]["2ch"]["retained_original_group_ids"] == [4, 6]
    assert provenance["stacks"]["2ch"]["heldout_original_group_ids"] == [5, 7]
    assert provenance["stacks"]["2ch"]["nesvor_input_original_group_ids"] == [4, 6]


def test_r1_custom_profile_reorders_complete_sax_cache_to_physical_manifest_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _prepared(tmp_path / "prepared")
    sax_source = source / "CYJ" / "sax" / "observations.npz"
    with np.load(sax_source) as original:
        observations = {name: original[name].copy() for name in original.files}
    physical_order = [10, 3, 40, 99]
    numeric_order = [3, 10, 40, 99]
    observations["group_idx"] = np.repeat(np.asarray(numeric_order, dtype=np.int64), 10)
    for ordinal, group in enumerate(physical_order):
        observations["affine_lps_rc"][observations["group_idx"] == group, 2, 3] = float(ordinal * 5)
    np.savez_compressed(sax_source, **observations)
    profile = tmp_path / "SAX100_2CH100_4CH100.json"
    profile.write_text(json.dumps({"schema": "phase4r_sparse_profile/v2", "profile_id": "SAX100_2CH100_4CH100", "stacks": {stack: {"mode": "all"} for stack in ("sax", "2ch", "4ch")}}))
    sparse = build_sparse_profile_condition(source, "CYJ", profile, None, tmp_path / "sparse", git_sha="a" * 40, audit_stack=lambda _path: {"regular_gap": True, "gap_mm": 5.0, "group_count": 4})
    full = _full_native_maps(source, tmp_path / "r0_native", monkeypatch)
    with np.load(full / "sax" / "native_maps.npz") as original:
        maps = {name: original[name].copy() for name in original.files}
    assert maps["group_idx"].tolist() == numeric_order
    for index, group in enumerate(numeric_order):
        maps["t1_ms"][index].fill(group)
        maps["t2_ms"][index].fill(group + 100)
        maps["b1"][index].fill(group + 0.5)
    np.savez_compressed(full / "sax" / "native_maps.npz", **maps)
    cache_manifest = json.loads((full / "sax" / "manifest.json").read_text())
    cache_manifest["native_maps_sha256"] = fit_first.sha256(full / "sax" / "native_maps.npz")
    (full / "sax" / "manifest.json").write_text(json.dumps(cache_manifest))

    reused = reuse_native_maps(full, sparse["manifest_path"], tmp_path / "r1_native")

    with np.load(reused / "sax" / "native_maps.npz") as maps, np.load(reused / "sax" / "geometry.npz") as geometry:
        assert maps["group_idx"].tolist() == physical_order
        for index, group in enumerate(physical_order):
            assert np.all(maps["t1_ms"][index] == group)
            assert np.all(maps["t2_ms"][index] == group + 100)
            assert np.all(maps["b1"][index] == group + 0.5)
            assert geometry["affine_lps_rc"][index, 2, 3] == index * 5
            assert np.all(geometry["pixel_spacing_rc_mm"][index] == 1)
            assert geometry["slice_thickness_mm"][index] == 8
    provenance = json.loads((reused / "reuse_manifest.json").read_text())
    sax = provenance["stacks"]["sax"]
    assert sax["cached_original_group_ids"] == numeric_order
    assert sax["retained_original_group_ids"] == sax["output_original_group_ids"] == physical_order
    assert sax["output_order_semantics"] == "manifest_retained_original_group_ids_physical_order"


@pytest.mark.parametrize(
    ("retained", "heldout"),
    (([4, 5, 6], []), ([4, 5, 6, 7], [7])),
)
def test_r1_sax_reuse_rejects_incomplete_or_heldout_groups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retained: list[int], heldout: list[int]) -> None:
    source = _prepared(tmp_path / "prepared")
    sparse = build_sparse_condition(source, "CYJ", "L50", 0, tmp_path / "sparse", git_sha="abc", audit_stack=lambda _path: {"regular_gap": True, "gap_mm": 10.0, "group_count": 2})
    manifest = json.loads(Path(sparse["manifest_path"]).read_text())
    manifest["stacks"]["sax"]["retained_original_group_ids"] = retained
    manifest["stacks"]["sax"]["heldout_original_group_ids"] = heldout
    invalid = tmp_path / "invalid_sax.json"
    invalid.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="sax: R1 reuse requires every full-data SAX group and no held-out SAX group"):
        reuse_native_maps(_full_native_maps(source, tmp_path / "r0_native", monkeypatch), invalid, tmp_path / "r1_native")


def test_r1_native_reuse_rejects_hash_ineligible_and_duplicate_group_gates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _prepared(tmp_path / "prepared")
    sparse = build_sparse_condition(source, "CYJ", "L50", 0, tmp_path / "sparse", git_sha="abc", audit_stack=lambda _: {"regular_gap": True, "gap_mm": 10.0, "group_count": 2})
    full = _full_native_maps(source, tmp_path / "r0_native", monkeypatch)
    manifest_path = Path(sparse["manifest_path"])
    manifest = json.loads(manifest_path.read_text())
    manifest["stacks"]["2ch"]["source_observation"]["sha256"] = "wrong"
    bad_hash = tmp_path / "bad_hash.json"; bad_hash.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="source observation SHA256 mismatch"):
        reuse_native_maps(full, bad_hash, tmp_path / "bad_hash")
    assert not (tmp_path / "bad_hash").exists()
    assert not list(tmp_path.glob(".bad_hash.staging-*"))
    manifest["status"] = "INELIGIBLE"
    ineligible = tmp_path / "ineligible.json"; ineligible.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="ELIGIBLE"):
        reuse_native_maps(full, ineligible, tmp_path / "ineligible")
    with np.load(full / "2ch" / "native_maps.npz") as maps:
        np.savez_compressed(full / "2ch" / "replacement.npz", t1_ms=maps["t1_ms"], t2_ms=maps["t2_ms"], b1=maps["b1"], group_idx=np.asarray([4, 4, 6, 7]))
    (full / "2ch" / "native_maps.npz").unlink()
    (full / "2ch" / "replacement.npz").rename(full / "2ch" / "native_maps.npz")
    cached = json.loads((full / "2ch" / "manifest.json").read_text())
    cached["native_maps_sha256"] = fit_first.sha256(full / "2ch" / "native_maps.npz")
    (full / "2ch" / "manifest.json").write_text(json.dumps(cached))
    with pytest.raises(ValueError, match="exactly once"):
        reuse_native_maps(full, manifest_path, tmp_path / "duplicate")
    assert not (tmp_path / "duplicate").exists()
    assert not list(tmp_path.glob(".duplicate.staging-*"))


def _legacy_native_maps(source: Path, destination: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    full = _full_native_maps(source, destination.with_name(destination.name + "_source"), monkeypatch)
    shutil.copytree(full, destination)
    for stack in ("sax", "2ch", "4ch"):
        (destination / stack / "manifest.json").unlink()
    (destination.parent / "input_manifest.json").write_text(json.dumps({
        "method": "fit_first", "subject_id": "CYJ",
        "dictionary": "frozen current HHZ/TRAD grid; normalized cosine; no interpolation/refinement",
        "observations": {stack: fit_first.sha256(source / "CYJ" / stack / "observations.npz") for stack in ("sax", "2ch", "4ch")},
    }))
    return destination


def _update_historical_observation_hash(legacy: Path, source: Path, stack: str) -> None:
    path = legacy.parent / "input_manifest.json"
    payload = json.loads(path.read_text())
    payload["observations"][stack] = fit_first.sha256(source / "CYJ" / stack / "observations.npz")
    path.write_text(json.dumps(payload))


def test_native_cache_migration_copies_without_refitting_and_is_reusable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _prepared(tmp_path / "prepared")
    legacy = _legacy_native_maps(source, tmp_path / "legacy", monkeypatch)
    monkeypatch.setattr(fit_first, "fit_native", lambda *_args, **_kwargs: pytest.fail("migration must not refit"))
    migrated = migrate_native_fit_cache(legacy, source / "CYJ", tmp_path / "migrated")
    for stack in ("sax", "2ch", "4ch"):
        for name in ("native_maps.npz", "geometry.npz"):
            assert fit_first.sha256(legacy / stack / name) == fit_first.sha256(migrated / stack / name)
        manifest = json.loads((migrated / stack / "manifest.json").read_text())
        assert manifest["numerical_fitting_rerun"] is False
        assert manifest["migration_source_path"] == str((legacy / stack).resolve())
        assert manifest["historical_input_manifest_sha256"] == fit_first.sha256(legacy.parent / "input_manifest.json")
        assert manifest["historical_method"] == "fit_first"
        assert manifest["historical_dictionary_declaration"] == "frozen current HHZ/TRAD grid; normalized cosine; no interpolation/refinement"
    sparse = build_sparse_condition(source, "CYJ", "L50", 0, tmp_path / "sparse", git_sha="abc", audit_stack=lambda _: {"regular_gap": True, "gap_mm": 10.0, "group_count": 2})
    assert reuse_native_maps(migrated, sparse["manifest_path"], tmp_path / "reused").is_dir()


def test_native_cache_migration_rejects_invalid_source_and_leaves_no_partial_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _prepared(tmp_path / "prepared")
    legacy = _legacy_native_maps(source, tmp_path / "legacy", monkeypatch)
    with np.load(source / "CYJ" / "2ch" / "observations.npz") as original:
        data = {name: original[name] for name in original.files}
    data["tr_ms"] = np.full_like(data["tr_ms"], 2.5)
    np.savez_compressed(source / "CYJ" / "2ch" / "observations.npz", **data)
    _update_historical_observation_hash(legacy, source, "2ch")
    destination = tmp_path / "migrated"
    with pytest.raises(ValueError, match="frozen timing/TR/VPS"):
        migrate_native_fit_cache(legacy, source / "CYJ", destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".migrated.staging-*"))
    destination.mkdir()
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        migrate_native_fit_cache(legacy, source / "CYJ", destination)


def test_native_cache_migration_rejects_group_and_vps_mismatches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    group_source = _prepared(tmp_path / "group_prepared")
    group_legacy = _legacy_native_maps(group_source, tmp_path / "group_legacy", monkeypatch)
    with np.load(group_legacy / "4ch" / "native_maps.npz") as original:
        maps = {name: original[name] for name in original.files}
    maps["group_idx"] = np.asarray([4, 5, 6, 99])
    np.savez_compressed(group_legacy / "4ch" / "native_maps.npz", **maps)
    with pytest.raises(ValueError, match="group IDs"):
        migrate_native_fit_cache(group_legacy, group_source / "CYJ", tmp_path / "bad_group")
    vps_source = _prepared(tmp_path / "vps_prepared")
    vps_legacy = _legacy_native_maps(vps_source, tmp_path / "vps_legacy", monkeypatch)
    with np.load(vps_source / "CYJ" / "sax" / "observations.npz") as original:
        observations = {name: original[name] for name in original.files}
    observations["vps"] = np.full_like(observations["vps"], 86)
    np.savez_compressed(vps_source / "CYJ" / "sax" / "observations.npz", **observations)
    _update_historical_observation_hash(vps_legacy, vps_source, "sax")
    with pytest.raises(ValueError, match="frozen timing/TR/VPS"):
        migrate_native_fit_cache(vps_legacy, vps_source / "CYJ", tmp_path / "bad_vps")


def test_native_cache_migration_rejects_historical_observation_identity_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _prepared(tmp_path / "prepared")
    legacy = _legacy_native_maps(source, tmp_path / "legacy", monkeypatch)
    historical = legacy.parent / "input_manifest.json"
    payload = json.loads(historical.read_text()); payload["observations"] = {stack: "wrong" for stack in ("sax", "2ch", "4ch")}; historical.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="historical R0 provenance"):
        migrate_native_fit_cache(legacy, source / "CYJ", tmp_path / "migrated")


def test_native_cache_migration_requires_expected_historical_manifest_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _prepared(tmp_path / "prepared")
    legacy = _legacy_native_maps(source, tmp_path / "legacy", monkeypatch)
    historical = legacy.parent / "input_manifest.json"
    historical.unlink()
    with pytest.raises(FileNotFoundError, match="input_manifest"):
        migrate_native_fit_cache(legacy, source / "CYJ", tmp_path / "missing")
    _legacy_native_maps(source, tmp_path / "method_legacy", monkeypatch)
    historical = tmp_path / "input_manifest.json"
    payload = json.loads(historical.read_text()); payload["method"] = "proposed"; historical.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="method"):
        migrate_native_fit_cache(tmp_path / "method_legacy", source / "CYJ", tmp_path / "wrong_method")
    _legacy_native_maps(source, tmp_path / "dictionary_legacy", monkeypatch)
    payload = json.loads(historical.read_text()); payload["dictionary"] = "other"; historical.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="dictionary"):
        migrate_native_fit_cache(tmp_path / "dictionary_legacy", source / "CYJ", tmp_path / "wrong_dictionary")
    _legacy_native_maps(source, tmp_path / "subject_legacy", monkeypatch)
    payload = json.loads(historical.read_text()); payload["subject_id"] = "DYZ"; historical.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="subject"):
        migrate_native_fit_cache(tmp_path / "subject_legacy", source / "CYJ", tmp_path / "wrong_subject")


def test_r0_cache_resolver_precedence_and_cohort_subject_paths() -> None:
    resolver = Path(__file__).resolve().parents[1] / "phase2_clean" / "server_commands" / "r0_native_cache_path.sh"
    def resolve(subject: str, run_id: str, extra: dict[str, str]) -> str:
        command = f'set -eu; DATA=/data; source {shlex.quote(str(resolver))}; phase4r_resolve_r0_native_maps {shlex.quote(subject)} {shlex.quote(run_id)}'
        return subprocess.check_output(["bash", "-c", command], text=True, env={**__import__("os").environ, **extra}).strip()
    assert resolve("CYJ", "r1", {"PHASE4R_R0_CACHE_ROOT": "/cache"}) == "/cache/CYJ/native_maps"
    assert resolve("DYZ", "r1", {"PHASE4R_R0_CACHE_ROOT": "/cache"}) == "/cache/DYZ/native_maps"
    assert resolve("CYJ", "r1", {"PHASE4R_R0_CACHE_ROOT": "/cache", "PHASE4R_R0_NATIVE_MAPS_ROOT": "/one"}) == "/one"
    assert resolve("HJL", "r1", {}) == "/data/Code_10w_runs/paper_joint_vs_2dfit/real/R0_native_cache/R0_PHASE4R/HJL/native_maps"
    assert resolve("HJL", "r1", {"PHASE4R_R0_CACHE_ID": "validated"}) == "/data/Code_10w_runs/paper_joint_vs_2dfit/real/R0_native_cache/validated/HJL/native_maps"


def test_r1_native_reuse_refuses_l100_and_sparse_runners_use_r0_cache(tmp_path: Path) -> None:
    source = _prepared(tmp_path / "prepared")
    l100 = build_sparse_condition(source, "CYJ", "L100", 0, tmp_path / "sparse", git_sha="abc")
    with pytest.raises(ValueError, match="L100 reuses R0"):
        reuse_native_maps(tmp_path / "missing", l100["manifest_path"], tmp_path / "r1_native")
    root = Path(__file__).resolve().parents[1] / "phase2_clean" / "server_commands"
    for name in ("run_real_sparse_fit_first.sh", "run_real_sparse_pilot_fit_first.sh"):
        text = (root / name).read_text()
        assert "--reuse-native-maps-root" in text
        assert "--sparse-manifest" in text
    for name in ("evaluate_real_sparse.sh", "evaluate_real_sparse_pilot.sh", "evaluate_real_sparse_L100.sh"):
        text = (root / name).read_text()
        assert "phase4r_resolve_r0_native_maps" in text
        assert "--r0-native-maps-root" in text


def test_heldout_reference_selection_uses_cached_original_groups_without_refitting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _prepared(tmp_path / "prepared")
    sparse = build_sparse_condition(source, "CYJ", "L50", 0, tmp_path / "sparse", git_sha="abc", audit_stack=lambda _: {"regular_gap": True, "gap_mm": 10.0, "group_count": 2})
    cache = _full_native_maps(source, tmp_path / "r0_cache", monkeypatch)
    selected = select_verified_native_cache_groups(cache, sparse["manifest_path"], "2ch", [5, 7])
    assert selected["group_idx"].tolist() == [5, 7]
    assert selected["source_support"].shape[0] == 2
    with pytest.raises(ValueError, match="authoritative held-out"):
        select_verified_native_cache_groups(cache, sparse["manifest_path"], "2ch", [4, 6])
    assert "fit_native" not in Path(heldout.__file__).read_text(encoding="utf-8")
    # Group-local cache values equal the historical deterministic held-out fit.
    with np.load(source / "CYJ" / "2ch" / "observations.npz") as original:
        data = {name: value[np.isin(original["group_idx"], [5, 7])] if value.shape[:1] == original["group_idx"].shape else value for name, value in original.items()}
    subset = tmp_path / "heldout_observations.npz"; np.savez_compressed(subset, **data)
    direct = fit_first.fit_native(subset, tmp_path / "historical_refit")
    with np.load(direct / "native_maps.npz") as expected:
        assert np.array_equal(selected["t1_ms"], expected["t1_ms"], equal_nan=True)
        assert np.array_equal(selected["t2_ms"], expected["t2_ms"], equal_nan=True)


def test_l100_heldout_reference_selection_uses_all_cached_lax_groups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _prepared(tmp_path / "prepared")
    sparse = build_sparse_condition(source, "CYJ", "L100", 0, tmp_path / "sparse", git_sha="abc")
    cache = _full_native_maps(source, tmp_path / "r0_cache", monkeypatch)
    selected = select_verified_native_cache_groups(cache, sparse["manifest_path"], "4ch", [4, 5, 6, 7])
    assert selected["group_idx"].tolist() == [4, 5, 6, 7]


@pytest.mark.parametrize("mutation, error", (
    ("map_hash", "SHA256 mismatch"), ("geometry_hash", "SHA256 mismatch"), ("source_hash", "source observation SHA256 mismatch"),
    ("duplicate", "exactly once"), ("missing", "exactly once"), ("dictionary", "dictionary contract"), ("vps", "frozen timing/TR/VPS"),
))
def test_heldout_cache_selection_rejects_invalid_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str, error: str) -> None:
    source = _prepared(tmp_path / "prepared")
    sparse = build_sparse_condition(source, "CYJ", "L50", 0, tmp_path / "sparse", git_sha="abc", audit_stack=lambda _: {"regular_gap": True, "gap_mm": 10.0, "group_count": 2})
    cache = _full_native_maps(source, tmp_path / "r0_cache", monkeypatch)
    manifest_path = Path(sparse["manifest_path"])
    if mutation == "map_hash":
        payload = json.loads((cache / "2ch" / "manifest.json").read_text()); payload["native_maps_sha256"] = "wrong"; (cache / "2ch" / "manifest.json").write_text(json.dumps(payload))
    elif mutation == "geometry_hash":
        payload = json.loads((cache / "2ch" / "manifest.json").read_text()); payload["geometry_sha256"] = "wrong"; (cache / "2ch" / "manifest.json").write_text(json.dumps(payload))
    elif mutation == "source_hash":
        payload = json.loads(manifest_path.read_text()); payload["stacks"]["2ch"]["source_observation"]["sha256"] = "wrong"; manifest_path.write_text(json.dumps(payload))
    elif mutation == "duplicate":
        with np.load(cache / "2ch" / "native_maps.npz") as maps:
            values = {name: maps[name] for name in maps.files}
        values["group_idx"] = np.asarray([4, 5, 5, 7]); np.savez_compressed(cache / "2ch" / "native_maps.npz", **values)
        payload = json.loads((cache / "2ch" / "manifest.json").read_text()); payload["native_maps_sha256"] = fit_first.sha256(cache / "2ch" / "native_maps.npz"); (cache / "2ch" / "manifest.json").write_text(json.dumps(payload))
    elif mutation == "missing":
        with np.load(cache / "2ch" / "native_maps.npz") as maps:
            values = {name: maps[name] for name in maps.files}
        values["group_idx"] = np.asarray([4, 5, 6, 8]); np.savez_compressed(cache / "2ch" / "native_maps.npz", **values)
        payload = json.loads((cache / "2ch" / "manifest.json").read_text()); payload["native_maps_sha256"] = fit_first.sha256(cache / "2ch" / "native_maps.npz"); (cache / "2ch" / "manifest.json").write_text(json.dumps(payload))
    elif mutation == "dictionary":
        payload = json.loads((cache / "2ch" / "manifest.json").read_text()); payload["dictionary_contract"] = {}; (cache / "2ch" / "manifest.json").write_text(json.dumps(payload))
    else:
        with np.load(source / "CYJ" / "2ch" / "observations.npz") as observations:
            values = {name: observations[name] for name in observations.files}
        values["vps"] = np.full_like(values["vps"], 86); np.savez_compressed(source / "CYJ" / "2ch" / "observations.npz", **values)
        payload = json.loads((cache / "2ch" / "manifest.json").read_text()); payload["source_observations"]["sha256"] = fit_first.sha256(source / "CYJ" / "2ch" / "observations.npz"); (cache / "2ch" / "manifest.json").write_text(json.dumps(payload))
        sparse_payload = json.loads(manifest_path.read_text()); sparse_payload["stacks"]["2ch"]["source_observation"]["sha256"] = fit_first.sha256(source / "CYJ" / "2ch" / "observations.npz"); manifest_path.write_text(json.dumps(sparse_payload))
    with pytest.raises(ValueError, match=error):
        select_verified_native_cache_groups(cache, manifest_path, "2ch", [5, 7])


def _dummy_volume(path: Path) -> Path:
    import nibabel as nib
    nib.save(nib.Nifti1Image(np.ones((1, 1, 1), np.float32), np.eye(4)), str(path))
    return path


def test_heldout_evaluation_records_immutable_cache_provenance_without_partial_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _prepared(tmp_path / "prepared")
    sparse = build_sparse_condition(source, "CYJ", "L50", 0, tmp_path / "sparse", git_sha="abc", audit_stack=lambda _: {"regular_gap": True, "gap_mm": 10.0, "group_count": 2})
    cache = _full_native_maps(source, tmp_path / "r0_cache", monkeypatch)
    monkeypatch.setattr(heldout, "reproject_volume_to_native_plane", lambda _volume, _affine, _native, shape: (np.full(shape, 1111.0), np.ones(shape, bool)))
    maps = [_dummy_volume(tmp_path / f"{name}.nii.gz") for name in ("proposed_t1", "proposed_t2", "fit_t1", "fit_t2")]
    output = tmp_path / "formal_heldout"
    report_path = heldout.evaluate_heldout(sparse["manifest_path"], *maps[:2], *maps[2:], output, cache)
    report = json.loads(report_path.read_text())
    assert report["provenance"]["native_reference_source"] == "R0_native_cache"
    assert report["provenance"]["native_fit_recomputed"] is False
    assert report["stacks"]["2ch"]["native_reference"]["heldout_original_group_ids"] == [5, 7]
    assert report["stacks"]["2ch"]["native_reference"]["native_maps_sha256"] == fit_first.sha256(cache / "2ch" / "native_maps.npz")
    assert not list(tmp_path.glob(".formal_heldout.staging-*"))


def test_heldout_evaluation_cache_failure_never_publishes_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _prepared(tmp_path / "prepared")
    sparse = build_sparse_condition(source, "CYJ", "L50", 0, tmp_path / "sparse", git_sha="abc", audit_stack=lambda _: {"regular_gap": True, "gap_mm": 10.0, "group_count": 2})
    cache = _full_native_maps(source, tmp_path / "r0_cache", monkeypatch)
    payload = json.loads((cache / "2ch" / "manifest.json").read_text()); payload["geometry_sha256"] = "wrong"; (cache / "2ch" / "manifest.json").write_text(json.dumps(payload))
    maps = [_dummy_volume(tmp_path / f"bad_{name}.nii.gz") for name in ("proposed_t1", "proposed_t2", "fit_t1", "fit_t2")]
    output = tmp_path / "formal_heldout"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        heldout.evaluate_heldout(sparse["manifest_path"], *maps[:2], *maps[2:], output, cache)
    assert not output.exists()
    assert not list(tmp_path.glob(".formal_heldout.staging-*"))


def test_r1_subject_uses_equal_fold_weighting_not_plane_weighting() -> None:
    planes = [
        {"subject": "CYJ", "method": "proposed", "level": "L50", "parameter": "t1_ms", "phase": 0, "support_voxels": 1, "sum_squared_error": 100.0, "sum_absolute_error": 10.0, "sum_signed_error": 10.0},
        *[{"subject": "CYJ", "method": "proposed", "level": "L50", "parameter": "t1_ms", "phase": 1, "support_voxels": 1, "sum_squared_error": 900.0, "sum_absolute_error": 30.0, "sum_signed_error": 30.0} for _ in range(3)],
    ]
    folds = pooled_r1_fold_metrics(planes)
    assert [row["pooled_rmse"] for row in folds] == [10.0, 30.0]
    subject = summarize_r1_subjects(folds)
    assert subject[0]["rmse_mean_over_folds"] == 20.0
    assert subject[0]["fold_values"] == [10.0, 30.0]


def test_r0_paired_deltas_require_matched_subjects_and_reference_means() -> None:
    records = []
    for subject, proposed, baseline in (("CYJ", 10.0, 12.0), ("DYZ", 20.0, 24.0), ("HHZ", 30.0, 29.0), ("HJL", 40.0, 45.0)):
        for method, rmse in (("proposed", proposed), ("fit_first", baseline)):
            records.append({"subject": subject, "method": method, "parameter": "t1_ms", "rmse": rmse, "mae": rmse / 2, "bias": 0.0, "reference_mean_ms": 1000.0, "prediction_mean_ms": 1000.0 + rmse})
    paired, summary = paired_r0_comparison(records)
    assert [row["delta_rmse"] for row in paired] == [-2.0, -4.0, 1.0, -5.0]
    assert summary["t1_ms"]["mean_delta"] == pytest.approx(-2.5)
    records[-1]["reference_mean_ms"] = 1001.0
    with pytest.raises(ValueError, match="reference mean mismatch"):
        paired_r0_comparison(records)
