"""CPU contracts for the Final5 CYJ Experiment 1.1 Fit-first route."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _native_stack(root: Path, stack: str, groups: list[int]) -> tuple[Path, Path, np.ndarray]:
    stack_root = root / stack
    stack_root.mkdir(parents=True)
    observations = stack_root / "observations.npz"
    np.savez_compressed(observations, group_idx=np.repeat(np.asarray(groups), 10))
    values = np.asarray(groups, dtype=np.float32)[:, None, None] + np.arange(4, dtype=np.float32).reshape(1, 2, 2)
    maps = stack_root / "native_maps.npz"
    np.savez_compressed(maps, t1_ms=values, t2_ms=values + 100, b1=values + 200, group_idx=np.asarray(groups))
    geometry = stack_root / "geometry.npz"
    np.savez_compressed(geometry, affine_lps_rc=np.repeat(np.eye(4)[None], len(groups), axis=0), pixel_spacing_rc_mm=np.ones((len(groups), 2)), slice_thickness_mm=np.full(len(groups), 8.0))
    (stack_root / "manifest.json").write_text(json.dumps({
        "source_observations": {"path": str(observations.resolve()), "sha256": _sha256(observations)},
        "native_maps_sha256": _sha256(maps), "geometry_sha256": _sha256(geometry),
    }))
    return observations, maps, values


def _profile(root: Path, sources: dict[str, Path], retained: dict[str, list[int]]) -> Path:
    stacks = {}
    for stack, source in sources.items():
        stacks[stack] = {
            "source_observation": {"path": str(source.resolve()), "sha256": _sha256(source)},
            "retained_original_group_ids": retained[stack],
            "training_stack_present": bool(retained[stack]),
        }
    path = root / "sparse_manifest.json"
    path.write_text(json.dumps({"subject": "CYJ", "profile_id": "L50", "case_id": "L50_fold0", "fold_id": 0, "fold_count": 2, "git_sha": "a" * 40, "stacks": stacks}))
    return path


def test_frozen_native_map_selection_uses_manifest_ids_and_never_refits(tmp_path: Path, monkeypatch) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import fit_first_exp1_1 as route

    native = tmp_path / "native"
    sax_source, _, sax_values = _native_stack(native, "sax", [0, 1])
    twoch_source, _, twoch_values = _native_stack(native, "2ch", [10, 20, 30])
    fourch_source, _, _ = _native_stack(native, "4ch", [4, 5])
    sparse = _profile(tmp_path, {"sax": sax_source, "2ch": twoch_source, "4ch": fourch_source}, {"sax": [0, 1], "2ch": [10, 30], "4ch": []})
    profile = json.loads(sparse.read_text())
    profile.update({"_manifest_path": str(sparse.resolve()), "_manifest_sha256": _sha256(sparse)})
    monkeypatch.setattr(route, "validate_sparse_profile_manifest", lambda *_args, **_kwargs: profile)
    monkeypatch.setattr(route, "fit_native", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("formal frozen-map route must not refit")), raising=False)

    selected = route.select_frozen_native_maps(native, sparse, tmp_path / "selected", subject="CYJ")

    assert selected["sax"] == [0, 1]
    assert selected["2ch"] == [10, 30]
    assert selected["4ch"] == []
    with np.load(tmp_path / "selected" / "2ch" / "native_maps.npz") as maps:
        assert maps["group_idx"].tolist() == [10, 30]
        assert np.array_equal(maps["t1_ms"], twoch_values[[0, 2]])
    with np.load(tmp_path / "selected" / "sax" / "native_maps.npz") as maps:
        assert np.array_equal(maps["t1_ms"], sax_values)
    assert not (tmp_path / "selected" / "4ch").exists()


def test_exp1_1_commands_are_independent_one_stage_without_sample_masks(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.fit_first_exp1_1 import build_exp1_1_nesvor_commands
    from paper_experiments.joint_vs_2dfit.phase2_clean.fit_first import build_independent_nesvor_commands

    inputs = [tmp_path / "selected" / "sax", tmp_path / "selected" / "2ch"]
    commands = build_exp1_1_nesvor_commands(inputs, tmp_path / "output" / "nesvor", device="1", nesvor_executable="nesvor")

    assert commands == build_independent_nesvor_commands(inputs, tmp_path / "output" / "nesvor", device="1", nesvor_executable="nesvor")
    assert len(commands) == 2
    assert all("--registration stack" in command and "--no-slice-scale" in command for command in commands)
    assert all("--no-transformation-optimization" not in command and "--sample-mask" not in command for command in commands)
    assert all("--n-iter 10000" in command and "--n-samples 256" in command and "--n-inference-samples 128" in command for command in commands)
    assert "--scale 2500" in commands[0] and "--scale 200" in commands[1]


def test_exp1_1_single_stack_commands_use_only_explicit_output_sampling_masks(tmp_path: Path) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean.fit_first_exp1_1 import build_exp1_1_nesvor_commands

    masks = {}
    for parameter in ("t1", "t2"):
        mask = tmp_path / "nesvor_inputs" / "sax" / parameter / "stack_mask.nii.gz"
        mask.parent.mkdir(parents=True, exist_ok=True); mask.write_bytes(parameter.encode())
        masks[parameter] = mask
    commands = build_exp1_1_nesvor_commands([tmp_path / "selected" / "sax"], tmp_path / "output" / "nesvor", device="1", nesvor_executable="nesvor", sample_masks=masks)

    assert len(commands) == 2
    for parameter, command in zip(("t1", "t2"), commands):
        assert f"--sample-mask {masks[parameter]}" in command
        assert "--registration stack" in command and "--no-slice-scale" in command
        assert "--no-transformation-optimization" not in command
        assert "--n-iter 10000" in command and "--n-samples 256" in command and "--n-inference-samples 128" in command
    assert "--scale 2500" in commands[0] and "--scale 200" in commands[1]


def test_l100_selection_preserves_all_standardized_12_15_10_maps(tmp_path: Path, monkeypatch) -> None:
    from paper_experiments.joint_vs_2dfit.phase2_clean import fit_first_exp1_1 as route

    native = tmp_path / "native"
    sources = {}
    retained = {}
    for stack, count in (("sax", 12), ("2ch", 15), ("4ch", 10)):
        source, _maps, _values = _native_stack(native, stack, list(range(count)))
        sources[stack], retained[stack] = source, list(range(count))
    sparse = _profile(tmp_path, sources, retained)
    profile = json.loads(sparse.read_text())
    profile.update({"profile_id": "L100", "case_id": "L100", "fold_id": None, "fold_count": 1, "_manifest_path": str(sparse.resolve()), "_manifest_sha256": _sha256(sparse)})
    monkeypatch.setattr(route, "validate_sparse_profile_manifest", lambda *_args, **_kwargs: profile)

    selected = route.select_frozen_native_maps(native, sparse, tmp_path / "selected", subject="CYJ")

    assert {stack: len(groups) for stack, groups in selected.items()} == {"sax": 12, "2ch": 15, "4ch": 10}


@pytest.mark.parametrize(
    ("case_id", "profile_id", "fold_id", "fold_count", "retained", "expected_stacks"),
    [
        ("L100", "L100", None, 1, {"sax": [0], "2ch": [0], "4ch": [0]}, ("sax", "2ch", "4ch")),
        ("L50_fold0", "L50", 0, 2, {"sax": [0], "2ch": [0], "4ch": [0]}, ("sax", "2ch", "4ch")),
        ("SAX100_2CH0_4CH0", "SAX100_2CH0_4CH0", None, 1, {"sax": [0], "2ch": [], "4ch": []}, ("sax",)),
    ],
)
def test_execute_creates_nesvor_output_dirs_and_final_manifest_admits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_id: str,
    profile_id: str,
    fold_id: int | None,
    fold_count: int,
    retained: dict[str, list[int]],
    expected_stacks: tuple[str, ...],
) -> None:
    """Exercise the actual producer finalization and Final5 admission together."""
    from paper_experiments.joint_vs_2dfit.phase2_clean import fit_first_exp1_1 as route
    from paper_experiments.joint_vs_2dfit.phase2_clean import final5_batch_runner as runner
    from paper_experiments.joint_vs_2dfit.phase2_clean.fit_first_route_contract import SCMR_ONE_STAGE_FIXED_ARGS

    native = tmp_path / "native"
    sources = {}
    for stack, groups in (("sax", [0]), ("2ch", [0]), ("4ch", [0])):
        source, _maps, _values = _native_stack(native, stack, groups)
        sources[stack] = source
    sparse = _profile(tmp_path, sources, retained)
    profile = json.loads(sparse.read_text())
    profile.update({"profile_id": profile_id, "case_id": case_id, "fold_id": fold_id, "fold_count": fold_count, "_manifest_path": str(sparse.resolve()), "_manifest_sha256": _sha256(sparse)})
    monkeypatch.setattr(route, "validate_sparse_profile_manifest", lambda *_args, **_kwargs: profile)
    output = tmp_path / "fit"

    def fake_run(command: str, *, shell: bool, check: bool) -> None:
        assert shell and check
        assert (output / "nesvor" / "t1").is_dir()
        assert (output / "nesvor" / "t2").is_dir()
        parameter = "t1" if "/nesvor/t1/" in command else "t2"
        target = output / "nesvor" / parameter
        (target / "model.pt").write_bytes(parameter.encode())
        (target / "slices").mkdir()
        (target / "slices" / "0.nii.gz").write_bytes(b"slice")
        (target / "volume_ms.nii.gz").write_bytes(b"volume")
        inputs = [str((output / "nesvor_inputs" / stack / parameter / "stack.nii.gz").resolve()) for stack in expected_stacks]
        (target / "result.json").write_text(json.dumps({"input_stacks": inputs, "thicknesses": [8] * len(expected_stacks), "registration": "stack", "no_slice_scale": True, "no_transformation_optimization": False, **SCMR_ONE_STAGE_FIXED_ARGS}))

    monkeypatch.setattr(route.subprocess, "run", fake_run)
    route.run_exp1_1_fit_first(subject="CYJ", sparse_manifest=sparse, standardized_native_dictionary_root=native, output=output, nesvor_executable="nesvor", device="0", expected_generator_git_sha="a" * 40, expected_manifest_sha256=_sha256(sparse), execute=True)

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["sparse_manifest"]["generator_git_sha"] == "a" * 40
    assert tuple(manifest["active_stacks"]) == expected_stacks
    expected_policy = "explicit_single_stack_native_finite_support" if len(expected_stacks) == 1 else "nesvor_default_estimated_mask"
    assert manifest["output_sampling"]["policy"] == expected_policy
    assert manifest["output_sampling"]["scope"] == "NeSVoR output INR sampling only; not a reconstruction/training input mask"
    if len(expected_stacks) == 1:
        assert set(manifest["output_sampling"]["sample_masks"]) == {"t1", "t2"}
    record = runner.CaseRecord("CYJ", case_id, profile_id, fold_id, fold_count, sparse, _sha256(sparse), "a" * 40, tmp_path)
    result = runner.validate_fit_first_reconstruction(output, record)
    assert result.valid and result.state == "SKIP_VALID"
