"""CPU-only endpoint-export policy contracts for the Q002 launcher."""
from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pytest


def _run_args(tmp_path: Path, *, endpoint_policy: str | None) -> Namespace:
    b6, decoder, protocol = (tmp_path / name for name in ("b6.json", "decoder.pt", "protocol.yaml"))
    for path in (b6, decoder, protocol):
        path.write_text("decoder" if path == decoder else path.name, encoding="utf-8")
    args = Namespace(
        experiment_id="Q002S640_B6_roi_same_anchor_mse", subject_id="CYJ",
        cropped_prepared_root=str(tmp_path / "prepared"), output=str(tmp_path / "output"),
        b6_model=str(b6), signal_simulator=str(decoder), protocol=str(protocol),
        device="cuda:0", anchor_batch_size=640, sparse_manifest=None,
        expected_sparse_generator_git_sha=None, expected_sparse_manifest_sha256=None,
    )
    if endpoint_policy is not None:
        args.endpoint_policy = endpoint_policy
    return args


def _patch_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, calls: list[tuple[str, str]]) -> None:
    from quality_experiments.Q002_B6_roi_same_anchor_mse import run_q002_reconstruction as runner

    monkeypatch.setattr(runner, "validate_formal_sparse_reconstruction_dependencies", lambda _args: None)
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner, "profiled_observation_paths", lambda *_args: [])
    monkeypatch.setattr(runner, "cropped_observation_paths", lambda *_args: [])
    monkeypatch.setattr(runner, "load_b6_resolved_config", lambda _path: {"decoder": {"checkpoint": "frozen"}, "training": {}, "export": {"output_resolution_mm": 1.0, "output_batch_size": 1}})
    monkeypatch.setattr(runner, "build_q002_route_config", lambda *_args, **_kwargs: {"resolved_config": {"training": {}, "export": {"output_resolution_mm": 1.0, "output_batch_size": 1}}, "route": {"sampling": {"shared_training_psf_samples": 8}, "endpoint_semantics": {"central": "central", "psf128": "posthoc"}}})
    monkeypatch.setattr(runner, "QuantPointDataset", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runner, "JointAnchorDataset", lambda *_args, **_kwargs: type("Joint", (), {"qc": {}})())

    def train(*_args: object, **kwargs: object) -> dict[str, object]:
        output = Path(_args[4]); output.mkdir(parents=True); (output / "model.pt").write_text("model", encoding="utf-8")
        return {"model": object(), "training_space": object(), "controls": {}, "intensity_normalization": {}, "stack_weights": {}, "fixed_monitor": {"seed": 1, "anchor_identity_sha256": "anchor"}, "resolved_config": {}, "primary_checkpoint": None, "global_iteration": 6000}
    monkeypatch.setattr(runner, "train_q002_reconstruction", train)

    def raw_export(*_args: object, **_kwargs: object) -> dict[str, str]:
        output = Path(_args[2]); output.mkdir(parents=True, exist_ok=True)
        paths = {"t1_ms": output / "T1.nii.gz", "t2_ms": output / "T2.nii.gz", "b1": output / "B1.nii.gz", "amplitude": output / "amplitude.nii.gz", "poses": output / "final_rigid_poses.json"}
        for path in paths.values(): path.write_text("raw", encoding="utf-8")
        calls.append(("raw", str(output)))
        return {name: str(path) for name, path in paths.items()}
    monkeypatch.setattr(runner, "export_quantitative_outputs", raw_export)
    monkeypatch.setattr(runner, "export_quantitative_outputs_psf128", lambda *_args, **_kwargs: calls.append(("psf128", "called")))
    monkeypatch.setattr(runner, "export_native_plane_reprojections", lambda *_args, **_kwargs: calls.append(("native", str(_args[3]))))
    decoder = tmp_path / "decoder.pt"
    decoder.write_text("decoder", encoding="utf-8")
    monkeypatch.setattr(runner, "APPROVED_DECODER_SHA256", runner._sha256(decoder))
    monkeypatch.setattr(runner, "_git_sha", lambda: "unused", raising=False)
    import subprocess
    monkeypatch.setattr(subprocess, "check_output", lambda command, **_kwargs: "deadbeef\n" if command[:3] == ["git", "rev-parse", "HEAD"] else "")


def test_central_only_keeps_k8_training_and_writes_only_central_endpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from quality_experiments.Q002_B6_roi_same_anchor_mse import run_q002_reconstruction as runner

    calls: list[tuple[str, str]] = []
    _patch_runner(monkeypatch, tmp_path, calls)
    output = runner.run(_run_args(tmp_path, endpoint_policy="central_only"))
    manifest = json.loads((output / "q002_route_manifest.json").read_text(encoding="utf-8"))

    assert manifest["route"]["sampling"]["shared_training_psf_samples"] == 8
    assert ("raw", str(output)) in calls
    assert any(name == "native" and target.endswith("evaluation/mapping_central_no_psf") for name, target in calls)
    assert not any("signal_psf_K8" in target for name, target in calls if name == "native")
    assert ("psf128", "called") not in calls
    assert manifest["endpoint_policy"] == "central_only"
    assert manifest["formal_endpoint"] == "central"
    assert manifest["training_psf_samples"] == 8
    assert manifest["central_output_sampling_psf_samples"] == 1
    assert "output_sampling_psf_samples" not in manifest
    assert manifest["psf128_export_generated"] is False
    assert manifest["signal_psf_k8_generated"] is False
    assert "psf128_contract" not in manifest
    assert (output / "final_rigid_poses.json").is_file()


def test_legacy_policy_is_default_and_retains_psf128_and_k8_signal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from quality_experiments.Q002_B6_roi_same_anchor_mse import run_q002_reconstruction as runner

    calls: list[tuple[str, str]] = []
    _patch_runner(monkeypatch, tmp_path, calls)
    output = runner.run(_run_args(tmp_path, endpoint_policy=None))
    manifest = json.loads((output / "q002_route_manifest.json").read_text(encoding="utf-8"))

    assert any("signal_psf_K8" in target for name, target in calls if name == "native")
    assert ("psf128", "called") in calls
    assert manifest["endpoint_policy"] == "legacy"
    assert manifest["central_output_sampling_psf_samples"] == 1
    assert "output_sampling_psf_samples" not in manifest
    assert manifest["psf128_export_generated"] is True
    assert manifest["signal_psf_k8_generated"] is True
    assert manifest["psf128_contract"] == {"resolution_mm": 1.0, "factor": 1.0, "samples": 128, "seed": 20260911}


def test_central_only_cannot_change_historical_k4_route(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from quality_experiments.Q002_B6_roi_same_anchor_mse import run_q002_reconstruction as runner

    args = _run_args(tmp_path, endpoint_policy="central_only")
    args.experiment_id = "Q005S640_K4_B6_roi_same_anchor_mse"
    with pytest.raises(ValueError, match="Q002S640/K8"):
        runner.run(args)

    calls: list[tuple[str, str]] = []
    _patch_runner(monkeypatch, tmp_path, calls)
    args.endpoint_policy = "legacy"
    output = runner.run(args)
    assert ("psf128", "called") in calls
    assert any("signal_psf_K8" in target for name, target in calls if name == "native")
    assert json.loads((output / "q002_route_manifest.json").read_text(encoding="utf-8"))["endpoint_policy"] == "legacy"


def test_parser_exposes_backward_compatible_explicit_endpoint_policy() -> None:
    from quality_experiments.Q002_B6_roi_same_anchor_mse.run_q002_reconstruction import build_parser

    args = build_parser().parse_args(["--cropped-prepared-root", "/crop", "--b6-model", "/b6", "--signal-simulator", "/decoder", "--output", "/out"])
    assert args.endpoint_policy == "legacy"
    explicit = build_parser().parse_args(["--cropped-prepared-root", "/crop", "--b6-model", "/b6", "--signal-simulator", "/decoder", "--output", "/out", "--endpoint-policy", "central_only"])
    assert explicit.endpoint_policy == "central_only"
