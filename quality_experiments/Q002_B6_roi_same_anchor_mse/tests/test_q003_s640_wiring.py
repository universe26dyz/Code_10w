from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    ("experiment_id", "anchor_batch_size", "expected_cosine"),
    [
        ("Q002_B6_roi_same_anchor_mse", 64, None),
        ("Q002S640_B6_roi_same_anchor_mse", 640, None),
        ("Q003S640_B6_roi_same_anchor_mse_plus_cosine", 640, {"enabled": True, "weight": 1.0, "epsilon": 1.0e-8}),
    ],
)
def test_runtime_launcher_wires_only_q003_cosine_to_shared_trainer(monkeypatch, tmp_path, experiment_id, anchor_batch_size, expected_cosine):
    from quality_experiments.Q002_B6_roi_same_anchor_mse import run_q002_reconstruction as launcher

    inputs = []
    for name in ("sax", "2ch", "4ch"):
        path = tmp_path / f"{name}.npz"
        path.write_bytes(name.encode())
        inputs.append(path)
    b6_model, decoder = tmp_path / "b6.pt", tmp_path / "decoder.pth"
    b6_model.write_bytes(b"b6"); decoder.write_bytes(b"decoder")
    captured = {}

    monkeypatch.setattr(launcher.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(launcher, "cropped_observation_paths", lambda _: inputs)
    monkeypatch.setattr(launcher, "load_b6_resolved_config", lambda _: {"decoder": {"checkpoint": str(decoder)}, "training": {}, "export": {"output_resolution_mm": 1.0, "output_batch_size": 1}, "bbox": {}})
    monkeypatch.setattr(launcher, "QuantPointDataset", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(launcher, "JointAnchorDataset", lambda *_args, **_kwargs: SimpleNamespace(qc={}))

    def fake_train(_scalar, _joint, _resolved, _protocol, output, **kwargs):
        captured.update(kwargs)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "model.pt").write_bytes(b"checkpoint")
        return {"model": object(), "training_space": object(), "protocol": object(), "decoder_metadata": {}, "intensity_normalization": {}, "controls": {}, "stack_weights": {}, "output_dir": output, "fixed_monitor": {"seed": 20260911, "anchor_identity_sha256": "monitor"}}

    def fake_raw(_model, _space, output, *_args, **_kwargs):
        output = Path(output)
        paths = {name: output / f"{name}.bin" for name in ("t1_ms", "t2_ms", "b1", "amplitude", "poses")}
        for path in paths.values(): path.write_bytes(b"raw")
        return paths

    monkeypatch.setattr(launcher, "train_q002_reconstruction", fake_train)
    monkeypatch.setattr(launcher, "export_quantitative_outputs", fake_raw)
    monkeypatch.setattr(launcher, "export_quantitative_outputs_psf128", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(launcher, "export_native_plane_reprojections", lambda *_args, **_kwargs: None)
    output = tmp_path / "output"
    launcher.run(Namespace(cropped_prepared_root="/cropped", b6_model=str(b6_model), signal_simulator=str(decoder), output=str(output), protocol="protocol.yaml", device="cuda:0", experiment_id=experiment_id, anchor_batch_size=anchor_batch_size))
    assert captured["anchor_batch_size"] == anchor_batch_size
    assert captured["fingerprint_cosine"] == expected_cosine
