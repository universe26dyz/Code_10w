"""K4 must be a distinct route with no hidden scientific control drift."""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest


def test_q005_k4_differs_from_q002_k8_only_by_training_psf_derived_locations() -> None:
    from quality_experiments.Q002_B6_roi_same_anchor_mse.contracts import Q002S640_EXPERIMENT_ID, Q005S640_K4_EXPERIMENT_ID, build_q002_route_config

    base = {"training": {"seed": 20260911, "scheduler": "frozen"}, "decoder": {"checkpoint": "/approved.pth"}, "export": {"sample_psf": 8, "endpoint_psf_samples": 128}}
    k8 = build_q002_route_config(base, cropped_prepared_root="/cropped", experiment_id=Q002S640_EXPERIMENT_ID, anchor_batch_size=640)
    k4 = build_q002_route_config(base, cropped_prepared_root="/cropped", experiment_id=Q005S640_K4_EXPERIMENT_ID, anchor_batch_size=640)
    assert k4["resolved_config"] == k8["resolved_config"]
    assert k4["route"]["sampling"]["shared_training_psf_samples"] == 4
    assert k4["route"]["sampling"]["data_psf_inr_location_count"] == 2560
    assert k4["route"]["sampling"]["psf_forward_chunk_size"] == 4
    assert {key: value for key, value in k4["route"]["sampling"].items() if key not in {"shared_training_psf_samples", "data_psf_inr_location_count", "psf_forward_chunk_size"}} == {key: value for key, value in k8["route"]["sampling"].items() if key not in {"shared_training_psf_samples", "data_psf_inr_location_count", "psf_forward_chunk_size"}}
    assert k4["route"]["scientific_change"] == "training_psf_samples_8_to_4_only"
    assert k4["route"]["endpoint_semantics"]["psf128"] == "post-hoc map-domain PSF128 endpoint, distinct from training K4"


def test_q005_k4_cpu_forward_uses_exactly_2560_locations_and_chunk_four() -> None:
    import torch
    from torch import nn
    from trad.modules.module_06_rigid_psf.rigid_psf_forward import TradQuantitativeForward

    class PSF(nn.Module):
        def __init__(self): super().__init__(); self.requested = []
        def sample_local_then_transform(self, xyz, group_idx, n_samples):
            self.requested.append(n_samples); return xyz[:, None, :].expand(-1, n_samples, -1)
    class INR(nn.Module):
        def __init__(self): super().__init__(); self.calls = []
        def forward(self, xyz):
            self.calls.append(tuple(xyz.shape)); value = xyz[:, 0]; return {"t1_ms": value, "t2_ms": value, "b1": value, "amplitude": torch.ones_like(value)}
    class Decoder(nn.Module):
        def forward(self, t1, t2, b1, timing, protocol, normalize=True): return t1[:, None].expand(-1, 10)

    psf, inr = PSF(), INR()
    forward = TradQuantitativeForward(inr, Decoder(), psf, protocol=object())
    prediction = forward.forward_fingerprint(torch.zeros((640, 3)), torch.zeros(640, dtype=torch.long), torch.zeros((640, 9)), n_psf_samples=4, psf_chunk_size=4)
    assert prediction.shape == (640, 10)
    assert psf.requested == [4]
    assert inr.calls == [(2560, 3)]


@pytest.mark.parametrize("experiment_id", ("Q002S640_B6_roi_same_anchor_mse", "Q005S640_K4_B6_roi_same_anchor_mse"))
def test_formal_sparse_routes_require_matching_manifest_and_frozen_dependency_pins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, experiment_id: str) -> None:
    from quality_experiments.Q002_B6_roi_same_anchor_mse import run_q002_reconstruction as route

    b6, decoder, protocol = (tmp_path / name for name in ("b6.json", "decoder.pt", "protocol.yaml"))
    b6.write_text("b6", encoding="utf-8"); decoder.write_text("decoder", encoding="utf-8"); protocol.write_text("protocol", encoding="utf-8")
    monkeypatch.setattr(route, "B6_CONFIG_SOURCE_SHA256", route._sha256(b6))
    monkeypatch.setattr(route, "APPROVED_DECODER_SHA256", route._sha256(decoder))
    manifest = {"subject": "CYJ", "prepared_root": str((tmp_path / "prepared").resolve()), "_manifest_sha256": "a" * 64}
    seen: list[dict] = []
    monkeypatch.setattr(route, "validate_sparse_profile_manifest", lambda *_args, **kwargs: seen.append(kwargs) or manifest)
    args = Namespace(experiment_id=experiment_id, sparse_manifest=str(tmp_path / "sparse.json"), expected_sparse_generator_git_sha="b" * 40, expected_sparse_manifest_sha256="a" * 64, expected_b6_model_sha256=route._sha256(b6), expected_protocol_sha256=route._sha256(protocol), b6_model=str(b6), signal_simulator=str(decoder), protocol=str(protocol), subject_id="CYJ", cropped_prepared_root=str(tmp_path / "prepared"), output=str(tmp_path / "fresh"))

    route.validate_formal_sparse_reconstruction_dependencies(args)

    assert seen == [{"subject": "CYJ", "strict": True, "expected_generator_git_sha": "b" * 40, "expected_manifest_sha256": "a" * 64}]
    args.expected_sparse_generator_git_sha = None
    with pytest.raises(ValueError, match="generator"):
        route.validate_formal_sparse_reconstruction_dependencies(args)
    args.expected_sparse_generator_git_sha = "b" * 40; args.expected_sparse_manifest_sha256 = "0" * 64
    with pytest.raises(ValueError, match="manifest SHA256"):
        route.validate_formal_sparse_reconstruction_dependencies(args)
    args.expected_sparse_manifest_sha256 = "a" * 64; args.expected_b6_model_sha256 = "0" * 64
    with pytest.raises(ValueError, match="B6 model SHA256"):
        route.validate_formal_sparse_reconstruction_dependencies(args)
    args.expected_b6_model_sha256 = route._sha256(b6); args.expected_protocol_sha256 = "0" * 64
    with pytest.raises(ValueError, match="protocol SHA256"):
        route.validate_formal_sparse_reconstruction_dependencies(args)
