"""Synthetic contracts for the independent read-only D2 K=8 evaluator."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from trad.modules.module_06_rigid_psf.rigid_psf_forward import GroupRigidPSF

from quality_experiments.D2_k8_signal_domain.core import export_k8_reprojections, prepare_empty_output, signal_metric_row, strict_paired_signal_support


class _IdentitySpace:
    @staticmethod
    def local_to_train(value):
        return value


class _ToyPSFModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.rigid_psf = GroupRigidPSF(torch.zeros((3, 6)), torch.tensor([[1.0, 1.0, 8.0]]).repeat(3, 1))
        self.register_buffer("intensity_scale", torch.ones(()))
        self.n_samples_seen: list[int] = []

    def forward(self, batch, n_samples):
        self.n_samples_seen.append(n_samples)
        samples = self.rigid_psf.sample_local_then_transform(batch["xyz"], batch["group_idx"], n_samples)
        return samples[..., 2].square().mean(dim=1) + self.anchor


def _prepared(path):
    images = np.zeros((10, 3, 3), dtype=np.float32)
    np.savez_compressed(path, images=images, masks=np.ones_like(images, dtype=bool), group_idx=np.zeros(10, dtype=np.int64), weight_idx=np.arange(10, dtype=np.int64), timing9_ms=np.ones((10, 9), dtype=np.float32), tr_ms=np.ones(10), vps=np.ones(10, dtype=np.int64), pixel_spacing_rc_mm=np.ones((10, 2), dtype=np.float32))


def test_strict_paired_support_requires_observed_and_both_predictions_to_be_finite() -> None:
    observed = np.array([[1.0, np.nan], [2.0, 3.0]])
    bloch = np.array([[1.0, 4.0], [np.nan, 3.0]])
    mlp = np.array([[1.0, 4.0], [2.0, np.nan]])
    masks = np.ones((2, 2), dtype=bool)

    assert np.array_equal(strict_paired_signal_support(masks, observed, bloch, mlp), np.array([[True, False], [False, False]]))


def test_signal_metrics_keep_signal_names_and_original_input_units() -> None:
    row = signal_metric_row(np.array([[1.0, 3.0], [2.0, 4.0]]), np.array([[2.0, 5.0], [2.0, 6.0]]), np.ones((2, 2), dtype=bool), min_pixels=1)

    assert row["units"] == "original_input_intensity"
    assert row["MAE_signal"] == pytest.approx(1.25)
    assert "MAE_ms" not in row and "RMSE_ms" not in row


def test_k8_export_is_seeded_and_writes_unambiguous_artifact_schema(tmp_path) -> None:
    sources = []
    for stack in ("sax", "2ch", "4ch"):
        source = tmp_path / stack / "observations.npz"; source.parent.mkdir(); _prepared(source); sources.append(source)
    target = tmp_path / "output"; model = _ToyPSFModel()

    paths = export_k8_reprojections(model, _IdentitySpace(), sources, target, evaluation_seed=20260911)

    artifact = paths["sax"]
    assert artifact.name == "signal_reprojection_sax_K8.npz"
    with np.load(artifact, allow_pickle=False) as data:
        assert {"observed", "predicted", "residual", "group_idx", "weight_idx", "timing9_ms", "masks", "tr_ms", "vps"}.issubset(data.files)
    assert set(paths) == {"sax", "2ch", "4ch"}
    assert set(model.n_samples_seen) == {8}


def test_nonempty_output_is_never_overwritten(tmp_path) -> None:
    output = tmp_path / "existing"; output.mkdir(); (output / "old.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty"):
        prepare_empty_output(output)
