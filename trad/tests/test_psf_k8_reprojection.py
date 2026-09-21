import numpy as np
import torch

from modules.module_06_rigid_psf.rigid_psf_forward import GroupRigidPSF
from modules.module_08_inference_export.reprojection import export_native_plane_reprojections


class _IdentitySpace:
    @staticmethod
    def local_to_train(value):
        return value


class _ToyPSFModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.rigid_psf = GroupRigidPSF(torch.zeros((1, 6)), torch.tensor([[1.0, 1.0, 8.0]]))
        self.register_buffer("intensity_scale", torch.ones(()))
        self.n_samples_seen = []

    def forward(self, batch, n_samples):
        self.n_samples_seen.append(n_samples)
        samples = self.rigid_psf.sample_local_then_transform(batch["xyz"], batch["group_idx"], n_samples)
        return samples[..., 2].square().mean(dim=1) + self.anchor

    @staticmethod
    def inr(world):
        shape = world.shape[:-1]
        return {"t1_ms": torch.full(shape, 1000.0, device=world.device), "t2_ms": torch.full(shape, 50.0, device=world.device)}


def _observations(path):
    images = np.zeros((10, 2, 2), dtype=np.float32)
    np.savez_compressed(
        path,
        images=images,
        masks=np.ones_like(images, dtype=bool),
        group_idx=np.zeros(10, dtype=np.int64),
        weight_idx=np.arange(10, dtype=np.int64),
        timing9_ms=np.ones((10, 9), dtype=np.float64),
        tr_ms=np.ones(10),
        vps=np.ones(10, dtype=np.int64),
        pixel_spacing_rc_mm=np.ones((10, 2), dtype=np.float64),
    )


def test_k8_reprojection_is_seeded_and_deterministic(tmp_path):
    source = tmp_path / "stack" / "observations.npz"
    source.parent.mkdir()
    _observations(source)
    first = tmp_path / "first"; first.mkdir()
    second = tmp_path / "second"; second.mkdir()
    third = tmp_path / "third"; third.mkdir()
    model = _ToyPSFModel()
    kwargs = {"output_psf": {"enabled": True, "n_samples": 8}, "export_parameter_maps": False}

    export_native_plane_reprojections(model, _IdentitySpace(), [source], first, evaluation_seed=123, **kwargs)
    export_native_plane_reprojections(model, _IdentitySpace(), [source], second, evaluation_seed=123, **kwargs)
    export_native_plane_reprojections(model, _IdentitySpace(), [source], third, evaluation_seed=124, **kwargs)

    a = np.load(first / "signal_reprojection_stack.npz")["predicted"]
    b = np.load(second / "signal_reprojection_stack.npz")["predicted"]
    c = np.load(third / "signal_reprojection_stack.npz")["predicted"]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert set(model.n_samples_seen) == {8}
