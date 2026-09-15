import copy
import numpy as np
import torch

from modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset
from modules.module_07_objective_training.trad_trainer import build_training_model, load_checkpoint, train_trad
from modules.module_08_inference_export.export_quantitative import export_quantitative_outputs
from modules.module_09_qc_benchmark.qc import validate_smoke_outputs


def _dataset(path):
    affine = np.eye(4); affine[:3, 0] = [0, 2, 0]; affine[:3, 1] = [1, 0, 0]; affine[:3, 2] = [0, 0, 6]
    images = np.stack([np.full((3, 3), 0.2 + 0.01 * weight, dtype=np.float32) for weight in range(10)])
    np.savez_compressed(path, images=images, masks=np.ones_like(images, dtype=bool), group_idx=np.zeros(10, dtype=np.int64), weight_idx=np.arange(10, dtype=np.int64), stack_idx=np.zeros(10, dtype=np.int64), acquisition_time_ms=np.arange(10, dtype=np.float64), timing9_ms=np.full((10, 9), 70.0), affine_lps_rc=np.repeat(affine[None], 10, axis=0), pixel_spacing_rc_mm=np.repeat([[2.0, 1.0]], 10, axis=0), slice_thickness_mm=np.full(10, 6.0), tr_ms=np.full(10, 3.2), vps=np.full(10, 32, dtype=np.int64))
    return QuantPointDataset([path])


def _config():
    return {"inr": {"coarsest_resolution_mm": 12.0, "finest_resolution_mm": 6.0, "level_scale": 1.5, "n_features_per_level": 2, "log2_hashmap_size": 8, "latent_features": 8, "width": 12, "depth": 1}, "training": {"device": "cpu", "seed": 9, "spatial_scaling": 30.0, "batch_size": 10, "psf_samples": 1, "stage_a_iterations": 2, "stage_b_iterations": 2, "intensity_normalization": {"enabled": True, "method": "trimmed_mean", "lower_quantile": 0.1, "upper_quantile": 0.9}, "learning_rates": {"encoding": 0.001, "network": 0.001, "rigid": 0.0001, "weight_decay": 0.01}, "scheduler_milestones": [0.75], "scheduler_gamma": 0.5}, "loss": {"transformation": 0.001, "quantitative": {"t1": 0.0, "t2": 0.0, "b1": 0.0}}, "export": {"output_resolution_mm": 12.0, "output_batch_size": 256}}


def test_tiny_staged_training_checkpoint_export_and_qc(tmp_path):
    dataset = _dataset(tmp_path / "observations.npz"); config = _config(); output = tmp_path / "outputs"
    result = train_trad(dataset, config, "configs/protocol_hhz_v1.yaml", output)
    model, space = result["model"], result["training_space"]
    torch.testing.assert_close(result["stage_a_axisangle_final"], model.rigid_psf.axisangle_init)
    assert model.rigid_psf.axisangle.grad is not None and torch.isfinite(model.rigid_psf.axisangle.grad).all()
    assert torch.isfinite(model.rigid_psf.transformation_loss(space.spatial_scaling))
    batch = {"xyz": space.local_to_train(dataset.xyz[:10]), "v": dataset.v[:10], "group_idx": dataset.group_idx[:10], "weight_idx": dataset.weight_idx[:10], "stack_idx": dataset.stack_idx[:10], "timing": dataset.timing[:10]}
    expected = model(batch, 1).detach()
    clone, _, _ = build_training_model(dataset, config, "configs/protocol_hhz_v1.yaml", torch.device("cpu"))
    load_checkpoint(output / "model.pt", clone, torch.device("cpu"))
    torch.testing.assert_close(clone(batch, 1), expected)
    export_quantitative_outputs(model, space, output, 12.0, 256)
    report = validate_smoke_outputs(output)
    assert report["rigid_tensor"] and report["deformable"] is False
    forbidden = ("deform", "b_net", "sigma_net", "log_var_slice", "logit_coef", "slice_scale", "weight_scale")
    assert all(not any(token in name.lower() for token in forbidden) for name in model.state_dict())
