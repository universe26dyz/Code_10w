import numpy as np
import torch

from modules.module_07_objective_training.trad_trainer import build_training_model, load_checkpoint, train_trad


def _dataset(path):
    affine = np.eye(4); affine[:3, 0] = [0, 2, 0]; affine[:3, 1] = [1, 0, 0]; affine[:3, 2] = [0, 0, 6]
    np.savez_compressed(path, images=np.full((10, 2, 2), 0.2, dtype=np.float32), masks=np.ones((10, 2, 2), dtype=bool), group_idx=np.zeros(10, dtype=np.int64), weight_idx=np.arange(10, dtype=np.int64), stack_idx=np.zeros(10, dtype=np.int64), acquisition_time_ms=np.arange(10, dtype=np.float64), timing9_ms=np.full((10, 9), 70.0), affine_lps_rc=np.repeat(affine[None], 10, axis=0), pixel_spacing_rc_mm=np.repeat([[2.0, 1.0]], 10, axis=0), slice_thickness_mm=np.full(10, 6.0), tr_ms=np.full(10, 3.2), vps=np.full(10, 32, dtype=np.int64))
    from modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset
    return QuantPointDataset([path])


def _config():
    return {"inr": {"coarsest_resolution_mm": 12.0, "finest_resolution_mm": 6.0, "level_scale": 1.5, "n_features_per_level": 2, "log2_hashmap_size": 8, "latent_features": 8, "width": 12, "depth": 1}, "training": {"device": "cpu", "seed": 2, "spatial_scaling": 30.0, "batch_size": 10, "psf_samples": 1, "stage_a_iterations": 1, "stage_b_iterations": 1, "learning_rates": {"encoding": 0.001, "network": 0.001, "rigid": 0.0001, "weight_decay": 0.01}, "scheduler_milestones": [0.75], "scheduler_gamma": 0.5}, "loss": {"transformation": 0.001, "quantitative": {"t1": 0.0, "t2": 0.0, "b1": 0.0}}, "export": {"output_resolution_mm": 12.0, "output_batch_size": 32}}


def test_checkpoint_contains_required_training_space_and_protocol_metadata(tmp_path):
    dataset = _dataset(tmp_path / "observations.npz")
    output = tmp_path / "outputs"
    train_trad(dataset, _config(), "configs/protocol_hhz_v1.yaml", output)
    model, _, _ = build_training_model(dataset, _config(), "configs/protocol_hhz_v1.yaml", torch.device("cpu"))
    checkpoint = load_checkpoint(output / "model.pt", model, torch.device("cpu"))
    assert {"center_ras_mm", "spatial_scaling"}.issubset(checkpoint["training_space"])
    assert checkpoint["validated_tr_ms"] == 3.2 and checkpoint["validated_vps"] == 32
