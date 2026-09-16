from pathlib import Path

import numpy as np
import torch

from modules.module_03_dataset_geometry.quantitative_point_dataset import QuantPointDataset
from modules.module_05_signal_decoder.mlp_model import MdmSignalMLP


def write_observations(path: Path) -> QuantPointDataset:
    affine = np.eye(4); affine[:3, 0] = [0, 2, 0]; affine[:3, 1] = [1, 0, 0]; affine[:3, 2] = [0, 0, 6]
    images = np.stack([np.full((3, 3), 0.2 + 0.01 * weight, dtype=np.float32) for weight in range(10)])
    np.savez_compressed(path, images=images, masks=np.ones_like(images, dtype=bool), group_idx=np.zeros(10, dtype=np.int64), weight_idx=np.arange(10, dtype=np.int64), stack_idx=np.zeros(10, dtype=np.int64), acquisition_time_ms=np.arange(10, dtype=np.float64), timing9_ms=np.full((10, 9), 70.0), affine_lps_rc=np.repeat(affine[None], 10, axis=0), pixel_spacing_rc_mm=np.repeat([[2.0, 1.0]], 10, axis=0), slice_thickness_mm=np.full(10, 6.0), tr_ms=np.full(10, 3.2), vps=np.full(10, 32, dtype=np.int64))
    return QuantPointDataset([path])


def write_functional_checkpoint(path: Path, timing_min=60.0, timing_max=80.0) -> None:
    model = MdmSignalMLP()
    torch.save({"state_dict": model.state_dict(), "architecture": "12-200-200-200-10", "input_normalization": "T1/1000,T2/1000,B1,timing9/1000", "output_normalization": "raw_l2_normalized", "protocol_hhz_v1": {"tr_ms": 3.2, "vps": 32, "fa_deg": [45.0, 45.0, 45.0], "ti_ms": [50.0, 150.0], "t2prep_ms": [35.0, 45.0, 55.0], "n_ramp_up": 10}, "parameter_ranges": {"t1_ms": [20, 2500], "t2_ms": [5, 200], "b1": [0.1, 1.2], "constraint": "T1>T2"}, "functional_fixture": True, "formal_candidate": False, "validation_status": "functional_smoke", "scientific_checkpoint": False, "timing9_min_ms": [timing_min] * 9, "timing9_max_ms": [timing_max] * 9}, path)


def recon_config(checkpoint: Path) -> dict:
    return {"inr": {"coarsest_resolution_mm": 12.0, "finest_resolution_mm": 6.0, "level_scale": 1.5, "n_features_per_level": 2, "log2_hashmap_size": 8, "latent_features": 8, "width": 12, "depth": 1}, "training": {"device": "cpu", "seed": 9, "spatial_scaling": 30.0, "batch_size": 10, "psf_samples": 1, "stage_a_iterations": 2, "stage_b_iterations": 2, "learning_rates": {"encoding": 0.001, "network": 0.001, "rigid": 0.0001, "weight_decay": 0.01}, "scheduler_milestones": [0.75], "scheduler_gamma": 0.5}, "loss": {"transformation": 0.001, "quantitative": {"t1": 0.0, "t2": 0.0, "b1": 0.0}}, "export": {"output_resolution_mm": 12.0, "output_batch_size": 256}, "decoder": {"checkpoint": str(checkpoint), "allow_functional_fixture_checkpoint": True}}
