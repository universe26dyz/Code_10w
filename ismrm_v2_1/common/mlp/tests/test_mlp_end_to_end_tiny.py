import torch

from modules.module_07_objective_training.mlp_trainer import build_mlp_training_model, load_mlp_reconstruction_checkpoint, train_mlp_reconstruction
from modules.module_08_inference_export.export_quantitative import export_quantitative_outputs
from modules.module_09_qc_benchmark.qc import validate_smoke_outputs
from online_helpers import recon_config, write_functional_checkpoint, write_observations


def test_tiny_mlp_reconstruction_checkpoint_export_and_qc(tmp_path):
    dataset = write_observations(tmp_path / "observations.npz")
    source = tmp_path / "checkpoint.pth"; write_functional_checkpoint(source)
    config, output = recon_config(source), tmp_path / "outputs"
    result = train_mlp_reconstruction(dataset, config, "configs/protocol_hhz_v1.yaml", output)
    model, space = result["model"], result["training_space"]
    assert model.rigid_psf.axisangle.grad is not None and torch.isfinite(model.rigid_psf.axisangle.grad).all()
    clone, _, _, _ = build_mlp_training_model(dataset, config, "configs/protocol_hhz_v1.yaml", torch.device("cpu"))
    load_mlp_reconstruction_checkpoint(output / "model.pt", clone, torch.device("cpu"))
    export_quantitative_outputs(model, space, output, 12.0, 256)
    report = validate_smoke_outputs(output)
    assert report["rigid_tensor"] and report["deformable"] is False
