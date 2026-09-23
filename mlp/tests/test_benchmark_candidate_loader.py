import torch

from mlp.modules.module_05_signal_decoder.checkpoint_loader import load_frozen_mlp_decoder_for_benchmark
from online_helpers import write_functional_checkpoint, write_observations


def test_benchmark_loader_accepts_unvalidated_nonfunctional_formal_candidate(tmp_path):
    dataset = write_observations(tmp_path / "observations.npz")
    checkpoint = tmp_path / "candidate.pth"
    write_functional_checkpoint(checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    payload.update({"functional_fixture": False, "scientific_checkpoint": True, "formal_candidate": True, "validation_status": "unvalidated"})
    torch.save(payload, checkpoint)
    decoder = load_frozen_mlp_decoder_for_benchmark(checkpoint, dataset, "mlp/configs/protocol_hhz_v1.yaml", device=torch.device("cpu"))
    assert decoder.training is False
