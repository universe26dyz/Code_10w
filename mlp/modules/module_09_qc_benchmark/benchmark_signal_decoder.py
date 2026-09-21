"""Fair, decoder-only Trad-versus-frozen-MLP benchmark on one timing pool."""

from __future__ import annotations

import argparse
import csv
import json
import resource
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from modules.module_05_signal_decoder.checkpoint_loader import load_frozen_mlp_decoder_for_benchmark
from modules.module_05_signal_decoder.synthetic_dataset import _protocol, load_timing_pool, protocol_from_record
from modules.module_05_signal_decoder.trad_teacher.trad_signal_simulator import TradSignalSimulator


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(callable_: Any, device: torch.device, repetitions: int) -> list[float]:
    durations = []
    for _ in range(repetitions):
        _synchronize(device); start = time.perf_counter(); callable_(); _synchronize(device)
        durations.append(time.perf_counter() - start)
    return [value * 1000.0 for value in durations]


def _inputs(batch_size: int, timing9: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu"); generator.manual_seed(20260911)
    t2 = 5.0 + 195.0 * torch.rand(batch_size, generator=generator)
    t1 = t2 + (2500.0 - t2) * torch.rand(batch_size, generator=generator)
    b1 = 0.1 + 1.1 * torch.rand(batch_size, generator=generator)
    timing = torch.from_numpy(np.repeat(timing9[None], batch_size, axis=0).astype(np.float32))
    return t1.to(device), t2.to(device), b1.to(device), timing.to(device)


def _measure(model: torch.nn.Module, protocol: object, base: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], device: torch.device, repetitions: int) -> tuple[list[float], list[float], torch.Tensor]:
    def run_forward() -> torch.Tensor:
        return model(*base, protocol, normalize=True)
    run_forward()  # one warm-up
    forward = _timed(run_forward, device, repetitions)
    def run_forward_plus_gradient() -> None:
        t1, t2, b1, timing = (value.detach().clone().requires_grad_(index < 3) for index, value in enumerate(base))
        model(t1, t2, b1, timing, protocol, normalize=True).sum().backward()
    run_forward_plus_gradient()  # one warm-up
    forward_grad = _timed(run_forward_plus_gradient, device, repetitions)
    return forward, forward_grad, run_forward().detach()


def _gradient(model: torch.nn.Module, protocol: object, base: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    t1, t2, b1, timing = (value.detach().clone().requires_grad_(index < 3) for index, value in enumerate(base))
    output = model(t1, t2, b1, timing, protocol, normalize=True)
    return torch.stack([torch.stack((gradient[0] * 1000.0, gradient[1] * 1000.0, gradient[2]), dim=-1) for weight in range(10) for gradient in [torch.autograd.grad(output[:, weight].sum(), (t1, t2, b1), retain_graph=True)]], dim=1)


def benchmark_signal_decoder(checkpoint_path: str | Path, timing_pool_path: str | Path | None, protocol_path: str | Path, output_path: str | Path, device_name: str, batch_sizes: list[int], repetitions: int = 3, *, rr_dataset_dir: str | Path | None = None) -> list[dict[str, Any]]:
    if not batch_sizes or any(size < 1 for size in batch_sizes) or repetitions < 1 or repetitions > 3:
        raise ValueError("batch_sizes must be positive and timed repetitions must be 1..3.")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested benchmark device unavailable: {device}")
    if rr_dataset_dir is not None:
        rr_root=Path(rr_dataset_dir); meta=json.loads((rr_root/"dataset_metadata.json").read_text()); protocol=protocol_from_record(meta["protocol"],protocol_path); timing=np.asarray(np.load(rr_root/"rhythms.npz",allow_pickle=False)["timing9_ms"][0],dtype=np.float64); pool=None
        dataset=SimpleNamespace(timing=torch.from_numpy(timing[None].astype(np.float32)), validated_tr_vps=lambda: (protocol.tr_ms,protocol.vps))
    else:
        if timing_pool_path is None: raise ValueError("Provide --rr-dataset-dir or --timing-pool.")
        pool = load_timing_pool(timing_pool_path); protocol = _protocol(pool, protocol_path); timing = np.asarray(pool["timing9_ms"][0], dtype=np.float64)
        dataset = SimpleNamespace(timing=torch.from_numpy(np.asarray(pool["timing9_ms"], dtype=np.float32)), validated_tr_vps=lambda: (protocol.tr_ms, protocol.vps))
        if bool(pool["functional_fixture"]): raise ValueError("Standalone formal benchmark rejects functional-fixture timing pools.")
    mlp = load_frozen_mlp_decoder_for_benchmark(checkpoint_path, dataset, protocol_path, device=device)
    trad = TradSignalSimulator().to(device).eval(); mlp.eval()
    rows = []
    for batch_size in batch_sizes:
        base = _inputs(batch_size, timing, device)
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
        trad_forward, trad_grad, trad_output = _measure(trad, protocol, base, device, repetitions)
        trad_peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
        mlp_forward, mlp_grad, mlp_output = _measure(mlp, protocol, base, device, repetitions)
        mlp_peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        error = mlp_output - trad_output
        gradient_error = _gradient(mlp, protocol, base) - _gradient(trad, protocol, base)
        reference_gradient = _gradient(trad, protocol, base)
        med=lambda x: float(statistics.median(x)); p90=lambda x: float(np.percentile(x,90)); k={2560:4,5120:8,10240:16}.get(batch_size)
        row = {"effective_points":batch_size,"psf_k_equivalent":k,"repetitions":repetitions,"device":str(device),"trad_forward_ms_median":med(trad_forward),"trad_forward_ms_p90":p90(trad_forward),"trad_forward_grad_ms_median":med(trad_grad),"trad_forward_grad_ms_p90":p90(trad_grad),"trad_peak_gpu_bytes":trad_peak,"mlp_forward_ms_median":med(mlp_forward),"mlp_forward_ms_p90":p90(mlp_forward),"mlp_forward_grad_ms_median":med(mlp_grad),"mlp_forward_grad_ms_p90":p90(mlp_grad),"mlp_peak_gpu_bytes":mlp_peak,"forward_speedup":med(trad_forward)/med(mlp_forward),"forward_grad_speedup":med(trad_grad)/med(mlp_grad),"signal_overall_rmse":float(error.pow(2).mean().sqrt()),"signal_per_weight_rmse":error.pow(2).mean(0).sqrt().cpu().tolist(),"gradient_absolute_error":float(gradient_error.abs().mean()),"gradient_relative_error":float((gradient_error.abs()/reference_gradient.abs().clamp_min(1e-8)).mean()),"gradient_cosine_similarity":float(torch.nn.functional.cosine_similarity((gradient_error+reference_gradient).reshape(batch_size,-1),reference_gradient.reshape(batch_size,-1),dim=-1).mean()),"timing_source":"rr_dataset" if pool is None else "timing_pool","process_peak_cpu_rss_kib":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
        rows.append(row)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Benchmark output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in rows[0] if key != "signal_per_weight_rmse"]); writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if key != "signal_per_weight_rmse"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--timing-pool"); parser.add_argument("--rr-dataset-dir"); parser.add_argument("--protocol", required=True); parser.add_argument("--output", required=True); parser.add_argument("--device", required=True); parser.add_argument("--batch-size", type=int, action="append", required=True); parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(benchmark_signal_decoder(args.checkpoint, args.timing_pool, args.protocol, args.output, args.device, args.batch_size, args.repetitions,rr_dataset_dir=args.rr_dataset_dir), indent=2))


if __name__ == "__main__":
    main()
