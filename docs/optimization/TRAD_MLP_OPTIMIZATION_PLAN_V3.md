# Trad 6k and broad-RR MLP optimization plan

## Measured target

The CYJ 10k profiler reports median whole iterations of 477 ms (A) and 494 ms
(B). Explicit Bloch forward is the largest measured forward section (about
112 ms), while monolithic backward is about 72% of a whole iteration. This
motivates decoder forward/gradient diagnostics, but does not prove or claim an
end-to-end acceleration. A 2k+4k run is an estimate of roughly 49 minutes at
the baseline per-iteration rate, not a guarantee.

## Fixed scientific contract

Trad remains direct Bloch reconstruction with the existing INR, HB1 stack
initialization, intensity normalization and physical K=8 PSF baseline. MLP
remains 12D `[T1/1000,T2/1000,B1,timing9/1000]` to 10D L2-normalized
fingerprint, uses the exact tensorized `TradSignalSimulator`, and does not use
EPG or add VPS as an input. T1/T2/B1 ranges remain 20–2500 ms, 5–200 ms, and
0.1–1.2 with T1>T2.

## Experiments and evaluation

`trad/configs/experiments_6k/` defines B6 (2k+4k) plus TV2, TV3, PSF4, PSF16,
VAR_S, VAR_P, AMP_EDGE and NP512. All inherit the same B6 budget and have
separate output-root labels. Wave order is B6/TV2/TV3, then PSF4/PSF16;
variance, amplitude guidance and NP512 are second-wave only. Screen with fixed
monitor MSE, profiler summary, final poses, maps, physics-consistent signal
reprojection, apparent maps and primary `myocardium_core_1px` metrics.

`REG_FULL.yaml` is intentionally blocked: the checkout contains no proven
full/pre-crop processed HB1 source plus exact full-to-crop group geometry and
MIND/MP-PCA provenance. Training images must not be resampled or modified to
invent one.

Signal metrics now use schema-v2 field names `MAE_signal`, `RMSE_signal`, and
`bias_signal` with `units=original_input_intensity`; quantitative maps retain
millisecond metric names. `trad.evaluation.profiling.summarize_timing_profile`
writes CSV/JSON N, mean, median, std, p10, p90, whole-iteration fraction and a
configured-iteration time estimate.

## mDM-inspired RR domain

The new RR generator samples mean HR 20–120 bpm and RR CV 0–50%, creates nine
normally distributed RR intervals per rhythm, rejects invalid sequence timing,
and calls the canonical `compute_duration_before_acq`. This borrows the broad,
coherent rhythm-domain idea from mDM, but is explicitly not byte-for-byte mDM.
Configs provide smoke (8×16), pilot (500×500), and formal (5000×2500) scales,
with disjoint train/valid/test rhythm IDs. Signal-only datasets set
`store_jacobian=false`, so no Jacobian tensor is calculated or stored.

## Commands (server; do not batch launch)

```bash
cd /data/dengyz/code/Code_10w/trad
conda run -n cr_dreme python -m scripts.run_training --config configs/experiments_6k/B6.yaml --protocol configs/protocol_hhz_v1.yaml --observations /data/dengyz/dataset/Code_10w_v1/prepared/CYJ/sax/observations.npz --observations /data/dengyz/dataset/Code_10w_v1/prepared/CYJ/lax/observations.npz --output-dir /data/dengyz/dataset/Code_10w_v1/Code_10w_runs/trad_v3/B6
conda run -n cr_dreme python -m scripts.run_training --config configs/experiments_6k/TV2.yaml --protocol configs/protocol_hhz_v1.yaml --observations /data/dengyz/dataset/Code_10w_v1/prepared/CYJ/sax/observations.npz --observations /data/dengyz/dataset/Code_10w_v1/prepared/CYJ/lax/observations.npz --output-dir /data/dengyz/dataset/Code_10w_v1/Code_10w_runs/trad_v3/TV2
conda run -n cr_dreme python -m scripts.run_training --config configs/experiments_6k/TV3.yaml --protocol configs/protocol_hhz_v1.yaml --observations /data/dengyz/dataset/Code_10w_v1/prepared/CYJ/sax/observations.npz --observations /data/dengyz/dataset/Code_10w_v1/prepared/CYJ/lax/observations.npz --output-dir /data/dengyz/dataset/Code_10w_v1/Code_10w_runs/trad_v3/TV3
```

```bash
cd /data/dengyz/code/Code_10w/mlp
conda run -n cr_dreme python -m modules.module_05_signal_decoder.generate_rr_mlp_dataset --config configs/rr_synthetic/smoke.yaml --output-dir /data/dengyz/dataset/Code_10w_v1/mlp_rr_v3/smoke
conda run -n cr_dreme python -m modules.module_05_signal_decoder.generate_rr_mlp_dataset --config configs/rr_synthetic/pilot_signalonly.yaml --output-dir /data/dengyz/dataset/Code_10w_v1/mlp_rr_v3/pilot
```

The formal 12.5M generation, formal MLP training, Wave 3, and online MLP
reconstruction remain manual scientific gates. Dictionary-cache equivalence
and the fair exported-volume map-domain PSF comparison also remain gates; they
are not represented as completed results in this implementation.
