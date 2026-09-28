# D2 K=8 signal-domain completion report

- Status: `CODE_READY_SERVER_RUN_PENDING`.
- Scope: final-checkpoint, read-only K=8 native weighted-signal inference and metrics only; no optimizer, training, reconstruction rerun, or baseline artifact modification.
- Existing path reused: stored `resolved_config` and training space, `QuantPointDataset`, baseline decoder builders/checkpoint loaders, and `export_native_plane_reprojections`; the D2 wrapper supplies `output_psf={enabled: true, n_samples: 8}` and evaluation seed `20260911`.
- Metrics: `signal_agreement_metrics()` in `original_input_intensity`, method-specific and strict paired support outputs, all requested stack/weight aggregations, and Bloch-vs-FrozenMLP predicted-signal disagreement.
- Local validation: targeted synthetic tests, package compilation, direct CLI help, and server-shell syntax check run in explicit `knesvr_torch` CPU environment. CUDA extension fallback warnings were observed during CLI import; no final CYJ checkpoint was loaded locally.
- Formal server command: `quality_experiments/server_commands/run_D2_CYJ.sh` on `cr_dreme`.
- Formal output root: `/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/quality_experiments_v1/CYJ/D2_k8_signal_domain_baseline_v1/`.
- Formal result: `FORMAL K=8 RESULT NOT RUN` locally; do not interpret historical K=1 signal reprojections as D2 results.
- PUSH PERFORMED: NO.
