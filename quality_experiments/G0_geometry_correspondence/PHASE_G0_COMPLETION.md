# G0 geometry/correspondence audit completion report

- Status: `CODE_READY_SERVER_RUN_PENDING`
- Git implementation commit: `5d803ab` — `Add G0 geometry correspondence audit`.
- Files changed: standalone discrete-orientation/Hungarian correspondence engine, optional weight-0 signal adapter, pose-drift reader, visualizations, runner, targeted tests, server command, ledger entry, and phase state.
- Baseline files intentionally not modified: `trad/`, `mlp/`, `reconstruction_core/`, baseline configurations, and all historical results.
- G0-map semantics: direct central/no-PSF T1/T2 only. The established baseline `_extract_quantitative_stack` is reused for the observation-level (10-weight) central export schema.
- Official pairing: native `g ↔ g` remains unchanged. Hungarian matching is output solely as `AUDIT BEST MATCH`.
- G0-signal: `DEPENDENCY_PENDING`; no D2-compatible formal signal artifacts are available locally. The adapter validates/selects only unique weight-0 rows when they become available.
- Local real CYJ data: unavailable. Formal result: NOT RUN / NOT AVAILABLE.
- Local validation: `conda run -n knesvr_torch python -m compileall -q quality_experiments/G0_geometry_correspondence` passed; targeted G0 pytest passed (9 tests); direct runner `--help` passed; `bash -n quality_experiments/server_commands/run_G0_CYJ.sh` passed.
- Corrected server output root: `/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/quality_experiments_v1/CYJ/G0_geometry_correspondence_baseline_v1/`.
- Final Git status after implementation commit: clean before this completion-record update.
- PUSH PERFORMED: NO.
- Next action: user manually pushes the local commit, server pulls it, then user runs `quality_experiments/server_commands/run_G0_CYJ.sh`.
