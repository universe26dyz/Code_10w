# D1 completion report

- Status: CODE_READY_SERVER_RUN_PENDING
- Git HEAD: `ae111b87e7d247c754ac64a658e3d7c33978b207`
- Files added: D1 read-only loader, metrics, figures, runner, targeted synthetic tests, metric definitions, root framework/ledger/state, and `server_commands/run_D1_CYJ.sh`.
- Files modified: none in `trad/`, `mlp/`, `reconstruction_core/`, `config/`, or `docs/`.
- Files intentionally not modified: all baseline source code, configurations, and historical server result artifacts.
- Local commands run: Phase 0 audit; `conda run --no-capture-output -n knesvr_torch python -m compileall -q quality_experiments/D1_psf_smoothness_decomposition`; `MPLCONFIGDIR=/tmp/d1-matplotlib conda run --no-capture-output -n knesvr_torch python -m pytest -q quality_experiments/D1_psf_smoothness_decomposition/test_d1_metrics.py`; `conda run --no-capture-output -n knesvr_torch python quality_experiments/D1_psf_smoothness_decomposition/run_d1_psf_smoothness_decomposition.py --help`; and `bash -n quality_experiments/server_commands/run_D1_CYJ.sh`.
- Tests run: targeted synthetic tests validate identity agreement, blur-related gradient/HF reduction, strict four-way support, overwrite protection, heterogeneous stack isolation, figure range contract, and PSF-increment aggregation.
- Test results: PASS — 8 targeted tests passed; compilation, CLI help, and shell syntax checks passed.
- Server commands generated: `quality_experiments/server_commands/run_D1_CYJ.sh`.
- Real data available locally: no; `/data/dengyz/...` was not accessed.
- Formal result generated: NO. Reason: local Codex has no access to server `/data` and formal CYJ execution was not run.
- Known blockers: formal server run pending in `cr_dreme`.
- Scientific caveats: D1 quantifies map-domain parameter PSF, not signal-domain PSF/dictionary effects; those are reserved for D2/D3. Main conclusions should use the explicit strict four-prediction support.
- Next phase: D2, only after this code-ready D1 state is accepted and its server output is available as needed.
