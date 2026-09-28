# Quality experiment ledger

## D1_BASELINE_PSF_SMOOTHNESS_V1

- Date: 2026-09-28
- Status: PLANNED
- Category: baseline diagnostic
- Scientific question / hypothesis: separate intrinsic direct-central-plane map smoothness from added map-domain PSF-induced smoothness without rerunning reconstruction.
- Baseline: CYJ `trad_B6` and `frozen_mlp_B6` from `CYJ_decoder_comparison_20260924_clean_v1`.
- Method: not yet run.
- Changed variable(s): none; read-only evaluation only.
- Unchanged controls: native reference provenance, native grids, map values, historical baseline outputs, and baseline source code.
- Code added: NOT AVAILABLE at initialization.
- Config added: NOT AVAILABLE at initialization.
- Commands actually run: NOT RUN at initialization.
- Local environment: `knesvr_torch`, CPU-only for validation.
- Server environment: `cr_dreme` (planned).
- Input artifacts: baseline central-plane exports, map-domain PSF comparisons, verified native reference, preprocessed provenance, and myocardium-mask bundle (planned).
- Output root: `/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/quality_experiments_v1/CYJ/D1_psf_smoothness_decomposition_baseline_v1/`.
- Output files: NOT RUN / NOT AVAILABLE.
- Key quantitative results: NOT RUN / NOT AVAILABLE.
- Visual findings: NOT RUN / NOT AVAILABLE.
- Interpretation: NOT RUN / NOT AVAILABLE.
- Limitations / blockers: real CYJ server data is unavailable locally.
- Git commit: NOT AVAILABLE at initialization.
- Next decision: implement and run the read-only D1 diagnostic.

## D2_BASELINE_SIGNAL_K8_V1

- Date: 2026-09-28
- Status: PLANNED
- Category: baseline diagnostic
- Scientific question / hypothesis: formal paired K=8 signal-domain evaluation.
- Baseline / method / changed variables / controls / code / config / commands / input artifacts / output files / quantitative results / visual findings / interpretation / git commit: NOT RUN / NOT AVAILABLE.
- Local environment: `knesvr_torch`, CPU-only for validation.
- Server environment: `cr_dreme` (planned).
- Output root: `/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/quality_experiments_v1/CYJ/D2_k8_signal_domain_baseline_v1/`.
- Limitations / blockers: server-only formal data.
- Next decision: start only after D1 completion.

## D3_BASELINE_SIGNAL_DICTIONARY_V1

- Date: 2026-09-28
- Status: PLANNED
- Category: baseline diagnostic
- Scientific question / hypothesis: PSF-integrated signal to exact native dictionary-fit apparent-map evaluation.
- Baseline / method / changed variables / controls / code / config / commands / input artifacts / output files / quantitative results / visual findings / interpretation / git commit: NOT RUN / NOT AVAILABLE.
- Local environment: `knesvr_torch`, CPU-only for validation.
- Server environment: `cr_dreme` (planned).
- Output root: `/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/quality_experiments_v1/CYJ/D3_signal_psf_dictionary_fit_baseline_v1/`.
- Limitations / blockers: exact native dictionary-fitting provenance must be found; server-only formal data.
- Next decision: start only after D2 completion.

## D1_BASELINE_PSF_SMOOTHNESS_V1 — implementation update

- Date: 2026-09-28
- Status: CODE_READY_SERVER_RUN_PENDING
- Method: read-only central-plane and map-domain-PSF artifact loader; method-specific and strict four-prediction supports; agreement/detail metrics; direct PSF−central increments; display-only figures.
- Changed variable(s): diagnostic support definition and evaluation mode only; no reconstruction variable changes.
- Unchanged controls: baseline code/results and map values remain read-only; native grids are not resized, padded, cropped, or stacked across heterogeneous stacks.
- Code added: `quality_experiments/D1_psf_smoothness_decomposition/{io.py,metrics.py,figures.py,run_d1_psf_smoothness_decomposition.py,test_d1_metrics.py,METRIC_DEFINITIONS.md}` and `quality_experiments/server_commands/run_D1_CYJ.sh`.
- Config added: none.
- Commands actually run: recorded in `PHASE_D1_COMPLETION.md`.
- Local environment: explicit `conda run -n knesvr_torch`, CPU-only.
- Server environment: `cr_dreme` command generated, not run.
- Input artifacts: exact paths are checked by generated server command and recorded in formal manifest.
- Output files: formal output NOT RUN / NOT AVAILABLE.
- Key quantitative results: NOT RUN / NOT AVAILABLE.
- Visual findings: NOT RUN / NOT AVAILABLE.
- Interpretation: NOT RUN / NOT AVAILABLE.
- Limitations / blockers: local environment cannot access `/data/dengyz/...`; formal evaluation awaits server execution.
- Git commit: `ae111b87e7d247c754ac64a658e3d7c33978b207` at implementation start; no commit created.
- Next decision: run `server_commands/run_D1_CYJ.sh` on the formal server, then update this entry with actual results.
