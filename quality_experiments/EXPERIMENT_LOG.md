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

## D1_BASELINE_PSF_SMOOTHNESS_V1 — failed server attempt and v2 hotfix

- Date: 2026-09-28
- First formal server attempt status: FAILED_PRECOMPUTE_INPUT_SCHEMA_VALIDATION.
- First formal server attempt reason: the v1 D1 loader incorrectly required a one-row-per-group central-plane archive. The historical baseline exporter stores ten observation rows per spatial group, one for each weight `0..9`.
- Formal CYJ metrics generated: NO.
- Failed result root: `D1_psf_smoothness_decomposition_baseline_v1`; it is an incomplete failed attempt and must not be overwritten or interpreted as a formal result.
- Hotfix status: CODE_READY_SERVER_RUN_PENDING.
- Hotfix method: require `weight_idx`; validate complete zero-based groups, exactly ten rows and weights `0..9` per group; select the unique weight-0 T1/T2 row; AND all ten masks; retain exact native shape.
- Implementation revision / corrected output root: `D1_psf_smoothness_decomposition_baseline_v2`.
- Local environment: explicit `knesvr_torch`, CPU-only.
- Local formal inputs: NOT AVAILABLE; no CYJ result was run locally.
- Local regression result: 13 targeted tests passed before final hotfix-policy verification.
- Server command: `quality_experiments/server_commands/run_D1_CYJ.sh`; it records the Git SHA, tees the full log, rejects non-empty `baseline_v2`, and never deletes v1.
- Key quantitative results / visual findings / interpretation: NOT RUN / NOT AVAILABLE.
- Next decision: commit the validated hotfix locally; user pushes the commit, server pulls it, then user runs the corrected D1 v2 server script.
- Local hotfix implementation commit: `43b0424` (`Fix D1 central-plane observation grouping`).
- PUSH PERFORMED: NO.

## G0_POSE_ERROR_CORRELATION_CYJ_V1

- Date: 2026-09-28
- Status: CODE_READY_SERVER_RUN_PENDING.
- Category: independent read-only post-hoc diagnostic; it does not change the G0 correspondence audit or `next_phase: D2`.
- Scientific question: whether corrected G0 v2 translation/rotation drift is associated with central-map `relative_rmse` or `1-correlation` across local groups.
- Required provenance: `G0_geometry_correspondence_baseline_v2/manifest.json` with its matching revision, `metrics/G0_per_slice.csv`, and `metrics/G0_pose_drift.csv`; join is strictly `stack + expected_group == group_idx`.
- Method: six stack/parameter combinations × four Pearson/Spearman associations; Spearman p-values receive one 24-test BH-FDR correction; leave-one-group-out rank sensitivity is recorded.
- Code/config: `quality_experiments/G0_pose_error_correlation/`; synthetic contracts only, explicit `knesvr_torch` CPU validation.
- Actual local artifact search: `/home/universe/SVR/multimap_postprogramming` contained no complete corrected G0 v2 bundle. `FORMAL_LOCAL_ANALYSIS = NOT_RUN_MISSING_G0_V2_ARTIFACTS`.
- Formal CYJ metrics, figures, conclusions: NOT RUN / NOT AVAILABLE. Missing required local paths are `metrics/G0_per_slice.csv` and `metrics/G0_pose_drift.csv` beneath a verified v2 bundle.
- Next decision: copy a verified corrected G0 v2 bundle locally and invoke the documented CLI; no reconstruction or follow-on phase is authorized by this diagnostic.

## G0_GEOMETRY_CORRESPONDENCE_BASELINE_V1 — pose-semantics hotfix

- Date: 2026-09-28
- G0 v1 formal correspondence status: VALID. For T1/T2 × SAX/2CH/4CH, diagonal assignment fraction was 1.0; non-diagonal and non-identity-orientation counts were 0.
- G0 v1 pose audit status: INVALID. It interpreted NeSVoR axis-angle vectors as `[translation, rotation]`; the vendored transform contract is `[rotation-vector, translation]`.
- Consequence: G0 v1 `translation_magnitude_mm`, `rotation_magnitude_deg`, `final_center_ras_mm`, `slice_normal_ras`, and `neighbor_center_distance_mm` must not be interpreted.
- G0 v2 status: CODE_READY_SERVER_RUN_PENDING.
- Hotfix: use vendored `RigidTransform(initial).inv().compose(final)` for relative motion and `ax_transform_points` for final centers/normals; no correspondence/Hungarian/signal logic changed.
- Formal v2 output root: `/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/quality_experiments_v1/CYJ/G0_geometry_correspondence_baseline_v2/`.
- Local formal G0 rerun: NOT RUN / NOT AVAILABLE; server-only CYJ artifacts are not available locally.
- Local regression result: 12 targeted tests passed before final validation.
- Next decision: commit the hotfix locally; user pushes, server pulls, then user runs the corrected G0 v2 script.
- Local hotfix implementation commit: `df336eb` (`Fix G0 pose audit transform semantics`).
- PUSH PERFORMED: NO.

## G0_GEOMETRY_CORRESPONDENCE_BASELINE_V1

- Date: 2026-09-28
- Status: CODE_READY_SERVER_RUN_PENDING
- Category: read-only geometry/correspondence diagnostic
- Scientific question / hypothesis: determine whether central/no-PSF reconstructed groups correspond to the same-index native groups, and quantify discrete orientation, assignment ambiguity, and available pose drift without changing official metrics.
- Baseline: CYJ FrozenMLP B6.
- Single intended change: independent content/orientation audit only.
- Held-fixed controls: FrozenMLP checkpoint, baseline run outputs, native reference provenance, pixel grids, official `g ↔ g` pairing, and all historical code/results.
- Code/config: `quality_experiments/G0_geometry_correspondence/` and `server_commands/run_G0_CYJ.sh`; no baseline config changed.
- Actual local commands: targeted synthetic tests, compile/CLI validation, and shell syntax validation; no formal CYJ inputs are local.
- Local/server environments: `knesvr_torch` CPU / `cr_dreme` server.
- Input provenance: verified native-reference bundle plus FrozenMLP B6 central-plane archives; optional D2-compatible signal artifacts are absent, so `G0-signal = DEPENDENCY_PENDING`.
- Output root: `/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/quality_experiments_v1/CYJ/G0_geometry_correspondence_baseline_v1/`.
- Metrics/visuals/interpretation: formal result NOT RUN / NOT AVAILABLE.
- Ruling: use the established baseline `_extract_quantitative_stack` to preserve the historical 10-weight central-plane semantics; cost if wrong is an audit incompatible with D1/baseline exports.
- Next decision: run the generated server command after this phase commit; do not start a later v2 phase automatically.
- Local implementation commit: `5d803ab` (`Add G0 geometry correspondence audit`).
- PUSH PERFORMED: NO.
