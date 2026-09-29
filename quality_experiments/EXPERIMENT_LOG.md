# Quality experiment ledger

## Q001_FULL_FOV_PREPARATION_V1

- Date: 2026-09-29
- Status: CODE_READY_LOCAL_INPUT_ROOT_REQUIRED.
- Method: independent full-FOV DICOM sort, MIND-to-HB1, MP-PCA, v7.3 provenance MAT, and a validated Q001 adapter over the existing SOP-resolved bridge; no reconstruction.
- Baseline contract: `preprocess_stack_v1` and all baseline bridge code remain unchanged. Q001 does not crop or pad images and does not require dimensions divisible by four.
- Local availability: MATLAB plus CYJ sax/2ch/4ch manifest-resolved DICOM directories are present. A non-example deployment-path configuration with an authorized new Q001 output root is absent, so real preprocessing was not run.
- Code: `quality_experiments/Q001_full_fov/`, including `run_Q001_full_fov_local_CYJ.sh` which requires an explicit new `Q001_OUTPUT_ROOT` and refuses overwrite.
- Formal/real result: NOT RUN; full-FOV prepared inputs NOT AVAILABLE.
- Next decision: generate and QC full-FOV inputs at an explicit new root, then review signal-domain fidelity before any reconstruction phase.
- Local implementation commit: `41331bd` (`Add Q001 full-FOV preprocessing and bridge`). PUSH PERFORMED: NO.
- Pre-run MATLAB helper-path hotfix: `2d59356` (`Fix Q001 MATLAB helper path`); formal/real result remains NOT RUN.

## Q001_FULL_FOV_PREPARATION_V1 — pre-run acceptance hotfix

- Date: 2026-09-29
- Status: CODE_READY_LOCAL_INPUT_ROOTS_REQUIRED. FORMAL/REAL CYJ FULL-FOV INPUT GENERATION: NOT RUN.
- Root cause fixed: the MATLAB metadata previously asserted MP-PCA semantics even when MP-PCA was disabled, and the runner did not make cropped-vs-full correspondence QC an acceptance gate.
- Fix: disabled MP-PCA now records `MIND_mag_reg`; the formal bridge accepts only exact `MP-PCA(full-FOV MIND_mag_reg)` and the explicit HB1 full-FOV geometry rule. Per-stack strict QC rejects spatial/SOP/timing/TR-VPS/spacing/thickness/group/observation/affine-origin failures while retaining pixel equality as NOT REQUIRED.
- Runner contract: requires user-selected `Q001_OUTPUT_ROOT` and `Q001_BASELINE_PREPARED_ROOT`; validates `CYJ/{sax,2ch,4ch}` baseline artifacts before any preprocessing; creates read-only comparison reports and `q001_input_manifest.json` with per-stack paths, hashes, shapes, provenance, and readiness status.
- Local validation: 12 Q001 targeted tests passed; 4 `trad` and 4 `mlp` baseline bridge/geometry tests passed in their respective project roots; Q001 Python compile/CLI, shell syntax, MATLAB `checkcode`, and a synthetic MP-PCA-disabled MATLAB provenance check passed.
- Formal result: no CYJ preprocessing, prepared inputs, reconstruction, D3, or Q002 was run. Waiting only for explicit output and baseline-prepared roots.
- Local implementation commit: `34b1e0b` (`Harden Q001 full-FOV input acceptance`). PUSH PERFORMED: NO.

## Q001_FULL_FOV_PREPARATION_V1 — failed real input-generation attempt and MATLAB hotfix

- Date: 2026-09-29
- Failed root (preserved, never overwritten): `/home/universe/SVR/data/Code_10w_v1/Q001_full_fov_inputs_v1`.
- Attempt status: `FAILED_INPUT_QC`; it failed before successful full-FOV preprocessing completion. The retained manifest has no stack MAT/prepared/QC artifacts.
- Root cause: Q001 `q001_validate_group_geometry` used MATLAB-incompatible temporary-result chained indexing (`double(...)(:)`) at line 99. The later Python CUDA-extension warning came from the failure trap and is not causal.
- Hotfix: mirror the verified baseline geometry-comparison structure: assign DICOM values to local variables before indexing; keep exact Rows/Columns checks and same-shape tolerance checks for ImagePositionPatient, ImageOrientationPatient, PixelSpacing, and SliceThickness. No scientific/preprocessing/QC or baseline behavior changed.
- Regression: synthetic one-group/ten-DICOM MATLAB runtime smoke executes the validator successfully and separately proves a PixelSpacing mismatch raises `Q001:Geometry`.
- Rerun state: `CODE_READY_REAL_INPUT_RERUN_PENDING`; next formal root must be `/home/universe/SVR/data/Code_10w_v1/Q001_full_fov_inputs_v2`. No reconstruction was run.
- Local hotfix implementation commit: `49f176c` (`Fix Q001 MATLAB geometry validator syntax`). PUSH PERFORMED: NO.

## Q001_CONTROLLED_RECONSTRUCTION_V1 — implementation and server migration preparation

- Date: 2026-09-29
- Ready input verification: `/home/universe/SVR/data/Code_10w_v1/Q001_full_fov_inputs_v2/q001_input_manifest.json` is `READY_FOR_Q001_RECON_IMPLEMENTATION`, with strict QC passed for SAX 14/140/[288,256], 2CH 15/150/[288,256], and 4CH 12/120/[256,288]. Source git: `8297845a562d70bc33a135dbd76e7bf0daa3e2ce`.
- Implementation: Q001A trains only full-FOV prepared observations; Q001B provides full-FOV observations only to one HB1 stack-registration pass then optimizes only original cropped observations. Both load only B6 `resolved_config`; B6 trained reconstruction state warm-start is rejected.
- Migration target: `/data/dengyz/dataset/Code_10w_v1/Q001_full_fov_inputs_v2`, copied by checksum rsync and verified there against all manifest artifact SHA256 values before either server route can run.
- Local MATLAB full-FOV original-MultiMap mapping was attempted without reconstruction. The matcher reached `begin build dict` on SAX but produced no mapping output in this environment; partial mapping root is preserved and no native-reference bundle or formal reconstruction result is claimed.
- Next phase: `Q001_SERVER_INPUT_MIGRATION_PENDING`. FORMAL Q001A/Q001B: NOT RUN. D3/Q002: NOT STARTED. PUSH PERFORMED: NO.

## Q001_CONTROLLED_RECONSTRUCTION_V1 — pre-server evaluation contracts

- Date: 2026-09-29
- Full-FOV native reference: `COMPLETE_LOCAL_VERIFIED` at `/home/universe/SVR/data/Code_10w_v1/Q001_full_fov_native_reference_v2`, schema `q001_full_fov_native_reference/v1`, with exact MP-PCA(full-FOV MIND_mag_reg) provenance. Map/source-MAT SHA256: SAX `f4b5b9d8765a106a8ca7c12fe67e6ecf8f84e1d1c03385a9d58a55728b2f84a7` / `eaa5f825e0d1cd2fbdde093de29e6b05d3e479f47780146ca55473c898121b82`; 2CH `be888e965794412b1f0be648bc67a21edde9f70a37018da93c5e448b6343ce5e` / `18d5a62d75808c92e6704ec60a4f062c8bece6ec0a423c9842d52a658b5ce8ec`; 4CH `546500dc5aab6236918d26b912a790cdf5d29fa43929fbd0f52bd9121597d72a` / `436f49d8263faf7b35b2c9583f912e2832bbff28b40b4f1fd36ce5b309ebebcf`.
- Contracts: added read-only full-FOV reference verification, empty-destination-only reference migration/verification commands, true-pooled Q001 map/signal metrics plus retained legacy D2-style macro diagnostics, checkpoint/export provenance checks, exact Q001A ROIs, strict Q001B/B6 common support, and shared-HB1 initialization identity-order regression coverage.
- Diagnostic record supplied by the completed local run: MATLAB R2025b, exit 0, default matcher parallel behavior, 186990 dictionary entries, SAX group 0 shape `[288,256]`, matcher internal wall `73.8955 s`, shell wall `92 s`, output exists, with no captured crash/OOM evidence. Pool1 was not needed. No matcher was rerun for this record.
- Status: `CODE_READY_SERVER_MIGRATION_PENDING`. FORMAL Q001A: NOT RUN. FORMAL Q001B: NOT RUN. SERVER MIGRATION: NOT RUN. D3/Q002: NOT STARTED. PUSH PERFORMED: NO.

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

## D2_BASELINE_SIGNAL_K8_V1 — implementation update

- Date: 2026-09-28
- Status: CODE_READY_SERVER_RUN_PENDING.
- Why historical exports are insufficient: their PSF export was disabled, so they are K=1 and do not match the reconstruction training physics (`training.psf_samples=8`).
- Method: load only final `trad_B6/model.pt` and `frozen_mlp_B6/model.pt`, restore their stored configuration/training space, and perform K=8 seeded native weighted-signal inference without an optimizer or reconstruction step.
- Code/config: `quality_experiments/D2_k8_signal_domain/` and `quality_experiments/server_commands/run_D2_CYJ.sh`; baseline trainer/exporter and all historical artifacts remain unchanged.
- Metrics: method-specific global/per-stack/per-weight/per-stack-weight signal metrics, strict Bloch/FrozenMLP paired support, and decoder prediction disagreement. Signal units are `original_input_intensity`.
- Local validation: synthetic K=8 export, metric-schema, strict-support, artifact-schema, overwrite-protection tests plus CPU CLI/import validation in `knesvr_torch`.
- Server command/output: run `server_commands/run_D2_CYJ.sh` on `cr_dreme`; output root is `/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/quality_experiments_v1/CYJ/D2_k8_signal_domain_baseline_v1/`.
- Formal results: FORMAL K=8 RESULT NOT RUN. No server-only CYJ checkpoints or prepared artifacts were loaded locally.
- Local implementation commit: `79ff7b6` (`Add D2 K8 signal-domain evaluation`). PUSH PERFORMED: NO.

## D2_BASELINE_SIGNAL_K8_V1 — pre-server paired-support hotfix

- Date: 2026-09-28
- Status: CODE_READY_SERVER_RUN_PENDING; no formal CYJ D2 execution occurred before or during this hotfix.
- Root cause: runner figure generation referenced undefined `paired_weight`; only paired stack-weight/global aggregations had been built.
- Fix: strict paired support now has explicit per-stack, per-weight, per-stack-weight, and global outputs. Primary figures and `RESULT_SUMMARY.md` use strict paired rows; method-specific metrics remain preserved as clearly labeled secondary diagnostics.
- Regression validation: synthetic three-stack/ten-weight downstream integration test writes paired metrics and all figure paths, covering the former runtime failure without `/data` or a formal checkpoint.
- Formal results: FORMAL K=8 RESULT NOT RUN. D3 remains PLANNED. Next action is D2 server run and review, then a signal-domain fidelity decision before any later conditional evaluation.
- Local hotfix implementation commit: `c9ece24` (`Fix D2 paired-weight evaluation path`). PUSH PERFORMED: NO.

## D2_BASELINE_SIGNAL_K8_V1 — externally reviewed formal server result

- Date: 2026-09-29
- Status: SERVER_RUN_COMPLETED_EXTERNALLY_REVIEWED. Source: user-provided server result; the server artifact contents were not re-verified locally.
- FrozenMLP strict-paired global: NRMSE `0.059260425456891364`, NCC `0.9156052955158083`, RMSE_signal `24.05715773093939`.
- Bloch-vs-FrozenMLP predicted-signal disagreement: NRMSE `0.014340639073069109`, NCC `0.9946402104890828`, RMSE_signal `4.702345780725444`.
- D3 remains PLANNED and was not started. The next experimental route remains Q001 full-FOV input generation and subsequent signal-domain review.

## G0 — externally reviewed server diagnostics

- Date: 2026-09-29
- Corrected G0 v2 pose audit: server run completed and externally reviewed. G0 v1 correspondence remains valid; v2 corrected pose values are interpretable and show no catastrophic rigid divergence. Source: externally reviewed, not locally re-verified artifacts.
- G0 pose-error correlation: formal diagnostic completed and reviewed from user-provided server results. Conclusion: pose drift is a local/secondary contributor, not a consistent cross-stack dominant explanation of central-map structural mismatch.

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
