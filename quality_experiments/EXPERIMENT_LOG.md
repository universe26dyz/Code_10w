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

## Q001_CONTROLLED_RECONSTRUCTION_V1 — pre-server correctness/provenance hotfix

- Date: 2026-09-29
- Fixed Q001A fixed-baseline endpoint: it now applies the exact full-FOV ROI only to the full-FOV prediction and compares it with the entire already-cropped verified reference; no resize/interpolation is permitted.
- Fixed provenance: route manifests now record the approved signal simulator SHA separately from the B6 model SHA; evaluation manifests use the distinct values and record verified cropped-reference plus B6 central/K8 artifact hashes.
- Route dependency: Q001A verifies both full-FOV and cropped references; Q001B verifies the cropped reference against its cropped preprocessed MAT provenance and does not require a full-FOV reference.
- Status unchanged: FORMAL Q001A/Q001B and SERVER MIGRATION remain NOT RUN; D3/Q002 remain NOT STARTED; PUSH PERFORMED: NO.

## Q001B — post-hoc map-domain PSF correction

- Formal Q001B reconstruction is COMPLETE. Its central/no-map-PSF metrics and K=8 signal evaluation are VALID.
- `mapping_map_psf_K32` parameter archives are INVALID_AS_MAP_PSF: native-plane parameter fields were point samples, so K=32 affected only signal generation. They remain historical provenance artifacts and are not a map-domain endpoint.
- True map-domain K=32 is now a post-hoc, no-reconstruction wrapper over final T1/T2 volumes, final rigid poses, route-matched prepared geometry, and the verified route reference. Status: `PENDING_POSTHOC`.
- Q001A remains NOT RUN; D3/Q002 remain NOT STARTED; PUSH PERFORMED: NO.

## Q001A — formal PSF128 export preparation

- Added an immutable `formal_export_psf128/` quantitative-volume export. Raw root `T1_3D.nii.gz`, `T2_3D.nii.gz`, `B1_3D.nii.gz`, and `amplitude_3D.nii.gz` retain point-sampled historical semantics.
- Frozen future convention: 1.0 mm resolution, output PSF factor 1.0, isotropic vendored NeSVoR `resolution2sigma`, K=128, seed 20260911; matched in convention to immutable `nesvor_v5_128_w1` only.
- Q001A code is READY_FOR_FORMAL_SERVER_RUN; its formal result remains NOT RUN. Q001B remains immutable. Q002/Q003 remain NOT STARTED. PUSH PERFORMED: NO.

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

## Q001A_FORMAL_PSF128_EXPORT — physical-coordinate and provenance hotfix

- Date: 2026-10-08
- Status: CODE_READY_LOCAL_TESTED_PENDING_COMMIT; formal Q001A reconstruction/export remains NOT RUN.
- Correction: K=128 Gaussian offsets are now sampled in physical RAS-mm using `resolution2sigma(1.0, isotropic=True)`, then transformed once through `physical_ras_to_train` before INR evaluation. The prior implementation added physical-mm sigma after training-space scaling and was invalid at `spatial_scaling=30`.
- Export safety/provenance: K=128 uses a bounded point budget with deterministic chunk-invariant sampling; raw and PSF128 share the same bbox resolver. The formal manifest records code/NeSVoR provenance, dirty state, checkpoint/pose hashes, physical sigma, and requested/effective chunk sizes.
- Historical Q001B results were not changed: reconstruction COMPLETE; central and K8 VALID; old parameter K32 INVALID_AS_MAP_PSF; true map-domain K32 COMPLETE.
- Q001A/Q002/Q003 formal runs: NOT RUN. PUSH PERFORMED: NO.

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

## Q001A — failed pre-training full-FOV intensity-normalization attempt

- Date: 2026-10-08
- Status: `FAILED_PRETRAIN_INTENSITY_QUANTILE`; code commit: `609bfaecd54b176ee3bd513d6900d8ebdfcd32d9`.
- Failure occurred before stack initialization, model construction, Stage A/B, checkpoint, or metrics: CUDA `torch.quantile` rejected the full-FOV masked-intensity tensor as too large.
- Formal metrics generated: false. Checkpoint generated: false. This is not a Q001A formal result and does not modify immutable Q001B results.
- Failed log is protected from overwrite by the server runner. Before rerun it must be retained under `Q001A_full_fov_full_recon_v1.FAILED_PRETRAIN_QUANTILE_609bfae.log` (or an equally explicit preserved name).
- Hotfix: exact CPU `torch.quantile` thresholds only when the explicit input-too-large runtime error occurs; strict `>`/`<` trimming and mean remain on the original tensor. Provenance records execution path and input count.
- Confirmed server reference root: `/data/dengyz/dataset/Code_10w_v1/reference_data/CYJ/Q001_full_fov_native_reference_v2`.
- Current Q001A code status: `READY_FOR_FORMAL_SERVER_RERUN` after local tests. Q002/Q003: NOT STARTED. PUSH PERFORMED: NO.

## Q001A — second failed pre-training quantile fallback attempt

- Date: 2026-10-08
- Status: `FAILED_PRETRAIN_INTENSITY_QUANTILE_CPU_FALLBACK`; code commit: `5fef4b19035f056a0679b7f1eaca643ff70e197f`.
- CPU `torch.quantile` raised the same input-too-large error, confirming an element-count limit in the formal-server PyTorch implementation rather than a CUDA-only limit. This occurred before stack initialization, model construction, Stage A/B, checkpoint, export, or metrics.
- Formal metrics generated: false. Checkpoint generated: false. The immutable Q001B result is unchanged.
- The next fallback uses exact CPU `torch.kthvalue` order statistics with `rank=q*(N-1)`, floor/ceil ranks, and linear interpolation. It uses every element and performs no approximation or subsampling; strict `>`/`<` trimming remains unchanged.
- The second failed log remains user-preserved before rerun as `Q001A_full_fov_full_recon_v1.FAILED_PRETRAIN_QUANTILE_CPU_5fef4b1.log`; the runner continues to refuse log overwrite.
- Current Q001A code status: `READY_FOR_FORMAL_SERVER_RERUN` after an actual `2**24 + 1` float32 CPU preflight. Q002/Q003: NOT STARTED. PUSH PERFORMED: NO.

## Q002_B6_ROI_SAME_ANCHOR_MSE

- Date: 2026-10-08
- Status: `CODE_READY_LOCAL_TESTED`; formal Q002 reconstruction/evaluation: `NOT_RUN`; PUSH PERFORMED: NO.
- Historical bookkeeping correction: user reported Q001A formal Stage A/B and evaluation completion at `e0e35158397a07556039fecedf1d98921d78ab2e`; Q001B remains completed. No Q001A manifest SHA or unprovided metric provenance is invented here.
- Control: B6 FrozenMLP baseline; only original cropped `CYJ/{sax,2ch,4ch}/observations.npz` are accepted for training and stack initialization. Q001 data/poses and all warm starts are rejected.
- Sampling: 64 integer native anchors, each holding ordered weights 0..9, for effective scalar budget 640. Training uses one shared K=8 PSF neighbourhood and vector MSE; cosine is evaluator-only.
- Support/QC: ten-way foreground AND plus finite observations; per-group intersection/union, mask equality, discarded scalar support, scalar per-weight counts, source SHA256, scalar-derived B6 stack weights, and fixed monitor identity hash are written to the Q002 route manifest.
- Evaluation: central/no-map-PSF T1/T2 and K8 signal compare Q002 and B6 only on strict finite common support. True-pooled outputs include per-group/per-stack/global metrics; fingerprint cosine is a read-only diagnostic. K128 reuse is 1 mm/factor 1/K128/seed 20260911.
- Server commands: `quality_experiments/server_commands/run_Q002_CYJ.sh` then `quality_experiments/server_commands/evaluate_Q002_CYJ.sh`. They require CUDA, approved-checkpoint checksum visibility, a clean Git worktree, and empty output targets; they do not delete paths.
- Q003: `NOT_STARTED`; no cosine training objective or other Q003 scientific change was implemented.

## Q002_B6_ROI_SAME_ANCHOR_MSE — pre-formal correctness hotfix

- Date: 2026-10-08
- Status: `CODE_READY_LOCAL_TESTED`; formal Q002 reconstruction/evaluation remains `NOT_RUN`.
- Corrected export order: raw quantitative export now writes `final_rigid_poses.json` before PSF128 export verifies its SHA; K128 remains 1 mm/factor 1/seed 20260911.
- Corrected evaluation: verified reference manifest/MAT provenance is loaded through `load_verified_reference`; K8 B6 data are aligned by `(group_idx, weight_idx)` and signal metrics use signal-domain units/true pooled summaries.
- Corrected diagnostics: observed-fingerprint cosine is reported separately for B6/Q002 with epsilon protection and near-zero norm counts; it is not a training objective. Inter-method prediction cosine remains separately named.
- Corrected training records: resolved config, training/monitor CSVs, fixed monitor identity, poses, timing record, route/experiment manifests, and result summary are generated by the Q002 path.
- Local verification: 73 targeted/regression tests passed; no formal GPU run and no historical B6/Q001 artifact was modified. Q003 remains NOT_STARTED. PUSH PERFORMED: NO.

## Q002_B6_ROI_SAME_ANCHOR_MSE — spatial-regularization control hotfix

- Date: 2026-10-08
- Status: `CODE_READY_LOCAL_TESTED`; formal Q002 remains `NOT_RUN`.
- Q002 data sampling is unchanged at 64 joint anchors × 10 weights/K8. Spatial regularization now independently reuses `CachedBalancedSampler` on original B6 scalar cropped support, yielding 640 scalar candidates and the B6-configured 256 effective regularization points per iteration.
- Training logs record candidate/effective counts and source. Evaluation validates full unique group×weight identity sets, records separate reconstruction/evaluation Git provenance, and writes observed-fingerprint cosine per group/stack/global with per-norm epsilon semantics.
- Phase-state bookkeeping now consistently records Q001A as externally reported complete, Q001B complete, Q002 ready but formal NOT_RUN, and Q003 NOT_STARTED.

## Q002_B6_ROI_SAME_ANCHOR_MSE — v1 forward-shape failure and immutable v2 retry

- v1 formal attempt status: `FAILED_PRETRAIN_FORWARD_FINGERPRINT_SHAPE_CONTRACT` at commit `5158f7b28362f6c9edd802a5fe08a1bbe4a63859`.
- Failure phase: Stage A first forward, before backward/optimizer; optimizer steps 0; checkpoint/export/metrics not generated. Existing v1 output root and `.log` are immutable and must not be overwritten.
- Root cause: Q002 incorrectly required fingerprint `[B*K,1,10]`; approved FrozenMLP contract is `[B*K,10]`, matching scalar forward gather semantics.
- v2 retry root: `Q002_B6_roi_same_anchor_mse_v2` with matching v2 log. It remains `READY_FOR_FORMAL_SERVER_RERUN`; formal successful result `NOT_RUN`.

## Q002S640_B6_ROI_SAME_ANCHOR_MSE

- Date: 2026-10-09
- Status: `CODE_READY_FORMAL_SERVER_RUN_PENDING`; formal reconstruction/evaluation: `NOT_RUN`; PUSH PERFORMED: NO.
- Single scientific variable: Q002-S640 uses 640 same-anchor samples per iteration, while immutable Q002-64 remains 64. Both use the approved FrozenMLP, B6 cropped SAX/2CH/4CH observations and initialization, complete ten-weight AND support, sampling with replacement, K=8 shared-anchor PSF, vector MSE only, seed `20260911`, Stage A/B `2000/4000`, B6 normalization, stack weights, optimizer, scheduler, bbox, and no warm start.
- Per S640 optimizer iteration: 640 spatial anchors, 10 weights per anchor, 6,400 signal residuals, and 5,120 data PSF/INR spatial locations. This is not called an effective scalar budget. Spatial regularization remains independently sampled once from B6 scalar cropped support: 640 scalar candidates, B6-configured 256 effective regularization points.
- Immutable formal output/log: `/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/quality_experiments_v1/CYJ/Q002S640_B6_roi_same_anchor_mse_v1` and matching `.log`. Q002-64 v1/v2 roots are read-only and never reused as output.
- Evaluation: `evaluate_Q002S640_CYJ.sh` requires Q002-64 v2 plus Q002-S640 artifacts and evaluates B6/Q002-64/Q002-S640 only on `reference_valid AND B6_valid AND Q002_64_valid AND Q002_S640_valid AND finite`. Primary comparison is S640 vs 64; B6 is the secondary final-utility comparison. Central T1/T2 and K8 signal true-pooled metrics use this one support; observed-fingerprint cosine is read-only and explicitly marked as macro-over-groups when summarized.
- Commands prepared, not run: `quality_experiments/server_commands/run_Q002S640_CYJ.sh`, then `quality_experiments/server_commands/evaluate_Q002S640_CYJ.sh`. Q003-S640 remains NOT_STARTED.

## Q003S640_B6_ROI_SAME_ANCHOR_MSE_PLUS_COSINE

- Date: 2026-10-09
- Status: `READY_FOR_FORMAL_SERVER_RUN_PENDING_Q002S640_REVIEW`; Q003-S640 formal reconstruction/evaluation: `NOT_RUN`; PUSH PERFORMED: NO.
- Parent control: immutable `Q002S640_B6_roi_same_anchor_mse` code path. Q002-64 formal execution is user-reported complete and negative versus B6; its historical roots remain read-only. Local phase metadata retains Q002-S640 as code-ready because current server execution state is not observable from this checkout; no Q002-S640 result is fabricated here.
- Single scientific change: add anchor-level normalized ten-weight cosine loss to the existing joint vector MSE. Exact constants are `weight=1.0`, `epsilon=1e-8`; the same scalar-derived anchor stack weights apply to MSE and cosine. Q003 retains Q002-S640's 640 anchors, 6400 residuals, K8/5120 data locations, B6 cropped inputs/initialization/normalization/stack weights, Stage A/B 2000/4000, seed 20260911, amplitude semantics, and independent B6 scalar regularization 640 candidates -> 256 points once per optimizer step.
- Safety/provenance: immutable target is `Q003S640_B6_roi_same_anchor_mse_plus_cosine_v1` with matching log; dedicated run/evaluation scripts require clean Git and reject existing/non-empty paths. Reconstruction manifests record cosine settings, route parent, B6/decoder paths and hashes, controls, monitor identity, checkpoint, and PSF128 contract.
- Evaluation: B6/Q002-S640/Q003-S640 metrics use one strict finite native common support; primary ablation is Q003-S640 vs Q002-S640, while B6 is the final-utility comparator. Cosine is observed-fingerprint diagnostic only for evaluation and macro summaries are explicitly labeled.

## Q003S640_B6_ROI_SAME_ANCHOR_MSE_PLUS_COSINE_10K

- Date: 2026-10-09
- Status: `READY_FOR_FORMAL_SERVER_RUN`; formal result: `NOT_RUN`; PUSH PERFORMED: NO.
- Q002-64 is externally reported complete and negative versus B6; Q002-S640 is externally reported complete and positive versus Q002-64 and B6 on the controlled CYJ evaluation. Both are immutable/read-only comparators.
- The unrun original Q003-S640 6k route is superseded for execution by this immutable 10k route. Its first 6000 steps remain the primary cosine-ablation checkpoint: 2000 Stage-A + the approved 4000 Stage-B schedule. The secondary continuation runs a further 4000 Stage-B updates without resetting optimizer, scheduler, sampler, model, poses, normalization, or seed.
- `checkpoints/model_iter_6000.pt` is the primary checkpoint; `model.pt` is the 10000-step secondary extended-optimization checkpoint. The route preserves 640 anchors, 10 ordered weights, 6400 signal residuals, K8/5120 data locations, cosine weight 1.0/epsilon 1e-8, B6 scalar regularization 640 -> 256 once per step, and all B6/Q002-S640 controls.
- Post-training-only artifacts include closed-CSV convergence curves and summaries, checkpoint-specific native-plane exports, strict B6/Q002-S640/Q003-6000/Q003-10000 common-support evaluation, and deterministic SCMR-range visualizations. Commands are `run_Q003S640_10K_CYJ.sh` then `evaluate_Q003S640_10K_CYJ.sh`; neither has been run locally.

## Q004S640_K1_B6_ROI_SAME_ANCHOR_MSE

- Date: 2026-10-09
- Status: `CODE_READY_FORMAL_SERVER_RUN_PENDING`; formal result: `NOT_RUN`; PUSH PERFORMED: NO.
- Immutable context: Q002-64 is the completed negative result; Q002-S640 is the completed positive controlled-map result; Q003-S640-10k is externally reported complete with improved signal/fingerprint fidelity but no overall map-agreement improvement. These routes and their artifacts remain read-only.
- Single Q004 scientific change: training PSF K8 becomes central K1. Q004 retains Q002-S640's MSE-only joint-vector objective, 640 anchors, 10 ordered weights, 6400 residuals, B6 cropped inputs/initialization/normalization/stack weights, Stage A/B 2000/4000, seed 20260911, optimizer/scheduler, amplitude/B1/rigid model, and independent scalar regularization 640 -> 256 once per optimizer step.
- K1 means the exact local central coordinate with no Gaussian draw or neighborhood averaging: 640 data PSF/INR locations per optimizer step. Q004 exports both central K1 and posthoc K8 signal reprojections. The evaluator writes all B6/Q002/Q004 derived central-K1 comparator products only under the new Q004 evaluation root and uses strict three-way supports.
