# Code_10w implementation status

Last updated: 2026-09-14

## Current boundary

Trad v1 is a frozen runnable baseline: project audit, preprocessing,
bridge/timing, dataset geometry, quantitative INR, HHZ direct signal model,
rigid/PSF, staged objective/training, checkpointing, physical-RAS inference,
export, and CPU tiny smoke are implemented. MLP v1 offline training and online
integration are runnable under the same INR/rigid/PSF/objective contract; its
current tiny MLP checkpoint is functional-fixture only, not scientifically
validated reconstruction.

## Audit record

- HHZ authority: `/home/universe/SVR/multimap_postprogramming/MultiMapCode_hhz`.
  Its `PreData.m` confirms NumImg=10, FA=[45,45,45], TI=[50,150], and
  T2prep=[35,45,55].
- DYZ legacy reference:
  `/home/universe/SVR/multimap_postprogramming/MultiMapCode/else/MultiMapCode`.
  Its `PreData.m` uses TI=[10,100]; it was used only for the available
  MP-PCA helper and the known-difference record.
- NeSVoR source: official `master` commit
  `2e96a91bdd30174210caea911e03a2778c65adbe`.
- mDM source: official `master` commit
  `a24ab1008d48e92ddf6f2bb8131f9a58dda23e95`.

## Phase 1 deliverables

- [x] Independent `trad/` and `mlp/` skeletons with Phase-2–9 placeholder
  directories and independent `third_party` trees.
- [x] NeSVoR pure-Python source, LICENSE, README, and provenance in both
  methods; only the three allowed mDM MLP reference files in `mlp`.
- [x] No vendored `.git` or generated Python cache remains.
- [x] HHZ MATLAB source copied unmodified to each `hhz_original/` directory.
- [x] `preprocessing_v1` provides path validation, explicit options, optional
  MP-PCA, optional figures, and an explicit smoke-slice limit.
- [x] `Mag_crop = MIND_mag_reg`; saved MAT output includes Mag, Mag_crop,
  Acq_time, TR, VPS, FA, TI, T2_prep, Info, per-slice HB1 geometry metadata,
  source filenames, crop indices, and MP-PCA metadata.
- [x] Protocol config explicitly fixes HHZ TI=[50,150] and records the DYZ
  TI=[10,100] difference.

## Phase 1 validation

- PASS — `find trad mlp -type d -name .git -print`: no output.
- PASS — `conda run -n knesvr_torch python -m compileall trad mlp`.
- PASS — MATLAB syntax/entry check for `preprocess_stack_v1` (arity 3).
- Not run by Phase-1 scope: DICOM/MIND/MP-PCA data smoke and all later-stage
  training/reconstruction tests.

## Phase 1 review correction / Phase 2 deliverables

- [x] Corrected the Phase-1 MP-PCA order from incorrect `MP-PCA → MIND` to
  intended `MIND → MP-PCA`; final `Mag_crop` is now either
  `MP-PCA(MIND_mag_reg)` or `MIND_mag_reg` when disabled.
- [x] Kept the existing actual v1 MP-PCA setting `center_data=true`; all
  MP-PCA options are explicit, and the contradictory copied comment is fixed.
- [x] Preprocessing validates constant TR/VPS and each native group’s 10
  DICOM geometries before processing, then records SOP/Series UIDs and
  zero-based crop offsets for cross-machine bridge use.
- [x] Module 02 uses one explicit h5py MATLAB-v7.3 loader, restores MATLAB
  column-major dimension order, validates `[Nx,Ny,10,Nslice]`/`[10,Nslice]`,
  resolves UID→current DICOM paths, computes HHZ timing, and saves NPZ/JSON.
- [x] Module 03 retains NeSVoR PointDataset-style flattened `xyz`/`v` batches
  while enforcing one `group_idx`/future rigid pose per complete 10-weight
  native slice. MIND weights share HB1 cropped DICOM-LPS geometry.

## Phase 2 validation

- PASS — static suite: `test_protocol`, `test_timing`, `test_dataset_grouping`,
  `test_mat_v73_loader`, `test_crop_geometry` (5/5 under `knesvr_torch`).
- PASS — real tiny smoke after failure recovery: `CYJ`, `multimap_2ch_901`,
  one group, `enable_mppca=true`; 10 weights/group, image `[10,145,129]`,
  timing `[1,9]`, cropped-HB1 geometry `[1,4,4]`, `center_data=true`.
- PASS — MLP receives byte-identical common preprocessing/Module 02/Module 03
  code and imports it successfully; no duplicated real smoke was run.

## Next-stage entry

## Phase 2 review correction / Phase 3 Trad deliverables

- [x] Changed Module 03 `xyz` from mislabeled DICOM-LPS world points to
  centered local `[column,row,slice]` mm. `group_resolution_xyz_mm` is
  `[col_spacing,row_spacing,slice_thickness]`; Module 02 retains
  DICOM-LPS affine provenance.
- [x] Added explicit LPS→RAS conversion and constructed each initial vendored
  NeSVoR `RigidTransform` from the cropped HB1 affine. All ten weights share
  the same native-slice group pose; no identity-pose substitution is used.
- [x] Added multi-stack `QuantPointDataset([paths])`, global group reindexing,
  preserved source stack IDs, RAS-transformed bounding boxes, and strict
  IOP norm/dot-product validation without orthogonalization.
- [x] Implemented Trad Module 04 only from vendored NeSVoR HashGrid,
  `HashEmbedder`, `build_encoding`, `build_network`,
  `compute_resolution_nlevel`, and `RigidTransform`: shared latent → bounded
  T1/T2/B1/A fields.
- [x] Implemented Trad Module 05A as a direct tensorized PyTorch equivalent
  of HHZ `sim_T1T2_10HB_bssfp.m` (no EPG), including the 16-readout centre
  window and Mz carry-over. Its protocol rejects DYZ TI=[10,100] and accepts
  only v1 HHZ TI=[50,150].
- [x] Implemented Trad Module 06 one-axis-angle-pose-per-group, local
  anisotropic Gaussian PSF (`resolution2sigma(..., isotropic=False)`), then
  rigid RAS transform. The forward evaluates/simulates every PSF sample,
  selects weight, applies A, and averages only predicted signal. It exposes
  `deformable=false` and has no deformation parameters.
- [x] Synchronized common corrected Module 02/03 geometry plus Module 04/06
  code and docs to `mlp`; only an import and byte-identity check was run.
  Trad-only Module 05A was not copied to mlp.
- [x] Added root `.gitignore` rules and removed generated `__pycache__` and
  pytest cache directories under `trad`/`mlp` after validation.

## Phase 3 validation

- PASS — `conda run -n knesvr_torch pytest -q` on
  `test_dataset_grouping`, `test_local_rigid_geometry`,
  `test_multistack_dataset`, `test_quantitative_inr`, `test_signal_decoder`,
  `test_rigid_group_pose`, and `test_psf_forward`: 8 passed.
- PASS — deterministic small forward benchmark: batch 1024, PSF K=4;
  output, INR gradients, and rigid gradients finite; axis-angle tensor `[1,6]`
  exists; `deformable=false`.
- PASS — MATLAB was available. One deterministic 20-case raw-HHZ parity run
  against `sim_T1T2_10HB_bssfp.m`: maximum absolute error
  `1.2351231148954867e-15`.
- PASS — MLP common-module import succeeded and four synchronized source files
  were byte-identical to Trad. No MLP full suite and no preprocessing/MIND run
  were repeated.

## Next-stage entry

## Phase 3 review correction / Phase 4 Trad runnable-chain deliverables

- [x] Removed the forced `HashEmbedder` concrete-type gate. CPU explicitly
  follows NeSVoR `USE_TORCH=True`; CUDA retains vendored NeSVoR/tinycudann
  backend selection through `build_encoding`/`build_network`.
- [x] Restored NeSVoR `PointDataset` physical-RAS bounding-box margin of
  `2 * max(group_resolution_xyz_mm)` on both sides.
- [x] Propagated validated Phase-1 DICOM TR/VPS into bridge NPZ data and
  `QuantPointDataset` group metadata. One online Trad protocol now requires
  equal TR (tolerance) and exactly equal VPS across all groups/stacks.
- [x] Added a non-mutating NeSVoR-style `training_space.py`: center physical
  RAS bbox, `spatial_scaling=30`, scale local geometry, compose `-center`
  before group poses, and provide tested inverse pose/query conversion.
- [x] Added `axisangle_init` to `GroupRigidPSF` and NeSVoR `trans_loss`
  semantics relative to the DICOM-derived initial group poses.
- [x] Added Trad Module 07 Stage A/B training (AdamW, encoding/network
  groups, scheduler, 10-weight MSE, static stack weights, optional normalized
  T1/T2/B1 gradient regularization), Module 08 physical-RAS NIfTI/pose export,
  and Module 09 artifact QC.
- [x] Added explicit smoke/server configs, preparation/training/smoke scripts,
  and a Chinese Trad data-flow README. No training CLI parameter silently
  falls back to phase-3 smoke defaults.
- [x] Synced only common bridge/dataset/INR/rigid/training-space infrastructure
  to MLP; no Trad decoder, MLP decoder, MLP training, or MLP full suite ran.

## Phase 4 validation

- PASS — unified CPU pytest suite (11 tests): dataset/multi-stack margin,
  training-space round-trip, INR, signal, rigid/PSF, dataset-derived protocol,
  checkpoint metadata/load, and end-to-end tiny train/export/QC.
- Initial smoke entry failed before training because executing
  `scripts/run_training.py` made `scripts/` the Python import root. Fixed it
  to `python -m scripts.run_training`, removed only that failed smoke's exact
  `/tmp/multimap_phase4_trad_smoke` directory, then reran once.
- PASS — real smoke: existing Phase-2 `CYJ_20260819_163756 / 2ch`, one group,
  all 10 weights, lightweight bridge only (no MIND/MP-PCA), Stage A=10 and
  Stage B=4, PSF K=2. QC reports all four fields finite, 14 log rows,
  rigid tensor present, and `deformable=false`.
- PASS — MLP common infrastructure import and byte-identity check. No MLP
  real smoke or full test was repeated.

## Phase 4 review correction validation

- [x] Quantitative regularization evaluates the fields at detached current
  group-pose physical-RAS coordinates rather than at scaled local pixels; it
  gives finite nonzero field gradients but no rigid gradient.
- [x] The server example and Trad README require SAX, 2CH, and 4CH prepared
  NPZ inputs and pass all of them explicitly.
- PASS — `test_quantitative_regularization_coordinates` plus
  `test_end_to_end_tiny`: 3 tests.  No MATLAB parity or MIND was repeated.

## Phase 5 MLP offline deliverables

- [x] MLP has synchronized copies of the fixed v1 protocol, data bridge,
  dataset/geometry, Quantitative INR, rigid/PSF, and the HHZ-compatible Trad
  signal teacher.  The teacher is not EPG.
- [x] `build_timing_pool.py` enforces full ten-weight groups, unique timing
  vectors, equal TR within tolerance, exact VPS, and provenance.  Timing pool
  TR/VPS are validated before synthetic training.
- [x] One-time HDF5 `train.h5`/`valid.h5`/`test.h5` generation samples the
  stated continuous T1/T2/B1 ranges, uses normalized 10-heartbeat HHZ teacher
  targets, and splits by timing ID (not sample).
- [x] The offline model input is 12D
  `[T1/1000,T2/1000,B1,timing2..10/1000]`; its output is a normalized 10D
  fingerprint.  It is exactly the mDM-style 3×(Linear 200, BN, LeakyReLU)
  network followed by Linear 10.
- [x] `FrozenMLPSignalDecoder` freezes parameters and keeps BatchNorm eval
  under outer `train()` while preserving T1/T2/B1 input gradients.
- [x] Best checkpoints retain architecture, normalizations, complete fixed
  HHZ protocol, validated TR/VPS, timing provenance, parameter ranges, and
  seed.  Metrics include overall/per-weight RMSE, MAE, max error, and
  T1/T2/B1 derivative error/cosine summaries.

## Phase 5 validation

- PASS — MLP offline pytest:
  `test_mlp_model_shape`, `test_mlp_timing_pool`, `test_mlp_dataset_split`,
  `test_mlp_teacher_fidelity`, `test_frozen_mlp_gradient`, and
  `test_frozen_mlp_batchnorm_eval`: 7 tests.
- PASS — one CPU tiny run at `/tmp/multimap_phase5_mlp_smoke`: 1024 train,
  256 validation, 256 test, 2 epochs; finite checkpoint, history, and test
  metrics written.  The underlying prepared case has only one timing vector,
  so the run explicitly uses a labelled functional three-vector fixture.  It
  validates code flow only and does not claim cross-timing generalization.
- Not implemented by design: online decoder insertion, MLP reconstruction,
  end-to-end reconstruction training, or an MLP benchmark.

## Phase 5 review correction / Phase 6 deliverables and validation

- [x] Offline HDF5 uses the single CPU-RAM TensorDataset path, non-singleton
  BatchNorm batches and batchwise-device fidelity. Teacher device is explicit.
- [x] Timing pools, metadata and checkpoints retain/check functional-fixture
  provenance, SHA256, timing ranges and complete protocol; mismatch/out-of-
  range timing errors are strict (except 1e-4 ms float32 storage precision).
- [x] MLP online reuses frozen Trad INR, group rigid pose, PSF, loss,
  regularization, AdamW stages, sampler and export. Only decoder differs;
  decoder parameters stay frozen/out of optimizer and BN remains eval.
- [x] Separate recon configs and explicit three-stack server launcher exist;
  formal server config rejects functional-fixture checkpoints.
- PASS — targeted offline regression (RAM HDF5, SHA mismatch, fixture guard),
  online suite (5 tests), one final CPU smoke after one float32-boundary repair,
  and one CPU decoder benchmark at 1,000/10,000 with 3-rep medians.
- Smoke: CYJ/2ch, one group/all ten weights, Stage A=10/B=4, PSF K=2; four
  finite NIfTI fields, 14 finite logs, rigid/no-deform and frozen BN. The
  exact failed temporary smoke root was removed before the single re-run.
- Not run: GPU training/benchmark, formal multi-subject timing-pool training,
  formal 3-stack reconstruction, Phase-6 MATLAB parity or image comparison.

## Next-stage entry

Implementation v1 is complete. Do not begin formal server training or alter
the fixed HHZ scientific contract without a new command. See
`IMPLEMENTATION_REPORT_v1.md`.

## Phase 7 formal MLP validation readiness

- [x] Formal timing pool supports only explicit manifest subject/stack
  provenance and retains all group rows; formal synthetic split is subject-
  disjoint and records train/pool timing domains separately.
- [x] Formal candidate checkpoints start `validation_status=unvalidated` and
  remain rejected by online reconstruction until a separate report is reviewed.
  `validate_formal_mlp.py` writes signal/gradient metrics and
  `awaiting_manual_review`, never an automatic approval.
- [x] Formal timing audit CSV/summary and separate Trad/MLP GPU peak-memory
  benchmark fields are implemented. Targeted Phase-7 pytest PASS (4/4).
- Not run / blocked on this host: formal manifest audit, CUDA teacher
  generation, one formal candidate training, held-out validation and GPU
  benchmark. Evidence: `nvidia-smi` is unavailable and no explicit
  multi-subject prepared-observation manifest was supplied. No full 3-stack
  reconstruction was started.

## Next-stage entry

Provide a formal source manifest and execute the documented CUDA sequence on
the server. Stop after `formal_timing_audit.csv`,
`formal_validation_report.json` and `decoder_gpu_benchmark.json` for manual
review; do not start full reconstruction yet.

## Deployment layer — dual environment (2026-09-14)

- [x] Added one explicit `deployment-v1` subject/stack manifest schema plus
  path examples. All local/server batch tools consume this contract and do not
  discover directories, subjects, stacks, missing parameters, or defaults.
- [x] Local MATLAB has dedicated full-stack preprocessing wrappers that call
  frozen Module 01 with `preprocess_options_v1([], false, true)` and emit
  per-stack `preprocess_qc.json`; server Python has no MATLAB invocation.
- [x] Server tooling validates transferred v7.3 MAT fields, prepares only
  exact manifest entries, produces required bridge artifacts and
  `PREPARED_BATCH_QC.json`, and provides strict formal-MLP/reconstruction
  readiness checks without installing packages or backend fallback.
- [x] Formal MLP manifest generation uses exact prepared paths. A separate
  manual approval transition verifies checkpoint/report SHA256 and preserves
  learned state. Online reconstruction still rejects unapproved candidates;
  benchmark-only loading permits only the specified non-functional candidate
  statuses.
- [x] Added fixed per-subject SAX/2CH/4CH Trad/MLP wrappers and updated server
  launchers to derive their root and use `cr_dreme`. No formal preprocessing,
  training, benchmark, or reconstruction was run in this deployment phase.
- PASS — deployment pytest suite under `knesvr_torch`: 6 tests across the five
  requested test files. Scientific algorithms are unchanged.

## Next-stage entry

Copy a reviewed deployment manifest and local preprocessed outputs to the
server, run the documented transfer/prepare/formal-validation sequence, then
stop for manual checkpoint review before any formal MLP reconstruction.
