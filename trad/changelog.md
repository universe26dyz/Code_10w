# Changelog

## 2026-09-10 — Full pipeline v1 scaffold / Phase 1

- Created the independent Phase-1 scaffold: `configs/`, `third_party/`, all
  Module 01–09 directories, `scripts/`, `tests/`, and `outputs/.gitkeep`.
- Vendored NeSVoR `master` commit `2e96a91bdd30174210caea911e03a2778c65adbe`
  as pure Python source plus its LICENSE/README; `third_party` contains no
  nested Git repository.
- Copied the unmodified HHZ MATLAB source subset into
  `modules/module_01_preprocess_matlab/hhz_original/`.
- Added `preprocessing_v1/preprocess_stack_v1.m` and explicit options. It
  preserves HHZ NumImg=10, FA=[45,45,45], TI=[50,150] ms, T2prep=[35,45,55]
  ms, reads TR/VPS from DICOM, saves bridge metadata, and assigns
  `Mag_crop = MIND_mag_reg`. It can explicitly limit smoke slices, suppress
  figures, and invoke the vendored MP-PCA routine.
- Recorded the authoritative HHZ TI=[50,150] vs DYZ legacy TI=[10,100]
  difference in the protocol and README; the DYZ value is not inherited.
- Tests: `find trad mlp -type d -name .git -print` PASS (no output);
  `conda run -n knesvr_torch python -m compileall trad mlp` PASS; MATLAB
  `preprocess_stack_v1` entry/arity check PASS.
- Not run: DICOM/MIND/MP-PCA data smoke, by Phase-1 test scope; no training,
  full-subject processing, or later Python pipeline was started.

## 2026-09-11 — Phase 1 review correction and Phase 2 bridge/dataset

- Corrected preprocessing order to `crop → MIND → optional MP-PCA → Mag_crop`.
  `Mag_crop_unregistered`, `MIND_mag_reg`, and final `Mag_crop` now have
  non-overlapping semantics; `Mag_crop` is always the later Python input.
- Kept `MPPCA_options.center_data=true` as the explicit v1 setting, passed all
  MP-PCA options explicitly, and corrected its contradictory copied comment.
- Added validation for constant DICOM TR/VPS, complete 10-weight native-slice
  geometry, and stable SOP/Series UID metadata. The bridge resolves UIDs only
  through explicit current `--dicom-dir`, never saved absolute paths.
- Added Module 02 v7.3 HDF5 loader, HHZ timing translation, DICOM-LPS crop
  geometry, manifest/NPZ preparation, and Module 03 NeSVoR-style
  `QuantPointDataset` with one future pose index per group.
- Tests: Phase-2 static suite PASS (5/5) using `knesvr_torch`; after the real
  MAT exposed single-group HDF5 shape handling, its focused loader regression
  test PASS. Installed pytest only because that required conda environment did
  not provide a pytest module.
- Tiny real smoke PASS: subject `CYJ`, stack `2ch`, one group, 10 weights,
  image shape `[10,145,129]`, timing `[1,9]`, geometry `[1,4,4]`,
  `center_data=true`; it ran MIND then MP-PCA exactly once successfully after
  a failed MP-PCA singleton-dimension attempt was repaired.

## 2026-09-11 — Phase 2 geometry correction and Phase 3 Trad core forward

- Corrected Module 03 coordinates to centered local `[column,row,slice]` mm,
  retained Module 02 cropped HB1 DICOM-LPS provenance, and explicitly converts
  at the Module 03/06 boundary to RAS-mm. Added strict IOP norm/orthogonality
  checks with no auto-orthogonalization.
- Each native group now has one initial NeSVoR `RigidTransform` derived from
  its cropped HB1 affine; all 10 weights share it. Added multi-stack input,
  global group indexing, preserved stack IDs, per-group `[col,row,thickness]`
  resolution, and RAS-transformed bounding boxes.
- Added Module 04 `QuantitativeINR`, using only vendored NeSVoR HashGrid
  primitives and a shared latent to bounded T1/T2/B1/A fields.
- Added Module 05A `TradSignalSimulator`, a direct tensorized HHZ
  `sim_T1T2_10HB_bssfp.m` recurrence. It has no EPG path and rejects any
  protocol other than v1 HHZ TI=[50,150].
- Added Module 06 rigid-per-group/local-anisotropic-PSF forward. It has
  `deformable=false`, no deformation parameters, and performs signal
  simulation before PSF averaging.
- Tests: the seven requested test files PASS (8 tests); batch=1024/K=4
  finite-gradient benchmark PASS; one MATLAB 20-case raw-signal parity PASS
  (max abs error `1.2351231148954867e-15`). No preprocessing/MIND rerun and no
  training were performed.

## 2026-09-11 — Phase 3 review correction and Phase 4 Trad runnable baseline

- Corrected shared bridge/dataset/INR/rigid infrastructure before training:
  removed forced HashEmbedder type checks; restored native NeSVoR CPU/tcnn
  backend behavior; restored physical RAS bbox `2*max_resolution` margin;
  wrote/validated DICOM TR/VPS; and added non-mutating NeSVoR center/scale=30
  training space with physical pose round-trip.
- `GroupRigidPSF` now preserves DICOM-derived `axisangle_init` and uses
  NeSVoR `trans_loss` relative to init. It remains rigid-only with no deform.
- Added Modules 07/08/09: Stage A/B AdamW training, exact 10-weight balanced
  MSE, static stack balancing, config-controlled normalized quantitative
  regularization, full checkpoint metadata, physical-RAS NIfTI export,
  physical final poses, and strict tiny-output QC.
- Added `smoke_cpu.yaml`, `server_train_example.yaml`, command scripts, and
  Chinese data-flow README. No formal server run was performed.
- Tests: unified requested suite PASS (11 tests). A first smoke process failed
  at Python module import before training; after changing the launcher to
  `python -m`, the only completed smoke PASS used existing Phase-2 CYJ/2ch,
  1 group/10 weights, 14 CPU iterations, PSF K=2. It produced finite four
  NIfTI fields, checkpoint, log, config, and physical poses; no MIND/MP-PCA
  was rerun.

## 2026-09-11 — Phase 4 review corrections before MLP offline work

- Quantitative field regularization now queries detached current
  `local → physical-RAS` group-pose coordinates, rather than scaled local
  slice coordinates.  This preserves the spatial-scaling normalization while
  ensuring regularization does not update rigid poses (including Stage A).
- `run_server_example.sh` and the README now require explicitly prepared SAX,
  2CH, and 4CH observation NPZ paths plus an output directory, and pass all
  three with repeated `--observations`.
- Regression tests: `test_quantitative_regularization_coordinates` and
  `test_end_to_end_tiny` PASS (3 tests).  No MATLAB parity or MIND processing
  was rerun.  Trad scientific behavior is frozen again after this correction.

## 2026-09-14 — Dual-environment deployment layer

- Added explicit deployment manifest/config examples, local-only MATLAB
  `local_preprocess_all.m`/`local_preprocess_one.m`, and per-stack
  `preprocess_qc.json`. They call frozen Module 01 with full-stack,
  figures-off, MP-PCA-on options and refuse overwrite.
- Added server-only transferred-v7.3 validation, exact manifest-driven
  observation preparation, batch QC, CUDA/environment readiness checking, and
  one-subject fixed SAX/2CH/4CH reconstruction wrapper. No MATLAB is invoked
  from server tools and no subject/stack discovery is performed.
- Server launcher now derives its root from its own location and uses
  `cr_dreme`; local smoke launchers remain unchanged. Trad scientific logic is
  unchanged.
