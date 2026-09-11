# Code_10w implementation status

Last updated: 2026-09-11

## Current boundary

Phase 3 Trad core forward model is complete. Project audit, scaffold,
corrected MATLAB preprocessing, MATLAB/Python bridge, timing, 10-weight group
dataset, RAS geometry, Quantitative INR, direct HHZ Trad signal simulation,
and rigid/PSF forward are implemented. MLP signal decoding/training, Trad
optimization loops, inference, and all later modules remain intentionally
unimplemented.

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

Await the next command. The next planned boundary is the remaining method
work specified by the pipeline; do not start a training loop or inference
without an explicit command.
