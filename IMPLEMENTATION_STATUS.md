# Code_10w implementation status

Last updated: 2026-09-10

## Current boundary

Phase 1 is complete. Only project audit, scaffold, upstream archiving, and
MATLAB preprocessing were implemented. Quantitative INR, signal simulators,
MLP, Python data bridge, rigid/PSF integration, training, and inference remain
intentionally unimplemented.

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

## Next-stage entry

Await the next command. The next planned implementation boundary is Module 02
(MATLAB/Python data bridge) only; no later Module 03–09 work has been started.
