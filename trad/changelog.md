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
