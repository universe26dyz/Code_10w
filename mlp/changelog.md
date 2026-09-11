# Changelog

## 2026-09-10 — Full pipeline v1 scaffold / Phase 1

- Created the independent Phase-1 scaffold: `configs/`, `third_party/`, all
  Module 01–09 directories, `scripts/`, `tests/`, and `outputs/.gitkeep`.
- Vendored NeSVoR `master` commit `2e96a91bdd30174210caea911e03a2778c65adbe`
  as pure Python source plus its LICENSE/README; `third_party` contains no
  nested Git repository.
- Vendored only the three permitted mDM MLP reference files from `master`
  commit `a24ab1008d48e92ddf6f2bb8131f9a58dda23e95`; no EPG implementation
  was copied as a formal model.
- Copied the unmodified HHZ MATLAB source subset into
  `modules/module_01_preprocess_matlab/hhz_original/`.
- Added the shared `preprocessing_v1/preprocess_stack_v1.m` and explicit
  options. It preserves HHZ NumImg=10, FA=[45,45,45], TI=[50,150] ms,
  T2prep=[35,45,55] ms, reads TR/VPS from DICOM, saves bridge metadata, and
  assigns `Mag_crop = MIND_mag_reg`.
- Recorded the authoritative HHZ TI=[50,150] vs DYZ legacy TI=[10,100]
  difference in the protocol and README; the DYZ value is not inherited.
- Tests: `find trad mlp -type d -name .git -print` PASS (no output);
  `conda run -n knesvr_torch python -m compileall trad mlp` PASS; MATLAB
  `preprocess_stack_v1` entry/arity check PASS.
- Not run: DICOM/MIND/MP-PCA data smoke, by Phase-1 test scope; no MLP
  training, full-subject processing, or later Python pipeline was started.

## 2026-09-11 — Phase 1 review correction and Phase 2 bridge/dataset sync

- Synchronized the independently runnable Phase 2 common code from Trad:
  corrected preprocessing, Module 02 bridge/timing/crop geometry, Module 03
  `QuantPointDataset`, tests, and module documentation.
- Corrected preprocessing order to `crop → MIND → optional MP-PCA → Mag_crop`;
  `Mag_crop_unregistered`, `MIND_mag_reg`, and final `Mag_crop` now have
  explicit non-overlapping semantics. `MPPCA_options.center_data=true` remains
  explicit and all MP-PCA options are passed without internal defaults.
- Added constant TR/VPS, complete 10-weight geometry, SOP/Series UID, and
  portable crop-offset metadata validation. The bridge resolves only current
  `--dicom-dir` SOPInstanceUID mappings.
- No MLP, teacher, Quantitative INR, signal decoder, PSF, rigid training, or
  reconstruction code was started. Version/import consistency is checked after
  this sync; the real smoke is intentionally not repeated for identical code.

## 2026-09-11 — Phase 2 geometry correction / shared Phase 3 core sync

- Synchronized the corrected local-coordinate Module 02/03 geometry contract:
  cropped HB1 DICOM-LPS provenance, explicit RAS conversion, one initial
  NeSVoR group pose per complete 10-weight slice, multi-stack global group
  reindexing, preserved stack IDs, and strict IOP validation.
- Synchronized common Module 04 Quantitative INR and Module 06 rigid/local
  anisotropic-PSF code and documentation from Trad. The copied code remains
  independently importable and does not add an MLP signal decoder, teacher,
  training loop, or inference implementation.
- Validation: common-module import PASS and synchronized geometry, dataset,
  INR, and rigid/PSF source files are byte-identical to Trad. The MLP full
  suite and preprocessing/MIND smoke were intentionally not repeated.

## 2026-09-11 — Shared Phase 4 infrastructure sync only

- Synchronized the review-corrected common Module 02 TR/VPS bridge, Module 03
  NeSVoR-margin/protocol metadata dataset, Module 04 native NeSVoR backend and
  spatial-scaling INR, Module 06 `axisangle_init`/transformation regularizer,
  and reusable training-space coordinate helper.
- Validation: import PASS and these synchronized files are byte-identical to
  Trad. No Trad signal decoder, MLP decoder/training, preprocessing/MIND smoke,
  or MLP full suite was run in this phase.

## 2026-09-11 — Phase 5 MLP offline HHZ surrogate

- Synchronized the fixed v1 protocol, bridge/dataset, Quantitative INR, and
  rigid/PSF common sources with Trad and copied only the HHZ-compatible Trad
  signal simulator as the MLP teacher; no EPG path was introduced.
- Added strict prepared-NPZ timing-pool construction: complete ten-weight
  groups, unique timing rows, identical TR (tolerance) and VPS (exact), with
  source/group provenance and clear mismatch errors.
- Added one-time HDF5 synthetic generation split by timing vector, not sample;
  it uses continuous T1/T2/B1 ranges and normalized HHZ teacher targets only.
- Added the exact mDM-style 12→200→200→200→10 MLP with BatchNorm/
  LeakyReLU, Adam/StepLR training, best-checkpoint metadata, held-out signal
  metrics, and derivative comparison.  Frozen inference keeps BN in eval
  mode and preserves input gradients.
- Tests: requested offline suite plus teacher fidelity PASS (7 tests).  One
  CPU tiny training run PASS (1024/256/256, 2 epochs), producing the best
  checkpoint, history, per-weight/overall error and derivative metrics.  Its
  one-real-timing input required an explicitly labelled functional fixture, so
  it is a code-path test rather than a cross-timing generalization result.
- Not implemented: MLP online reconstruction, decoder replacement in the
  rigid/PSF forward, end-to-end MLP training, or any reconstruction benchmark.

## 2026-09-11 — Phase 5 review corrections / Phase 6 online integration

- Reworked offline HDF5 training to load each split once into CPU-RAM
  `TensorDataset`; training BatchNorm requires batch≥2 and `drop_last=true`.
  Fidelity transfers values batchwise and only gradient samples to device.
- Added explicit teacher device, timing-pool `functional_fixture`, timing-pool
  SHA256, timing min/max, protocol/split provenance and strict dataset/pool/
  protocol/checkpoint compatibility checks.
- Added strict checkpoint loader and Trad-compatible frozen decoder signature.
  Formal online config rejects functional-fixture checkpoints; only a fixed
  1e-4 ms float32 storage-boundary tolerance is accepted.
- Added MLP online reconstruction by minimally reusing the frozen Trad staged
  INR/rigid/PSF/export/QC chain. Frozen MLP parameters are absent from optimizer
  groups and BatchNorm statistics are checked unchanged. Added separate recon
  configs, explicit SAX+2CH+4CH launcher and decoder-only benchmark.
- Validation: targeted offline regression PASS (3 files after fixture repair);
  online suite PASS (5 tests); one CPU online functional smoke and one CPU
  benchmark PASS. The tiny checkpoint is `scientific_checkpoint=false` and
  does not establish scientific reconstruction quality.

## 2026-09-11 — Phase 7 formal-validation readiness

- Added explicit manifest-derived formal timing pools retaining every group’s
  `subject_id`, stack and source provenance; numeric timing duplicates are not
  deduplicated for subject validation. Added subject-disjoint 3/1/1 split,
  timing-domain metadata, formal timing audit and held-out validation CLI.
- Candidate checkpoints are now `formal_candidate` with
  `validation_status=unvalidated`; training no longer auto-approves a real
  pool. Online loader rejects any unapproved formal candidate. Functional
  smoke still needs its explicit allowance.
- Decoder benchmark now resets and records Trad/MLP CUDA peaks separately and
  writes forward/backward speedups; CPU RSS is diagnostic only.
- Targeted tests PASS (4/4). Formal execution was not started: this host has
  no CUDA and no explicit multi-subject prepared manifest, so no protocol
  audit, generation, candidate training, validation or GPU benchmark was
  fabricated or replaced with a fallback.
