# ismrm_v2_1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone `ismrm_v2_1` pipeline repository whose operator-facing code is organized into Arabic-numeral modules (`01_`–`09_`), with bilingual module manuals, while preserving the currently running subject jobs.

**Architecture:** Copy only the verified corrected pipeline's code dependency closure into `common/`, expose ordered operator modules under `pipeline/I_*` through explicit wrappers and selected source copies, and keep all subject data/results under ignored `subjects/` and `runs/` trees. Every wrapper resolves paths from the project root and records protocol, model, manifest, code and output SHA identities.

**Tech Stack:** Python 3.10, MATLAB batch, Bash, PyTorch/CUDA, NeSVoR, YAML/JSON/NPZ/NIfTI, pytest, Git.

**Spec:** `docs/superpowers/specs/2026-11-02-ismrm-v2-1-design.md`

## Global Constraints

- The project root is `/data/dengyz/code/ismrm_v2_1` on the server; current live runs under `scmr_v5_rebuild_20261102_corrected` are copied, never moved.
- Active protocol is SCMR-v5: FA `[45,45,45]`, TI `[10,100]`, T2-prep `[35,45,55]`, ramp-up `10`, TR `2.61 ms`, VPS `87`.
- Active decoder is the `[10,100]` checkpoint with SHA256 `338f2c0b8db90851f733ed31caa2c99914605fed3f7215ec2f69f19eba59583b`.
- B6 model SHA256 is `15981fbe93d2c70e14755ea756a02856c9383481e7d92d239619268a628a29b5`.
- Corrected protocol SHA256 is `3865b6bde1154750d4ed687456823fafad7b0cad5a61f763decb27dd1a4ed1c7`.
- No active module may import or shell out to `/data/dengyz/code/Code_10w` after migration.
- No patient data, large model files or large result trees are committed to GitHub; their local copies are ignored and tracked by manifests/SHA records.
- Arabic-numbered module names are operator-facing order; import-compatible package names remain under `common/` to avoid accidental algorithm changes.
- Every module has both `README_zh-CN.md` and `README_en.md` with file order, function, inputs, outputs, location, commands, checks and troubleshooting.

## Review Focus

- Correct decoder versus historical `[50,150]` decoder — `verify_runtime_contract.py` must reject the old protocol and accept only SHA `338f2c...`.
- L100 prepared-root identity versus profile-case prepared roots — sparse manifest validation tests must cover both forms.
- Server-only MATLAB/Python path resolution — wrapper tests must fail if an absolute `Code_10w` path remains.
- Copying live subject files without mutation — migration manifest tests must compare source/destination SHA and preserve source paths.
- Metrics/final-pose outputs for all five subjects — dry-run tests must enumerate HHZ, CYJ, DYZ, DYL_2 and HJL without subject-specific code.

---

### Task 1: Freeze the live corrected source record

**Files:**
- Create: `ismrm_v2_1/runtime/source_records/server_git_status.txt`
- Create: `ismrm_v2_1/runtime/source_records/live_run_contract.json`
- Create: `ismrm_v2_1/runtime/source_records/model_protocol_sha256.json`
- Create: `ismrm_v2_1/runtime/source_records/source_file_manifest.jsonl`
- Create: `scripts/capture_live_source_record.sh`

**Interfaces:**
- Consumes: server repository HEAD/status, current run roots, correct decoder, B6 model, protocol and subject manifests.
- Produces: immutable source commit/dirty-tree record, active model/protocol SHA record and copy manifest.

- [ ] **Step 1: Add a capture script that only reads live paths and writes a timestamped record.**
- [ ] **Step 2: Include all five subject names and current run-root paths, without stopping the live jobs.**
- [ ] **Step 3: Run the script on a test fixture and verify it never writes to a source path.**
- [ ] **Step 4: Record the actual server `HEAD` used by the current runs before copying code.**

### Task 2: Create the standalone repository skeleton and ignore policy

**Files:**
- Create: `/data/dengyz/code/ismrm_v2_1/.gitignore`
- Create: `/data/dengyz/code/ismrm_v2_1/VERSION`
- Create: `/data/dengyz/code/ismrm_v2_1/configs/`, `common/`, `pipeline/`, `runtime/`, `subjects/`, `runs/`, `docs/`, `tests/`
- Create: `scripts/bootstrap_project.sh`

**Interfaces:**
- Consumes: Task 1 source record.
- Produces: empty standalone Git repository with deterministic directories and data exclusion rules.

- [ ] **Step 1: Create the project root and all Arabic-numbered module directories.**
- [ ] **Step 2: Write `.gitignore` rules for DICOM, MAT, NPZ, NIfTI, model binaries, run logs and subject data while allowing manifests and README files.**
- [ ] **Step 3: Add `VERSION` with project schema and source commit identity.**
- [ ] **Step 4: Run a skeleton validator that confirms every required module and bilingual README placeholder exists.**

### Task 3: Materialize the import-compatible common code closure

**Files:**
- Copy: `common/trad/` from current corrected `trad/`
- Copy: `common/mlp/` from current corrected `mlp/` excluding outputs/cache
- Copy: `common/reconstruction_core/`
- Copy: selected `common/quality_experiments/`
- Copy: selected `common/paper_experiments/joint_vs_2dfit/`
- Create: `scripts/verify_no_legacy_code_path.py`

**Interfaces:**
- Consumes: Task 1 verified source commit and current code tree.
- Produces: import-compatible packages with no `_garbage`, legacy archive, unrelated synthetic experiments or old subject-only entrypoints.

- [ ] **Step 1: Build a dependency allowlist from the actual server entrypoints: preprocessing, bridge, sparse, fit-first, Proposed, final-pose and evaluation.**
- [ ] **Step 2: Copy only files in that allowlist, preserving package-relative imports.**
- [ ] **Step 3: Remove `__pycache__`, pytest caches, generated outputs and obsolete archived routes.**
- [ ] **Step 4: Run import compilation and `verify_no_legacy_code_path.py`; fail on active `/data/dengyz/code/Code_10w` references.**

### Task 4: Build ordered Arabic-numeral pipeline modules

**Files:**
- Create: `pipeline/01_RAW_DICOM/`
- Create: `pipeline/02_PREPROCESS_MATLAB/`
- Create: `pipeline/03_PREPARE_OBSERVATIONS/`
- Create: `pipeline/04_SPARSE_PROFILES/`
- Create: `pipeline/05_SCMR_V5_FIT_FIRST/`
- Create: `pipeline/06_PROPOSED/`
- Create: `pipeline/07_FINAL_POSE/`
- Create: `pipeline/08_EVALUATION_AND_VISUALIZATION/`
- Create: `pipeline/09_INTEGRITY_AND_REPORTING/`

**Interfaces:**
- Consumes: `common/` packages, project-root environment variables, subject manifest and module predecessor outputs.
- Produces: one explicit operator entrypoint per stage; each entrypoint accepts `PROJECT_ROOT`, `SUBJECT`, and stage-specific input/output roots.

- [ ] **Step 1: Map current scripts to modules:** DICOM/manifest; MATLAB preprocessing; observation bridge; sparse; SCMR/native cache and fit-first; Proposed; final-pose; evaluation/QC; integrity/reporting.
- [ ] **Step 2: Replace hard-coded `Code_10w` roots with project-root resolution and explicit arguments.**
- [ ] **Step 3: Ensure L100 and profile-case prepared-root semantics are preserved exactly.**
- [ ] **Step 4: Add dry-run mode to each wrapper that prints inputs, outputs, host, executable, protocol SHA and model SHA without running reconstruction.**
- [ ] **Step 5: Run shell syntax checks and Python compilation for every module.**

### Task 5: Add bilingual module manuals

**Files:**
- Create: `pipeline/01_RAW_DICOM/README_zh-CN.md`, `README_en.md`
- Create: `pipeline/02_PREPROCESS_MATLAB/README_zh-CN.md`, `README_en.md`
- Create: `pipeline/03_PREPARE_OBSERVATIONS/README_zh-CN.md`, `README_en.md`
- Create: `pipeline/04_SPARSE_PROFILES/README_zh-CN.md`, `README_en.md`
- Create: `pipeline/05_SCMR_V5_FIT_FIRST/README_zh-CN.md`, `README_en.md`
- Create: `pipeline/06_PROPOSED/README_zh-CN.md`, `README_en.md`
- Create: `pipeline/07_FINAL_POSE/README_zh-CN.md`, `README_en.md`
- Create: `pipeline/08_EVALUATION_AND_VISUALIZATION/README_zh-CN.md`, `README_en.md`
- Create: `pipeline/09_INTEGRITY_AND_REPORTING/README_zh-CN.md`, `README_en.md`

**Interfaces:**
- Consumes: Task 4 module wrappers and their actual command-line interfaces.
- Produces: operator manuals that do not rely on historical documentation.

- [ ] **Step 1: For each module, document purpose, execution order and server/local location.**
- [ ] **Step 2: List every code file in that module and explain its responsibility.**
- [ ] **Step 3: Document exact inputs, outputs, paths, manifest fields, SHA checks and pass markers.**
- [ ] **Step 4: Add copy-paste examples for HHZ and a generic subject.**
- [ ] **Step 5: Add a troubleshooting section for the known failures: wrong decoder protocol, partial output roots, manifest prepared-root mismatch, missing DICOM path and source-observation SHA mismatch.**
- [ ] **Step 6: Compare Chinese and English manuals for equivalent commands and path semantics.**

### Task 6: Materialize runtime configuration and model assets

**Files:**
- Create: `configs/protocol_scmr_v5_10_100.yaml`
- Create: `configs/subject_stack_manifest.example.json`
- Create: `configs/models_manifest.json`
- Create: `runtime/model_assets/decoder_scmr_v5_10_100.pth` (local ignored copy)
- Create: `runtime/model_assets/b6_model.pt` (local ignored copy)
- Create: `runtime/environment/cr_dreme_pip_freeze.txt`
- Create: `runtime/environment/nesvor_pip_freeze.txt`
- Create: `runtime/environment/matlab_version.txt`
- Create: `runtime/runtime_contract.json`

**Interfaces:**
- Consumes: correct decoder path/SHA, B6 path/SHA, corrected protocol path/SHA and server environment records.
- Produces: project-local runtime assets and an explicit contract that rejects the old `[50,150]` decoder.

- [ ] **Step 1: Copy the correct decoder only; assert SHA `338f2c...` and `ti_ms=[10,100]`.**
- [ ] **Step 2: Copy B6 and corrected protocol; assert their frozen SHA values.**
- [ ] **Step 3: Record MATLAB, Python and NeSVoR versions without embedding license identifiers.**
- [ ] **Step 4: Run `verify_runtime_contract.py` against both the correct and historical decoder fixtures.**

### Task 7: Copy five subject examples and live results after terminal state

**Files:**
- Create: `subjects/{HHZ,CYJ,DYZ,DYL_2,HJL}/...`
- Create: `runs/{HHZ,CYJ,DYZ,DYL_2,HJL}/...`
- Create: `runtime/source_records/migration_manifest.jsonl`
- Create: `docs/migration_record.md`

**Interfaces:**
- Consumes: HHZ archived reference and completed live subject run roots.
- Produces: project-local copies with source/destination SHA, copy time, source commit and terminal status.

- [ ] **Step 1: Copy HHZ verified input/ROI/native-cache/reference outputs first.**
- [ ] **Step 2: After each live subject reaches terminal status, copy its raw/preprocessed/prepared/ROI/sparse inputs and reconstruction/evaluation outputs without moving the source.**
- [ ] **Step 3: Generate per-file SHA entries and compare source/destination bytes.**
- [ ] **Step 4: Mark incomplete or failed cases explicitly; never present them as completed examples.**

### Task 8: Generate project-level bilingual manuals and verification commands

**Files:**
- Create: `docs/README_zh-CN.md`
- Create: `docs/README_en.md`
- Create: `docs/pipeline_contract.md`
- Create: `scripts/run_pipeline.sh`
- Create: `scripts/verify_project_integrity.py`
- Create: `scripts/verify_runtime_contract.py`
- Test: `tests/test_project_layout.py`

**Interfaces:**
- Consumes: all ordered module manuals, runtime contract and migration manifest.
- Produces: beginner-readable end-to-end instructions and independent checks.

- [ ] **Step 1: Write the top-level Chinese manual from DICOM to metrics with a command for every module.**
- [ ] **Step 2: Write the English manual with equivalent commands and warnings.**
- [ ] **Step 3: Document the five-subject example paths and the difference between tracked metadata and ignored data/results.**
- [ ] **Step 4: Add a dry-run pipeline command that resolves all paths without launching MATLAB/CUDA.**
- [ ] **Step 5: Add layout, path-independence and SHA contract tests.**

### Task 9: Final verification and repository initialization

**Files:**
- Modify: all files created by Tasks 1–8 as required by verification.
- Create: `runtime/source_records/final_verification.json`

**Interfaces:**
- Consumes: completed project and migration manifest.
- Produces: verified initial commit ready for a new GitHub remote.

- [ ] **Step 1: Run Python compilation, shell syntax checks and targeted tests.**
- [ ] **Step 2: Run project integrity and runtime contract checks.**
- [ ] **Step 3: Run dry-run entrypoints for all five subjects and every module.**
- [ ] **Step 4: Confirm no live run process was interrupted and no source path was modified.**
- [ ] **Step 5: Initialize the new Git repository and commit only code, configs, manuals, templates, tests and metadata.**
- [ ] **Step 6: Record the initial commit and GitHub setup instructions; do not push without explicit user instruction.**
