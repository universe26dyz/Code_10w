# CYJ Sparse K4 and Common Slice Standardization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add immutable CYJ sparse-profile/K4 experiment preparation and later all-subject 12/15/10 standardized-input tooling without launching reconstruction.

**Architecture:** Keep legacy Phase-4R inputs and summaries unchanged. Add a versioned custom-profile manifest route, route adapters that consume its explicit identities, and independent immutable standardization/reuse-audit tools. Server wrappers only emit/validate commands and never start GPU work unless explicitly invoked outside this code-change window.

**Tech Stack:** Python 3, NumPy/NPZ, JSON manifests, existing PyTorch CPU-test environment, Bash wrappers.

**Spec:** `/home/universe/Downloads/CODEX_11_12_CYJ_sparse_K4_slice_standardization_v2.md`

## Global Constraints

- Use only CPU tests/static checks; do not run CYJ training, NeSVoR, GPU work, preprocessing, dictionary matching, or formal data outputs.
- Preserve old `phase4r_sparse_v1` readers and frozen Q002/Q004/K256 identities and controls.
- New artifacts default beneath `phase4r_v2_common_slice_grid_20261022`, never beneath `paper_joint_vs_2dfit`.
- Existing prepared NPZs, DICOMs, maps, checkpoints, and outputs are read-only inputs.
- All created outputs are immutable; dry-run writes no experiment output.
- Every server wrapper defaults to dry-run and must not launch GPU training, NeSVoR, or write experiment output without an explicit external execute action.
- Commit locally only; do not push, reset, clean, or remove user files.

## Review Focus

- A non-contiguous/non-zero-based source `group_idx` must use audited physical ordinal, not `group_idx % stride` (Task 1 test).
- A `none` stack must have zero training rows but exhaustive held-out identities and no fabricated observation (Tasks 1–2 tests).
- K4 must differ from Q002/K8 only in `training_psf_samples=4`, consume exactly 2,560 INR locations per 640-anchor batch, and preserve renderer/sample-PSF, endpoint PSF, and all other controls (Task 3 test).
- 16→15 trimming must normalize stack-normal sign and drop the high signed-coordinate endpoint only after valid spacing/contiguous geometry audits (Task 4 test).
- Reused dictionary data must fail on any per-group content/geometry mismatch, despite a different standardized NPZ SHA (Task 5 test).

---

### Task 1: Versioned sparse-profile resolver and immutable derived inputs

**Files:**
- Modify: `paper_experiments/joint_vs_2dfit/phase2_clean/sparse.py`
- Create: `paper_experiments/joint_vs_2dfit/configs/sparse_profiles/CYJ_*.json`
- Test: `paper_experiments/joint_vs_2dfit/tests/test_sparse_profiles_v2.py`

**Interfaces:**
- Produces: `resolve_sparse_profile(...) -> dict`, `build_sparse_profile_condition(...) -> dict` and `phase4r_sparse_profile_v2` manifests.
- Consumes: existing prepared NPZ fields and HB1 physical geometry audit.

- [ ] Write failing tests for A/B complementary ordinal folds, C/D/E all/none stacks, CYJ 14/15/12 counts, override precedence, dry-run no-write, and legacy builder compatibility.
- [ ] Run the new test module and confirm failure before implementation.
- [ ] Implement explicit profile/case/fold identity, mode/stride selection by spatial ordinal, complete group/weight validation, empty-stack omission, immutable output/provenance, and CLI dry-run/printed resolved configuration.
- [ ] Add the five profile JSON files and validate all seven cases via CPU fixtures.
- [ ] Run sparse-profile and legacy sparse tests; record result.

### Task 2: Custom-profile consumers and zero-stack training preparation

**Files:**
- Modify: `paper_experiments/joint_vs_2dfit/phase2_clean/q004s640.py`, `final_sparse_native_metrics.py`, `final_stratified_native_metrics.py`, `final_cyj_reporting.py`, `build_final_subject_figure_configs.py`
- Modify: `quality_experiments/Q002_B6_roi_same_anchor_mse/contracts.py`, dataset/training entry points
- Test: existing and new Phase-2/Q002 tests

**Interfaces:**
- Consumes: Task 1 explicit `case_id`, `profile_id`, stack membership and prepared-root provenance.
- Produces: case-aware adapter validation, profile-isolated aggregation, and datasets/losses that accept one to three active stacks.

- [ ] Write failing tests for arbitrary eligible profile IDs, fully held-out/no-held-out metrics semantics, all-source labels, active-stack loss normalization without fake rows, Fit-first preservation of all source dictionary/reference groups, and explicit rejection of unsupported active-stack configurations.
- [ ] Run targeted tests and confirm failure before implementation.
- [ ] Implement narrow legacy fallback only for v1 conditions; custom manifests use explicit identities exclusively.
- [ ] Make loaders omit empty stacks; validate non-empty active set and renormalize only active existing stack weights. Fully held-out stacks remain evaluable against the full source geometry; no unsupported configuration may silently fall back.
- [ ] Run affected Phase-2 and Q002 tests; record result.

### Task 3: Q005S640 K4 route and dry-run-only 1.2 command

**Files:**
- Modify: `quality_experiments/Q002_B6_roi_same_anchor_mse/contracts.py`, `training.py`, `run_q002_reconstruction.py`
- Create/Modify: a Q005 K4 entry point and `server_commands/run_cyj_experiment_1_2_k4_psf_v2.sh`
- Test: `quality_experiments/Q002_B6_roi_same_anchor_mse/tests/test_q005s640_k4.py`

**Interfaces:**
- Produces: `Q005S640_K4_B6_roi_same_anchor_mse` resolved route with `training_psf_samples=4`.
- Consumes: unchanged frozen Q002 controls and endpoint settings.

- [ ] Write failing tests comparing K4 against Q002/K8 resolved controls and asserting that `training_psf_samples=4` is the only intended scientific delta, location count is 2,560, and renderer/sample-PSF, endpoint PSF, and every other training control match the comparator constraints.
- [ ] Implement K4 as a distinct route; separate training PSF, endpoint PSF, renderer PSF, and forward chunk fields.
- [ ] Add dry-run command generation only; no K4 launch is permitted in this task.
- [ ] Run Q005/Q002/Q004/K256 contract tests and confirm frozen routes are unchanged.

### Task 4: Physical 12/15/10 slice standardization

**Files:**
- Create: `paper_experiments/joint_vs_2dfit/phase2_clean/standardize_subject_slices.py`
- Create: `paper_experiments/joint_vs_2dfit/tests/test_standardize_subject_slices.py`
- Create: `server_commands/standardize_subject_inputs_12_15_10.sh`, `server_commands/audit_standardized_subject_inputs.sh`

**Interfaces:**
- Produces: immutable standardized NPZs and `slice_standardization_manifest.json` with source-original-to-new identity mapping.
- Consumes: full prepared observation NPZ geometry and group-independent preprocessing contract.

- [ ] Write failing geometry fixtures for physical sorting, dominant-axis normal-sign normalization, even symmetric trim, deterministic 16→15 high-coordinate drop, spacing/contiguity rejection, five-subject 12/15/10 target validation, and unchanged source SHA.
- [ ] Implement geometry projection/sign normalization, contiguous selection, complete 10-weight row subsetting, group reindexing, five-subject target validation, DYL_2 `READY_FOR_STANDARDIZATION` gate, and immutable/dry-run behavior.
- [ ] Run standardizer tests and shell syntax checks.

### Task 5: Dictionary cache reuse audit and reconstruction runbook controls

**Files:**
- Create: `paper_experiments/joint_vs_2dfit/phase2_clean/reuse_standardized_native_dictionary_maps.py`
- Create: tests for reuse audit
- Create: `server_commands/run_standardized_subject_reconstruction.sh`

**Interfaces:**
- Consumes: Task 4 mapping plus historical cache/maps and per-group source content.
- Produces: reindexed native cache with `REUSE_VERIFIED`/`RECOMPUTE_REQUIRED`/`BLOCKED` provenance.

- [ ] Write failing tests for per-group fingerprint mismatch and valid reindex/subset preservation.
- [ ] Implement strict cache/map verification and subset writer; never infer reuse from subject name or whole-file SHA alone. Only decoder/B6 artifacts may be reused after SHA256 verification.
- [ ] Add status/dry-run command checks and CPU tests. Require fresh Proposed reconstruction, Fit-first T1/T2 reconstruction, poses, central/psf128 endpoints, native bundles, support masks, metrics, figures, and summaries after standardization; old 3D-derived artifacts are never reusable.

### Task 6: Server status wrappers, runbooks, and Prompt 01 review

**Files:**
- Create: requested 1.1 status/run wrappers and docs listed in the task specification
- Test: shell syntax and dry-run fixture tests

**Interfaces:**
- Consumes: Task 1 profile manifests, Task 3 K4 resolved route, Tasks 4–5 standardized/reuse manifests.
- Produces: sequential, immutable, stop-on-partial/mismatch command plans only.

- [ ] Write dry-run/status tests proving no GPU command or output write occurs without explicit execute mode, including all standardization and reconstruction wrappers.
- [ ] Implement wrappers and all required design/runbook docs, including `PROMPT_01_IMPLEMENTATION_REVIEW.md`.
- [ ] Run Bash syntax checks and targeted docs/config validation.

### Task 7: Integrated verification and local commit

**Files:**
- Test: all touched Phase-2 and Q002 suites

- [ ] Run the mandated compileall, both test suites, diff check, and Bash syntax checks; isolate unrelated baseline failures without changing XCAT fixtures.
- [ ] Inspect staged paths to ensure only code/docs/tests are included and no generated experiment output is staged.
- [ ] Commit the complete phase locally with the specified commit message; do not push.

## Self-review

- Spec coverage: Tasks 1–2 cover 1.1 profiles and consumers; Task 3 covers K4; Tasks 4–5 cover standardization/reuse; Task 6 covers wrappers/docs; Task 7 covers verification.
- Interface consistency: custom profile manifests are produced in Task 1 before every consumer in Task 2; standardized mapping is produced in Task 4 before reuse in Task 5.
- Ruling: the supplied MD is the design specification; the work should execute inline because interfaces are tightly coupled and the repository rules prohibit subagents.
