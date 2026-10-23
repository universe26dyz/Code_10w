# Phase4R Final Multi-subject Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add fail-closed contracts that turn an explicitly selected CYJ profile/method into a reproducible five-subject standardized reconstruction and reporting pipeline.

**Architecture:** Keep scientific selection explicit and immutable. Standardization and native-map reuse remain separate evidence-producing stages; baseline consumes validated reuse outputs without refitting. Server wrappers only construct and audit pinned future commands, with dry-run as default.

**Tech Stack:** Python, JSON manifests with SHA256, Bash wrappers, NumPy, pytest.

**Spec:** `/home/universe/.codex/attachments/3470ac59-046f-4679-acbd-c43b11b7fef2/pasted-text.txt`

## Global Constraints

- Local-only, CPU tests only; never execute GPU, NeSVoR, formal reconstruction, or server commands.
- Do not mutate frozen manifests, R0 cache, checkpoints, B6, decoder, protocol, or historical results.
- All formal wrappers default to `--dry-run`, require explicit `--execute`, exact SHA pins, and fresh outputs.
- Preserve physical-ordinal selection, independent T1/T2 pose policy, endpoint semantics, and paired-support semantics.

## Review Focus

- Q005 must enforce exactly the same sparse/B6/decoder/protocol admission as Q002S640; test every absent or bad pin.
- A selected profile must materialize by local physical ordinal rather than CYJ numeric group IDs; test distinct subject IDs and fold policy.
- Standardized reuse must be validated from artifact contents and hashes, not `REUSE_VERIFIED`; test mapping/content/geometry/SHA corruption.
- Baseline standardized mode must never call dictionary fitting and must use only SAX parameter-matched output sample masks.
- Population pairing must reject endpoint/support mismatches and treat subjects, not folds or voxels, as independent units.

---

### Task 1: Formal Q002/Q005 admission and K4 runner

**Files:** `run_q002_reconstruction.py`, `run_cyj_experiment_1_2_k4_psf_v2.sh`, Q005 tests.

- [ ] Write failing tests for Q005 formal pins and dry-run/execute command admission.
- [ ] Generalize the formal sparse validator to Q002S640 and Q005S640_K4 and keep legacy routes unchanged.
- [ ] Implement the Q005 wrapper’s read-only preflight, explicit execute command, and post-execution audit contract.

### Task 2: Explicit final method selection and common-support evidence

**Files:** create `final_method_selection.py`, create method-selection evaluator and tests.

- [ ] Define immutable v1 selection schema that accepts explicit profile/method/evidence only.
- [ ] Validate evidence hashes, method/profile/fold policy, dependency hashes, and never infer a winner from folders or metrics.
- [ ] Produce central/PSF128, T1/T2-separated three-method common-support metrics for CYJ selection evidence.

### Task 3: Standardization wrappers and profile materialization

**Files:** standardization wrapper, create selected-profile materializer, cohort config, tests.

- [ ] Preserve existing standardization algorithm; add execute preflight/post-validation wrapper.
- [ ] Materialize the selected frozen profile on each standardized subject via `build_sparse_profile_condition` with explicit fold policy.
- [ ] Add the canonical five-subject config without changing historical configs.

### Task 4: Strict standardized dictionary reuse and Fit-first consumer

**Files:** reuse validator, `baseline.py`, tests.

- [ ] Add public strict validator for reuse manifests and every referenced output artifact.
- [ ] Add mutually exclusive standardized-reuse baseline mode that validates, never refits, exports inputs, and records all provenance.
- [ ] Reuse the existing parameter-matched SAX sample-mask command path for standardized Fit-first.

### Task 5: Pinned reconstruction orchestrator and readiness registry

**Files:** standardized reconstruction wrapper, readiness registry, tests.

- [ ] Require final selection, standardization, reuse, profile materialization and dependencies; default read-only dry-run.
- [ ] Remove stale handoff/DNS text as an independent gate while retaining it as historical evidence.
- [ ] Open readiness only when every subject’s verified evidence and SHA pins pass.

### Task 6: Standardized native provenance and population aggregation

**Files:** native-reference/evaluation adapters, create `aggregate_final_multi_subject.py`, tests.

- [ ] Allow standardized authoritative native provenance without fabricating legacy sparse manifests.
- [ ] Aggregate five immutable subject summaries by subject with endpoint/parameter/metric separation and audited paired support.
- [ ] Emit CSV/JSON manifest artifacts and reject incomplete or incompatible inputs.

### Task 7: Verification and commit

- [ ] Run all task-specific CPU tests plus the requested regression suite.
- [ ] Run compileall, bash syntax validation, and `git diff --check`.
- [ ] Create one local commit and do not push.
