# Q001 Controlled Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add controlled Q001A full-FOV and Q001B full-FOV-registration/cropped-reconstruction server-ready routes without starting either reconstruction.

**Architecture:** A new `quality_experiments/Q001_recon` package owns immutable input/provenance validation, B6-derived config diffs, Q001B rigid-delta transfer, native-reference packaging, and evaluation metadata. It delegates model training/export to the existing FrozenMLP code only at explicit server execution time. Shell commands only migrate/verify inputs or invoke the package after all guards pass.

**Tech Stack:** Python 3.9, PyTorch, NumPy, existing NeSVoR `RigidTransform`, MATLAB original MultiMap dictionary matcher, Bash/rsync.

**Spec:** `/home/universe/.codex/attachments/4bc67f77-7554-4214-9b3b-d546924a220d/pasted-text.txt`

## Global Constraints

- No local CPU formal reconstruction, automatic server reconstruction, D3, Q002, push, baseline trainer/exporter modifications, resizing, interpolation, or B6 reconstruction warm-start.
- B6 `model.pt` `resolved_config` is the sole reconstruction configuration source; only documented Q001 route metadata/input overrides are permitted.
- Q001A optimizes full-FOV observations only; Q001B uses full-FOV observations only for stack registration and optimizes cropped observations only.
- Q001B transfers `delta.compose(dicom_group_pose)` with `RigidTransform`, rotation-first axis-angle and `trans_first=True`.
- Every output root refuses non-empty content; formal results are never claimed locally.

## Review Focus

- A malformed or hash-mismatched input manifest must fail before migration or reconstruction; Task 1 tests it.
- An unauthorized B6 configuration difference or B6 reconstruction state must fail; Task 2 tests it.
- A Q001B route must not route full observations into optimization or invoke registration twice; Task 3 tests it.
- World points, centers, and normals must transform identically across full/cropped representations; Task 3 tests it.
- Evaluation must use fixed zero-based ROIs and one final-checkpoint hash across central/K32/K8 hooks; Task 4 tests it.

---

### Task 1: Input contract and server migration verifier

**Files:** Create `quality_experiments/Q001_recon/contracts.py`, `quality_experiments/Q001_recon/tests/test_contracts.py`; create server verifier/migration commands.

- [ ] Write failing tests for READY manifest hashes, exact CYJ shapes/counts and tampered input rejection.
- [ ] Implement manifest validation and read-only SHA256 verifier.
- [ ] Run the contract tests and compile checks.

### Task 2: B6-derived configuration contract

**Files:** Create `quality_experiments/Q001_recon/config.py`, tests, Q001A/B config templates.

- [ ] Write failing synthetic-checkpoint tests for B6 config load, allowed route diffs, frozen decoder, and warm-start rejection.
- [ ] Implement B6 resolved-config loader and approved-diff validator.
- [ ] Run the config tests.

### Task 3: Q001A/Q001B route and physical pose transfer

**Files:** Create `quality_experiments/Q001_recon/routes.py`, `pose_transfer.py`, tests, server run commands.

- [ ] Write failing tests for route-only observations, one registration call, and point/center/normal rigid consistency.
- [ ] Implement route planning and physical delta composition using vendored `RigidTransform`.
- [ ] Run route/pose tests.

### Task 4: Native reference and evaluation hooks

**Files:** Create Q001 full-FOV MATLAB wrapper/reference packager and evaluation metadata/ROI module with tests; create evaluate commands.

- [ ] Write failing tests for exact ROIs, no resize metadata, PSF K32/K8 metadata, and final-checkpoint identity.
- [ ] Implement original-MultiMap full-FOV reference wrapper and Q001 evaluation plans.
- [ ] Generate the local full-FOV reference when the real MAT/matcher are available; run tests.

### Task 5: Documentation, state, and final verification

**Files:** Modify experiment ledger/state; create completion report.

- [ ] Record ready v2 inputs and server-migration-pending state without claiming reconstruction results.
- [ ] Run targeted tests, baseline regression, CLI/help/shell syntax, diff checks; commit only phase files.
