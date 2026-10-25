# Final5 Legacy ROI Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive auditable Final5 cropped SAX ROI bundles from the four immutable historical full-FOV bundles without mask resampling.

**Architecture:** A new AHA adapter module consumes only explicit historical ROI, Final5 SAX geometry, and group-mapping paths.  It verifies an integer physical crop per source plane, writes a new immutable ROI bundle, and records all hashes.  Final5 readiness resolves only that derived bundle; evaluation continues to use one shared ROI and native lattice for Proposed and Fit-first.

**Tech Stack:** Python, NumPy, nibabel, pytest.

**Spec:** `docs/superpowers/specs/2026-10-25-final5-legacy-roi-adaptation.md`

## Global Constraints

- Do not modify reconstruction, fitting, endpoints, native maps, sparse manifests, formal metrics, or historical ROI inputs.
- Accept only explicit file paths and manifest-declared group identities; no recursive discovery or inferred layer order.
- Permit integer crop and declared discrete orientation only; reject resampling and clipped landmarks/myocardium.
- One derived ROI is shared by Proposed and Fit-first.
- Commit locally only; never push.

## Review Focus

- A crop with non-integral physical offset must be rejected; Task 1 test `test_derivation_rejects_non_integer_crop`.
- An omitted legacy landmark must prevent bundle publication; Task 1 test `test_derivation_rejects_landmark_outside_crop`.
- CYJ-like standardized groups must follow manifest identity rather than slice number; Task 2 test `test_mapping_uses_manifest_group_identity`.
- Subject-root resolution must not silently return legacy full-FOV ROI; Task 2 test `test_final5_root_does_not_fallback_to_legacy_roi`.
- Both endpoints must retain the same ROI support; existing paired-support regression remains in the final AHA test suite.

### Task 1: Strict cropped-ROI derivation

**Files:**
- Create: `paper_experiments/joint_vs_2dfit/aha/final5_legacy_roi.py`
- Create: `paper_experiments/joint_vs_2dfit/aha/run_final5_legacy_roi.py`
- Test: `paper_experiments/joint_vs_2dfit/tests/test_final5_legacy_roi.py`

**Interfaces:**
- Produces `derive_final5_cropped_roi(*, subject, legacy_roi, final5_sax_geometry, output) -> Path`.
- Produces a bundle containing the four canonical ROI files plus `roi_derivation_manifest.json`.

- [ ] Write failing synthetic tests for exact integer crop, non-integral crop rejection, and landmark clipping.
- [ ] Run the focused test and confirm failures arise from the missing derivation interface.
- [ ] Implement explicit geometry/group loading, exact per-plane crop, provenance hashing, immutable output, and CLI.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Final5-only resolution and group binding

**Files:**
- Modify: `paper_experiments/joint_vs_2dfit/aha/final5_adapter.py`
- Modify: `paper_experiments/joint_vs_2dfit/aha/final5_aha16.py`
- Modify: `paper_experiments/joint_vs_2dfit/tests/test_final5_aha16_adapter.py`
- Test: `paper_experiments/joint_vs_2dfit/tests/test_final5_legacy_roi.py`

**Interfaces:**
- Consumes Task 1 derived bundle manifest.
- Produces strict Final5 ROI resolution and validates a mapping's derived ROI provenance.

- [ ] Write failing tests proving subject-root lookup rejects legacy-only bundles and mapping requires the derivation manifest.
- [ ] Run the focused tests and confirm they fail for the missing Final5-only behavior.
- [ ] Implement the minimal resolver/mapping validation changes without altering paired support or endpoint reprojection.
- [ ] Run focused adapter and derivation tests and confirm they pass.

### Task 3: Local preparation and verification

**Files:**
- Modify: `docs/evaluation/AHA_LOCAL_RUNBOOK.md`
- Test: `paper_experiments/joint_vs_2dfit/tests/test_final5_legacy_roi.py`

**Interfaces:**
- Documents explicit server-to-local input paths and a dry-run that publishes nothing when those paths are absent.

- [ ] Add a dry-run test that reports missing explicit Final5 geometry without falling back to legacy artifacts.
- [ ] Add the exact local/server invocation and output layout to the runbook.
- [ ] Run all AHA tests, `compileall`, and `git diff --check`.
- [ ] Commit the completed implementation locally without push.
