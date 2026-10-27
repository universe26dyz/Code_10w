# HHZ Single-method Final-pose Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate immutable HHZ single-method final-pose AHA16 and myocardial QA before the paired method is available.

**Architecture:** Add one standalone downstream module beside the existing paired AHA code. It consumes caller-supplied current native/prepared paths plus either Fit-first or Proposed final-pose samples, validates a current mapping to the historical ROI bundle, and writes method/reference statistics and figures under an explicitly diagnostic output root. It never embeds an `ismrm` analysis root.

**Tech Stack:** Python, NumPy, Nibabel, Matplotlib, existing AHA/ROI helpers.

**Spec:** `docs/superpowers/specs/2026-10-27-hhz-fit-first-only-final-pose-design.md`

## Global Constraints

- Do not change paired Proposed-vs-Fit-first modules or their output schema.
- Fit-first samples must be final-pose `prediction_ms` with `valid`; Proposed samples must be final-pose `{parameter}_ms`; no endpoint substitute.
- Mapping and ROI asset identity gates must fail closed.
- Results are diagnostic only and immutable.

## Review Focus

- A historical mapping whose prepared-SAX SHA does not match the corrected current prepared root must be rejected.
- A valid native map pixel with an invalid method final-pose sample must be excluded, not interpolated or replaced.
- Per-plane AHA and pooled myocardial outputs must record the same support definition.
- T1/T2 arrays with group or shape mismatch must stop before publishing output.
- Existing paired APIs must remain unchanged.

---

### Task 1: Single-method plane loading and support contract

**Files:**
- Create: `paper_experiments/joint_vs_2dfit/aha/hhz_fit_first_only.py`
- Test: `paper_experiments/joint_vs_2dfit/tests/test_hhz_fit_first_only.py`

**Interfaces:**
- Consumes: native SAX maps and either Fit-first or Proposed plane archives.
- Produces: `load_single_method_plane(...) -> dict[str, np.ndarray]`, with `reference`, `prediction`, `valid`, and strict shape/schema validation.

- [x] **Step 1: Write failing loader tests** for valid final-pose support and for missing/mismatched plane arrays.
- [x] **Step 2: Run** the test and observe missing-module failure.
- [x] **Step 3: Implement the minimal plane loader** that validates either native-bundle schema, group identity, plane shape, method validity, and finite values.
- [x] **Step 4: Run the targeted tests** and confirm pass.

### Task 2: Diagnostic AHA16 and myocardial writers plus CLI

**Files:**
- Modify: `paper_experiments/joint_vs_2dfit/aha/hhz_fit_first_only.py`
- Modify: `paper_experiments/joint_vs_2dfit/tests/test_hhz_fit_first_only.py`

**Interfaces:**
- Consumes: Task 1 plane data, `require_cyj11_mapping`, SCMR ROI bundle, `legacy_aha16` labels/statistics/renderers.
- Produces: `build_single_method_final_pose(...) -> Path` and a `python -m` CLI that writes immutable AHA16, myocardial, figures, and manifest artifacts.

- [x] **Step 1: Write failing writer tests** asserting diagnostic manifest flags, final-pose sampling, shared support count, and refusal to overwrite.
- [x] **Step 2: Run the targeted test file** and observe the expected failure.
- [x] **Step 3: Implement writers** that reuse existing ROI/AHA primitives and compute either Fit-first-vs-reference or Proposed-vs-reference metrics without a paired input.
- [x] **Step 4: Run targeted tests** and confirm pass.
- [x] **Step 5: Run the relevant existing AHA test file once** to verify paired code remains green.
- [ ] **Step 6: Commit the complete feature** with only the module, tests, and these two design/plan files.
