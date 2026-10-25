# Subject-specific endpoint validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Permit explicit full-input validation for DYZ, DYL_2, HHZ, and HJL while preserving CYJ's historical input-layout contract.

**Architecture:** The existing planner remains the only command builder.  CYJ continues to require its audited 14/15/12 layout; every other explicitly named subject is admitted with its observed complete 10-weight groups recorded in provenance.  Cross-subject endpoint validation omits metric generation by explicit `--skip-metric`, so Final5 v1/v2 mask schemas are never silently mixed.

**Tech Stack:** Python, argparse, pytest, Bash.

**Spec:** User request of 2026-10-26 in this thread.

## Global Constraints

- No reconstruction/metric scientific setting changes.
- All roots are caller supplied; no path discovery.
- CYJ historical 14/15/12 admission remains strict.
- Cross-subject first pass produces Proposed, Fit-first, and central endpoints only.
- No server execution from local Codex; commit locally only.

## Review Focus

- CYJ standardized 12/15/10 input must still fail; tested in Task 1.
- A non-CYJ complete 12/16/10 input must be admitted and recorded; tested in Task 1.
- Skipping metric must not require a native-mask path or emit metric argv; tested in Task 2.
- Metric mode must retain the existing native-mask validation and metric command; existing CYJ tests cover this.
- Wrapper must pass `--skip-metric` only when explicitly requested; tested in Task 2.

### Task 1: Subject-aware planner admission

**Files:**
- Modify: `paper_experiments/joint_vs_2dfit/phase2_clean/cyj11_untrimmed_reproduction.py`
- Test: `paper_experiments/joint_vs_2dfit/tests/test_cyj11_untrimmed_reproduction.py`

- [x] Write a failing test for a `DYZ` 12/16/10 full input that produces a plan with the subject's own SHA/count records.
- [x] Run the test and confirm the current CYJ-only guard fails.
- [x] Preserve CYJ's fixed count rejection while admitting other explicit subjects with complete ten-weight groups.
- [x] Run the focused planner tests.

### Task 2: Explicit endpoint-only mode

**Files:**
- Modify: `paper_experiments/joint_vs_2dfit/phase2_clean/cyj11_untrimmed_reproduction.py`
- Modify: `paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_cyj11_untrimmed_reproduction.sh`
- Modify: `docs/CYJ_1_1_UNTRIMMED_REPRODUCTION_RUNBOOK.md`
- Test: `paper_experiments/joint_vs_2dfit/tests/test_cyj11_untrimmed_reproduction.py`

- [x] Write a failing test proving endpoint-only mode omits metric and does not need a native mask.
- [x] Implement `--skip-metric` and `CYJ11_SKIP_METRIC=1`; leave current CYJ default metric behavior unchanged.
- [x] Add a concise runbook section for explicit cross-subject endpoint validation.
- [x] Run targeted tests, compileall, shell syntax, and diff check.
- [ ] Commit the complete change locally without pushing.
