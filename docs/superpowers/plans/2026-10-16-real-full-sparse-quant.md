# Real Full and Sparse Quantitative Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real-only Phase-4R execution path for R0 full-data agreement and R1 sparse long-axis held-out quantitative evaluation.

**Architecture:** Keep the existing combined synthetic/real pipeline unchanged.  Add sibling real-only preflight and server orchestration, derive immutable sparse inputs from prepared observations, and evaluate native held-out maps with one method-neutral support rule.  Aggregate only at fold, subject, then cohort level.

**Tech Stack:** Python, NumPy, existing HB1 audit, existing dictionary matcher, shell server runners.

**Spec:** `/home/universe/Downloads/Codex_05_PHASE4R_v2_REAL_FULL_AND_SPARSE_QUANT.md`

## Global Constraints

- Retain only CYJ, DYZ, HHZ, HJL at TR=2.61 ms, VPS=87, with atomic weights 0..9.
- Do not run GPU/NeSVoR/XCAT jobs locally or modify prepared observations.
- Reuse Q002S640_B6_roi_same_anchor_mse, current dictionary fitting, and the actual HB1 audit.
- Exclude held-out long-axis observations and every derived map/mask from reconstruction.
- Do not add signal-domain metrics; record them as deferred by design.

## Review Focus

- Sparse groups have complete 10-weight fingerprints and are reindexed contiguously.
- Retained and held-out long-axis identities are disjoint and exhaustive.
- Evaluation support is derived before either method's finite prediction filter.
- L100 references R0's unmodified prepared source rather than a duplicate reconstruction.
- Aggregation treats subjects, not pixels or planes, as cohort samples.

### Task 1: Real-only contracts and R0 evaluation

**Files:** `phase2_clean/preflight_real.py`, `phase2_clean/evaluation.py`, `phase2_clean/aggregate_real.py`, tests.

- [ ] Add failing tests for preflight isolation, native-plane metrics, means, Bland–Altman, and subject-level aggregation.
- [ ] Implement the sibling real-only verifier and R0 metric/aggregation helpers.
- [ ] Run the focused tests.

### Task 2: Sparse source construction and held-out evaluation

**Files:** `phase2_clean/sparse.py`, `phase2_clean/evaluation.py`, tests.

- [ ] Add failing tests for exact phase selection, atomic fingerprints, group mapping, no leakage, gap audit, and method-neutral support.
- [ ] Implement immutable sparse builder plus held-out dictionary/reprojection evaluator.
- [ ] Run the focused tests.

### Task 3: Server-only orchestration

**Files:** `phase2_clean/server_commands/*real*`, `phase2_clean/runner.py`, runbook, tests.

- [ ] Add failing tests for real-only preflight commands and required sparse pilot arguments.
- [ ] Implement non-overwriting real R0/R1 scripts that only dispatch on the server.
- [ ] Run focused tests and one complete local test pass.
