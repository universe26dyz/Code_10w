# Q002 B6 ROI Same-Anchor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a server-ready Q002 B6-cropped same-anchor ten-weight MSE route without changing B6 scalar reconstruction semantics or running a formal reconstruction.

**Architecture:** Q002 lives in an independent experiment package.  It groups original B6 cropped scalar observations by integer native pixel identity, reuses B6's scalar dataset to derive normalization and stack weights, and uses an additive shared-K8 fingerprint forward path.  A narrow Q002 loop reuses the existing model construction, optimizer, scheduler, regularization, checkpoint, export, and provenance machinery while replacing only scalar sampling/loss.

**Tech Stack:** Python 3, PyTorch, NumPy, pytest, existing Trad/MLP reconstruction stack, Bash.

**Spec:** `docs/superpowers/specs/2026-10-08-q002-b6-roi-same-anchor-design.md`

## Global Constraints

- B6 cropped inputs only: `CYJ/{sax,2ch,4ch}/observations.npz`; reject Q001 full-FOV paths.
- Load B6 `resolved_config` only; forbid B6/Q001 checkpoint or pose warm-start.
- Preserve B6 seed, INR/FrozenMLP, optimizer/scheduler, regularization, bbox, normalization, stack weighting, Stage A/B 2000/4000, and K=8.
- `training.batch_size=640` denotes 64 anchors × 10 weights; the sampler must never produce 640 anchors.
- Q002 objective is only the weighted 10-vector MSE.  No cosine loss or Q003 objective is implemented.
- Existing scalar `QuantPointDataset`, `CachedBalancedSampler`, scalar forward, logs, and artifacts retain their current behavior.
- Formal scripts must require CUDA, clean Git, matching checksums, and absent/empty output/log/evaluation targets; they must not delete paths.
- Do not stage or modify existing unrelated untracked `git reset --hard HEAD@{0}` / `git reset --hard HEAD@{1}` files.

## Review Focus

- Missing or duplicate weight membership must fail before a tensor is constructed.
- Masks with disjoint valid pixels must produce no hidden zero-filled anchor vectors and fail for an empty group.
- Shared fingerprint forward must invoke PSF sampling once, not once per weight, and amplitude must multiply before PSF averaging.
- Joint support attrition must be recorded rather than silently changing B6 stack weights or normalization.
- B6/Q002 comparisons must reject shape/provenance mismatch and pool only strict finite common support.

### Task 1: Q002 contracts and joint anchor dataset

**Files:**
- Create: `quality_experiments/Q002_B6_roi_same_anchor_mse/{__init__.py,contracts.py,joint_dataset.py}`
- Create: `quality_experiments/Q002_B6_roi_same_anchor_mse/tests/test_joint_dataset.py`

**Interfaces:**
- Produces `Q002_EXPERIMENT_ID`, `cropped_observation_paths(root)`, `build_q002_route_config(...)`, and `JointAnchorDataset`.
- `JointAnchorDataset(paths, scalar_dataset)` exposes anchor `xyz`, `observed`, `group_idx`, `stack_idx`, `timing`, `row_col`, QC, and fixed-monitor construction.

- [ ] Write failing tests for exact 0..9 membership/order, integer identity, timing/geometry mismatch rejection, ten-way AND support, empty-group rejection, source/full-FOV exclusion, support attrition, and seed/hash-stable monitor identities.
- [ ] Run the focused tests and verify they fail because the package interfaces do not exist.
- [ ] Implement contract and dataset validation from NPZ metadata, while deriving B6 scalar stack weights and normalization from the unmodified scalar dataset.
- [ ] Run focused tests and verify they pass.

### Task 2: Shared-K8 fingerprint forward

**Files:**
- Modify: `trad/modules/module_06_rigid_psf/rigid_psf_forward.py`
- Modify: `reconstruction_core/orchestration.py`
- Create: `quality_experiments/Q002_B6_roi_same_anchor_mse/tests/test_fingerprint_forward.py`

**Interfaces:**
- Produces `TradQuantitativeForward.forward_fingerprint(local_xyz_mm, group_idx, timing9_ms, n_psf_samples, profile=None) -> Tensor[B,10]`.
- Produces `ReconstructionTrainingModel.forward_fingerprint(batch, n_psf_samples, profile=None) -> Tensor[B,10]`.

- [ ] Write failing tests proving one PSF call per anchor batch, `[B,10]` output, manual K=1/K=8 amplitude semantics, and finite gradients.
- [ ] Run the focused tests and verify the new method is absent.
- [ ] Add only the additive fingerprint methods; leave scalar `forward` untouched.
- [ ] Run focused tests and relevant scalar forward regression tests.

### Task 3: Anchor sampler, vector objective, and minimal trainer

**Files:**
- Create: `quality_experiments/Q002_B6_roi_same_anchor_mse/{training.py,run_q002_reconstruction.py}`
- Create: `quality_experiments/Q002_B6_roi_same_anchor_mse/tests/test_training.py`

**Interfaces:**
- Produces `JointAnchorSampler.sample(anchor_batch_size)`, `joint_vector_mse(predicted, observed, stack_idx, stack_weights)`, and `train_q002_reconstruction(...)`.

- [ ] Write failing tests for 64×10 batches, with-replacement stack sampling, stack-weight formula, manual MSE, copied B6 Stage A/B/optimizer/scheduler control, scalar normalization equality, and no scalar sampler behavior changes.
- [ ] Run the focused tests and verify the Q002 trainer interfaces are absent.
- [ ] Implement the dedicated vector loop by importing existing orchestration helpers for model build, regularization, optimizer, checkpoint, profile, Git provenance, and normalization; document each reuse in the manifest.  Do not clone the scalar loop beyond the necessary vector batching/loss boundary.
- [ ] Run focused tests and existing scalar sampler/trainer regressions.

### Task 4: Q002 exports, common-support evaluator, and provenance

**Files:**
- Create: `quality_experiments/Q002_B6_roi_same_anchor_mse/{evaluate_q002.py,evaluation.py,metrics.py}`
- Create: `quality_experiments/Q002_B6_roi_same_anchor_mse/tests/test_evaluation.py`

**Interfaces:**
- Produces `strict_q002_common_support(...)`, true-pooled global/per-stack/per-group rows, route/evaluation manifests, and a read-only evaluator CLI.

- [ ] Write failing tests for strict map/signal common support, true pooling versus macro behavior, fingerprint cosine diagnostic-only output, K128 evaluation plan, checkpoint/Git SHA provenance, and B6/Q001 read-only behavior.
- [ ] Run focused tests and verify the evaluator interface is absent.
- [ ] Implement exports through existing `export_quantitative_outputs_psf128` and `export_native_plane_reprojections`; implement evaluation without constructing/training a model.
- [ ] Run focused tests and Q001 metrics regressions.

### Task 5: Server commands, records, and local checks

**Files:**
- Create: `quality_experiments/server_commands/{run_Q002_CYJ.sh,evaluate_Q002_CYJ.sh}`
- Modify: `quality_experiments/EXPERIMENT_LOG.md`
- Modify: `quality_experiments/codex_phase_state.yaml`
- Create: `quality_experiments/Q002_B6_roi_same_anchor_mse/tests/test_server_commands.py`

- [ ] Write failing tests that inspect Q002-only scripts for CUDA/conda/checksum/clean-Git/empty-root/log/evaluation guards and ban full-FOV/Q001 runner delegation/destructive deletion.
- [ ] Run focused tests and verify scripts are absent.
- [ ] Implement scripts and append-only experiment records marking Q002 code ready only after successful local checks; keep Q001A external result/Q001B completion and Q003 NOT_STARTED intact.
- [ ] Run `pytest` targeted suites and relevant D1/D2/G0/Q001/B6 regressions once, plus `compileall`, CLIs `--help`, `bash -n`, YAML parse, and `git diff --check`.
- [ ] Stage only the plan/spec and Q002-related files, commit with `Add B6-ROI same-anchor fingerprint MSE reconstruction`, and record the commit SHA without pushing.
