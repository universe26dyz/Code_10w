# Q002 B6 ROI Same-Anchor Design

## Goal

Implement a server-ready, non-warm-start Q002 reconstruction route that tests
`64 native anchors x 10 fingerprint weights` against the original B6 cropped
cardiac observations, while holding B6 physics, optimization, and evaluation
controls fixed.  The formal reconstruction and evaluation remain server-only.

## Scope and controls

Q002 reads only `CYJ/{sax,2ch,4ch}/observations.npz` under the B6 cropped
prepared root.  It must not read Q001 full-FOV prepared data or Q001 poses.
It loads B6 `model.pt` only to copy `resolved_config`; B6 model state,
optimizer state, and trained poses are rejected as warm starts.  The approved
FrozenMLP checkpoint must equal `resolved_config.decoder.checkpoint`.

The copied configuration remains the source of INR, decoder, optimizer,
scheduler, regularization, bbox, normalization, seed `20260911`, Stage A/B
iterations `2000/4000`, and K=8.  The only training difference is grouping
the B6 scalar observations into a joint anchor batch: 64 anchors per step,
each with all ten weights, for an effective scalar budget of 640.  Q002 does
not add cosine loss, map/reference supervision, VARPRO, segmentation, or
regularization changes.

## Components

`quality_experiments/Q002_B6_roi_same_anchor_mse/` is the isolated Q002
module.  Its joint dataset builds identity from native integer
`(global_group_idx, row, col)`, verifies each group contains exactly one
weight `0..9`, orders weights explicitly, and forms support exclusively as
the ten-way foreground-mask intersection plus finite observations.  It emits
per-stack/per-group mask attrition, scalar-candidate counts, geometry/timing
checks, source hashes, and a seed-stable monitor anchor list/hash.  Empty
joint groups fail.

The module has a sampler over anchors, not scalar pixels.  Its stack sampling
weights derive from `QuantPointDataset.validate_balanced_samples()` on the
original scalar cropped dataset.  The sampler records the joint-support stack
distribution separately instead of renormalizing B6 weights.

The existing scalar path is preserved.  `TradQuantitativeForward` gains an
additive fingerprint method: sample a group-local K=8 neighbourhood once,
evaluate the INR and FrozenMLP once, multiply per-sample amplitude, then mean
the `[B,K,10]` fingerprint to `[B,10]`.  Scalar `forward` retains its present
sampling, gather, state-dict, and numeric behaviour.  The training model
exposes the same additive API.

Q002 training reuses orchestration helpers for B6 configuration validation,
model construction, intensity normalization (including exact fallback), pose
initialization, optimizer/scheduler, regularization, checkpoint format,
profiling, and Git provenance.  A small dedicated loop is necessary because
the established scalar loop assumes `weight_idx` and balances every scalar
weight; the Q002 loop instead consumes anchor vectors and computes
`mean_a(stack_weight[stack[a]] * mean_w(error[a,w]^2))`.  It applies the same
two optimizer stages and regularizers.  Its output includes a Q002-specific
manifest and result summary without claiming a formal result.

Server commands are explicit Q002-only wrappers with CUDA/input/checksum,
empty output/log/evaluation, and clean-Git guards.  They never invoke Q001
commands and never remove paths.  The evaluator is read-only: it compares
central native-plane T1/T2 and K8 signal exports against FrozenMLP B6 on the
strict finite common support, writes true-pooled global/per-stack/per-group
metrics and read-only cosine diagnostics, and reports Q001 artifacts only as
non-comparable historical references.  PSF128 volume export reuses the
existing 1 mm, factor 1, K128, seed `20260911` export implementation.

## Data flow

`cropped observations -> scalar QuantPointDataset (normalization and B6 stack
weights) + JointAnchorDataset (membership/QC) -> 64-anchor sampler -> one K8
forward_fingerprint -> joint MSE + unchanged B6 regularization -> checkpoint
and exports -> strict B6/Q002 common-support evaluator`.

Q003 can reuse the joint dataset, sampler, fingerprint forward, and vector
MSE.  It is deliberately absent from Q002; a future Q003 objective may add
only a normalized fingerprint cosine term.

## Failure handling and verification

Malformed membership, mismatched group metadata/timing, non-identical native
geometry, absent joint support, non-finite values, mismatch of approved
decoder, nonempty outputs, or a dirty server worktree are hard failures.
Local tests use synthetic NPZ fixtures/mocks and do not fabricate server
artifacts or execute formal GPU reconstruction.  Coverage includes grouping,
support, sampling, normalization equivalence, shared-K8/amplitude gradients,
joint loss, scalar regressions, B6 controls, export/provenance, evaluator
common support, and scripts/CLI/static checks.
