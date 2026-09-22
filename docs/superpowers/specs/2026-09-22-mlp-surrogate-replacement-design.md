# MLP surrogate replacement: Part 3/4 design

## Goal and boundary

This change prepares a controlled comparison of the direct HHZ Bloch decoder
against a frozen, validated MLP decoder.  The signal decoder is the sole
permitted experimental variable.  It does not start a formal reconstruction,
modify checkpoint approval metadata, alter historical result directories, or
create a substitute reference/mask.

The controlled MLP result directory will be selected only at invocation time:
`/data/dengyz/dataset/Code_10w_v1/Code_10w_runs/mlp_surrogate_v1/CYJ_B6`.
The repository configuration must not hard-code an MLP checkpoint path.

## Shared reconstruction orchestration

The existing Trad route becomes the single owner of reconstruction orchestration:

- prepared observation loading and group ordering;
- global normalization and intensity scaling;
- HB1 stack initialization;
- INR and rigid/PSF construction;
- variance strategy, data/regularization/transformation loss, optimizer, and
  scheduler;
- staged iteration loop and exports.

It accepts a small decoder factory interface.  The Trad factory produces
`TradSignalSimulator`; the MLP factory uses the existing strict
`load_frozen_mlp_decoder` gate and returns `FrozenMLPSignalDecoder`.  Both
implement the same decoder call contract:

`(t1_ms [...], t2_ms [...], b1 [...], timing9_ms [...,9], protocol,
normalize=True) -> fingerprint [...,10]`.

The shared `TradQuantitativeForward` remains the only caller: local PSF sample
→ final rigid RAS pose → INR → decoder → selected weight/amplitude → sample
mean.  No MLP parameter belongs to the optimizer; it remains frozen and in
BatchNorm evaluation mode, while gradients flow through its operations to the
reconstruction parameters.

## Controlled configuration and checkpoint gate

A new MLP-surrogate B6 configuration derives the complete controlled settings
from Trad B6 and contains no checkpoint pathname.  The command receives the
checkpoint explicitly.  The existing loader remains authoritative: a
functional fixture, unvalidated formal candidate, timing-domain mismatch, or
protocol mismatch aborts before optimization.  The current located candidate
checkpoints are unvalidated, so implementation and tests may complete but no
formal MLP reconstruction may run.

The integration guide will state that a validated checkpoint must first pass
the existing formal validation/manual approval workflow, then be supplied to
the MLP command.  It will not suggest editing `validation_status` by hand.

## Timing profile extension

Existing `IterationProfiler` CSV fields remain unchanged:

`iteration, stage, section, milliseconds`.

Both methods will append run-level samples in the same file using a dedicated
run stage and these section names: `checkpoint_load`, `data_loading`,
`initialization`, `optimization_total`, `validation`, `export`, and
`total_runtime`.  Absent operations are recorded as explicit zero-duration
not-applicable samples only where necessary to preserve the same section set.
Iteration-level section names are normalized to include `forward_decoder`,
`loss`, `backward`, and `optimizer_step`; finer existing sections remain.

The existing summary function is extended, not replaced, to write the same
`timing_profile_summary.json` schema for Trad and MLP.  It reports per-stage,
per-section statistics and total runtime without treating run-level samples as
optimization iteration estimates.

## Tests and acceptance

Tests will compare the resolved Bloch and MLP controlled configurations and
prove equality for normalization, PSF, loss, and optimizer parameter groups.
They will verify both decoders return ten signals; MLP parameters are frozen;
and gradients reach only reconstruction parameters.  Profiling tests will
verify the common CSV columns, required run/iteration section names, and
summary JSON for both routes.  No test uses a checkpoint that bypasses the
existing approval gate.

Acceptance is source-level parity plus targeted test evidence.  A formal run
is explicitly blocked until an approved checkpoint is available.
