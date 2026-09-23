# MLP surrogate replacement inspection

Inspection date: 2026-09-23
Repository inspected: `/home/universe/SVR/multimap_postprogramming/Code_10w`  
Baseline commit inspected before the controlled-route integration:
`9700bc82c7e8be49e833b405ab63aac181ca97cd`

The requested server code path `/data/dengyz/code/Code_10w` is absent in the
current environment.  This report therefore records the available working
clone and does not claim that it inspected a different server checkout.

## Reconstruction entry points

- Trad: `trad/scripts/reconstruct_subject.py` resolves `sax`, `2ch`, and
  `4ch` prepared observations, then calls `trad/scripts/run_training.py`.
- Trad and MLP facades both call the single
  `reconstruction_core/orchestration.py:train_reconstruction` engine.
- Trad selects `bloch_decoder_factory`; MLP selects
  `frozen_mlp_decoder_factory` after strict checkpoint validation.

The current MLP route reuses the same quantitative INR, group-rigid PSF forward
(`TradQuantitativeForward`), staged optimizer, exports, and native reprojection
code.  Its sole model difference is the frozen decoder factory.

## Bloch decoder call chain

`bloch_decoder_factory` creates `TradSignalSimulator`, injects it into the
shared `ReconstructionTrainingModel`, and calls it from
`rigid_psf_forward.py` after local PSF sampling, final group-rigid RAS
transformation, and INR evaluation.  The simulator implementation is
`trad/modules/module_05_signal_decoder/trad_signal_simulator.py`; it returns a
10-heartbeat fingerprint.  `TradQuantitativeForward` selects the observed
weight `0..9`, applies amplitude, and averages PSF samples.

## MLP interface and replacement point

`frozen_mlp_decoder_factory` loads `FrozenMLPSignalDecoder` through
`mlp/modules/module_05_signal_decoder/checkpoint_loader.py` and supplies it to
the same `TradQuantitativeForward` constructor.  The replacement boundary is
therefore the `signal_simulator` argument only.

`FrozenMLPSignalDecoder.forward(t1_ms, t2_ms, b1, timing9_ms, protocol,
normalize=True)` accepts matching `[...]` tensors and `timing9_ms [...,9]`,
then returns `[...,10]`.  It keeps all MLP parameters frozen and forces
BatchNorm evaluation mode without a `no_grad` block, preserving gradients to
the INR tissue inputs.

`make_input12` builds the exact MLP input `[...,12]` as
`[T1/1000, T2/1000, B1, timing9/1000]`.  `MdmSignalMLP` is
12→200→200→200→10 with L2-normalized output.  The checkpoint loader requires
the matching input/output normalization metadata, HHZ protocol, tissue ranges,
and observed timing range before online use.

## Normalization

Both decoders supply an L2-normalized 10-weight fingerprint to the shared
forward.  The shared forward multiplies by INR amplitude after selecting the
observed weight and averages spatial PSF samples.  Subject intensity scaling
is retained in the reconstruction model/export provenance; it is not a decoder
specific normalization.

## Timing instrumentation

The shared engine uses `IterationProfiler` and writes the same CSV fields for
both routes: `iteration,stage,section,milliseconds`.  Iteration-level sections
include `forward_decoder`, `loss`, `backward`, and `optimizer_step`, alongside
the retained detailed sections.  A `run` stage records `checkpoint_load`,
`data_loading`, `initialization`, `optimization_total`, `validation`, `export`,
and `total_runtime`.  Summary generation writes
`timing_profile_summary.json` with `decoder_type: Bloch|FrozenMLP`.

## CYJ_B6 checkpoint

The located baseline checkpoint is:

`/home/universe/SVR/data/Code_10w_v1/Code_10w_runs/trad_v3/CYJ_B6/model.pt`

No reconstruction was started and no historical output directory was modified.
