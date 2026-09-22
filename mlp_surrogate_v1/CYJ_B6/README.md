# CYJ_B6 FrozenMLP controlled reconstruction

This configuration inherits the existing Trad B6 reconstruction settings.  The
only intended variable is `decoder`: baseline uses `bloch_decoder_factory`; the
MLP route uses `frozen_mlp_decoder_factory` through the same
`reconstruction_core.train_reconstruction` loop.

Before a formal run, set `decoder.checkpoint` to a checkpoint accepted by
`load_frozen_mlp_decoder`.  That loader requires a matching protocol and timing
domain plus `validation_status: approved_by_manual_review` for a non-fixture
formal candidate.  The template is blocked and intentionally has no checkpoint
path, so it cannot start a formal reconstruction by itself.
