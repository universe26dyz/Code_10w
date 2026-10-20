# Native-plane evaluation coordinate contract

## Proposed / Q002 / Q004

The native pixel lattice is converted through the historical local grid and
`space.local_to_train` into the checkpoint's final learned rigid pose:

`native pixels -> local grid -> local_to_train -> final learned pose -> INR(world_train)`.

Central prediction evaluates that coordinate directly. PSF128 converts the
same centre with `train_ras_to_physical`, applies the frozen physical-RAS
Gaussian offsets, maps them back through `physical_ras_to_train`, then queries
and averages T1/T2 in map domain. A physical-RAS NIfTI audit must first convert
`world_train` with `train_ras_to_physical`; training-space coordinates must
never be passed to a physical NIfTI affine.

## Fit-first

The R0/native affine defines reference pixel ordering, but the independent
T1/T2 NeSVoR reconstructions use their parameter-specific final
`--output-slices` pose. The immutable endpoint is:

`original [row,col] -> proven NeSVoR I/O reindex/flip -> final T1 or T2 pose -> physical RAS -> volume_ms -> original [row,col]`.

`model.pt` is checkpoint provenance, not a per-slice pose store. Output-slice
values validate I/O correspondence; predictions are sampled from `volume_ms`.
Historical SCMR `registered_ms`/`synthetic_ms` likewise use parameter-specific
registered final geometry.

> 配准与 pose 优化会影响重建本身；若在最终评估端点改用 R0 affine，便会造成重建 pose 与评估 plane 不一致。

Before a Fit-first final-pose bundle can be built, audit v3 must pass
`RECONSTRUCTION_INPUT_GEOMETRY_VALID`: R0 geometry and exact NeSVoR
`load_stack`/`affine2transformation` initial geometry agree after recorded
reindexing, spacing, corner/centre and normalized-value checks. The physical
equivalence threshold is `1e-3 mm`, solely for the R0-to-initial exact NeSVoR
representation/reindex round trip; it accommodates NIfTI/float32 round-trip
error and does not relax strict value correspondence. It is never used to ask
whether `initial_vs_final` learned pose is zero: that motion remains computed
directly from NeSVoR internal rigid transforms. A raw affine difference is
only an axis-convention diagnostic, not evidence of motion. A real mismatch
records `RECONSTRUCTION_INPUT_GEOMETRY_MISMATCH`, fails closed, and does not
alter reconstruction, training, or formal metrics.
