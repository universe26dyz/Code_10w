# Native-plane coordinate contract

## Proposed Q002/Q004

The native pixel lattice is converted to the historical centred local grid,
then `space.local_to_train`, then the checkpoint's final learned rigid pose.
The resulting `world_train` coordinates are the only inputs to `model.inr`.
The original R0/native affine must not replace that learned pose.

Central prediction is `model.inr(world_train)`.  PSF128 first converts the
same centre with `space.train_ras_to_physical`, adds the frozen formal K=128
physical-RAS Gaussian offsets, converts samples with
`space.physical_ras_to_train`, evaluates the INR, and averages T1/T2 in map
domain.  A physical-RAS NIfTI audit sample must convert `world_train` with
`space.train_ras_to_physical` before applying its affine.

## Fit-first

Fit-first runs NeSVoR with stack registration and transformation optimization.
For each parameter and original group, its final native-plane geometry is the
matching `nesvor/<parameter>/slices/<output-index>.nii.gz` affine emitted by
`--output-slices`.  `model.pt` is checkpoint provenance, not a per-slice pose
table.  R0 geometry may be used only when the immutable pose audit proves the
two complete voxel geometries mathematically identical.

Historical SCMR `registered_ms` and `synthetic_ms` are likewise represented on
parameter-specific final registered slice geometry.  Fit-first and SCMR must
not silently substitute the original native affine for those final poses.
