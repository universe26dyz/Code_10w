# Q004 endpoint and SCMR-mask evaluation design

## Scope

Add read-only evaluation and visualization consumers for the canonical CYJ
L100_fold0 Q004 checkpoint.  No training, checkpoint mutation, formal-output
overwrite, or server execution is part of this change.

## Direct-native endpoint audit

`direct_native_endpoint_audit.py` will compare one central-Q004 checkpoint by
two inference endpoints on identical current native planes:

- `direct_native`: construct the historical Q004 model from the checkpoint and
  authoritative L100 prepared observations, then query the INR at the exact
  native world coordinates.  This path does not open the central 3-D volume.
- `volume_reprojected`: open the explicit central T1/T2 1-mm NIfTI volumes and
  linearly sample the same native world coordinates.

Both predictions use the same R0 native reference, source support, nearest
reprojected frozen-L100 mask, finite eligibility, affine, group identity, and
metric implementation.  New immutable output contains native predictions,
per-plane and voxel-pooled metrics, endpoint-to-endpoint differences, figures,
and input SHA provenance.

## SAX longitudinal renderer

The selected SAX centre row/column defines one physical plane and a common
physical extent.  The source-stack longitudinal panel retains source slice
centres, nearest interpolation, and source support.  Each reconstruction is
sampled directly from its 3-D volume onto a separate dense 1-mm plane using
linear interpolation and only `in_volume AND finite` support.  No native-stack
support masks reconstructed panels by default.

## Current-native mask generation

`current_native_central_component_mask.py` builds the formal primary support
from `source_support AND phase4r_l100`, both already on the current native
lattice. It then applies the SCMR-style 8-connected central/largest-component
selection rule only. Historical training, synthetic, fitting-valid, registered,
and learned-pose geometry are not inputs.

`mask_sensitivity_audit.py` compares `primary_native_mask` and
`phase4r_l100` while holding reference and prediction fixed.

## Verification

Tests cover direct-versus-volume source separation, identical support,
determinism, immutable output, dense 1-mm reconstruction sampling,
current-native component selection, L100 support, and support sensitivity. Run
targeted pytest, compileall, and `git diff --check` locally.
