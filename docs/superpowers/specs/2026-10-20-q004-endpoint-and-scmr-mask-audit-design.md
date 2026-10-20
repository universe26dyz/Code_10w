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

## SCMR-style mask generation

`scmr_central_component_mask.py` exposes two independent functions:

- `build_legacy_strict_common(...)`: reproduce the legacy per-plane strict
  intersection: training mask AND synthetic support AND fitting-valid mask AND
  finite original/registered/synthetic maps.
- `select_scmr_central_component(...)`: 8-connected 2-D components; select the
  image-centre component when its area is at least 25% of the largest component,
  otherwise select the largest.  It performs no morphology, hole filling,
  interpolation, or border removal.

The production SCMR-style path uses all six explicit inputs to build the strict
common support before selecting the component.  An independent
`phase4r_l100_central_component` control applies the same selection rule to an
explicit current L100 mask alone, so the selection rule can be tested without
legacy-input dependencies.  Both masks are generated on current native-plane
geometry; no legacy bitmap is reused.

`mask_sensitivity_audit.py` will compare named supplied supports
`scmr_style_central_component` and `phase4r_l100`, including intersection and
asymmetric differences, while holding reference and prediction fixed.

## Verification

Tests cover direct-versus-volume source separation, identical support,
determinism, immutable output, dense 1-mm reconstruction sampling, legacy
strict-mask logic, central-component selection, L100 control, and support
sensitivity.  Run targeted pytest, compileall, and `git diff --check` locally.
