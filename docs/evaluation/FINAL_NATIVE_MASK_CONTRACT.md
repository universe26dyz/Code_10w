# Final native support contract

The final native-plane paper endpoint reads only an immutable
`phase4r_current_native_central_component_mask/v1` artifact. Its base and
primary supports are:

`base_native_support = source_support AND phase4r_l100`

`primary_native_mask = 8-connected central/largest component(base_native_support)`.

The selection rule is SCMR-style only: select the image-centre component when
its area is at least 25% of the largest component; otherwise select the
largest. No morphology, interpolation, resize, historical geometry, or other
reconstruction support participates. Historical SCMR reconstruction masks are
not formal inputs.

Paired Proposed versus Fit-first metrics use:

`primary_native_mask AND finite(reference) AND proposed_endpoint_valid AND fit_first_endpoint_valid AND finite(proposed_prediction) AND finite(fit_first_prediction)`.

Native reference panels use `primary_native_mask AND finite(reference)`; each
method/error panel additionally requires that method's endpoint-valid and
finite prediction.

Build the immutable formal artifact from current inputs only:

```bash
paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/build_current_native_central_component_masks.sh \
  R0_NATIVE_MAPS_ROOT L100_SPARSE_MANIFEST Q004_FROZEN_NATIVE_AUDIT \
  NEW_NATIVE_MASK_ROOT PYTHON
```

Longitudinal figures are a separate endpoint: source SAX is an acquisition
reference, while Fit-first/Q002/Q004 panels sample each reconstructed 3-D
volume on the same dense physical plane. Their support remains `in_volume AND
finite`; native-plane masks and native bundle stacking are not used there.
