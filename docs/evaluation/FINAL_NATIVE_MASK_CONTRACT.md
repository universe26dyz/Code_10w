# Final native support contract

The final native-plane paper endpoint uses one immutable primary support:

`source_support AND scmr_style_central_component AND endpoint-valid/in-volume AND finite(reference) AND finite(prediction)`.

For Proposed versus Fit-first metrics, the common support additionally requires
both methods' valid finite predictions. `scmr_style_central_component` is read
only from a `scmr_central_component_mask/v2` artifact and retains the historical
six-input strict-common construction before 8-connected central/largest
component selection. It is never reconstructed from displayed images or model
predictions.

`phase4r_l100` and `phase4r_l100_central_component` remain sensitivity/control
domains. The latter is not historical SCMR support; it only isolates the
component-selection effect from the six-input construction.

Longitudinal figures are a separate endpoint: source SAX is an acquisition
reference, while Fit-first/Q002/Q004 panels sample each reconstructed 3-D
volume on the same dense physical plane. Their support remains `in_volume AND
finite`; native-plane masks and native bundle stacking are not used there.
