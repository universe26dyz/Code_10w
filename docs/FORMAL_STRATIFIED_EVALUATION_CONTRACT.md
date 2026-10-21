# Formal stratified native evaluation contract

`final_stratified_native_metrics.py` is a report-only evaluator for a named, eligible L100/L50/L25 sparse manifest. It samples both immutable endpoint exports on every original R0 native source plane, then applies the current-native `primary_native_mask`.

The paired support for every parameter and plane is:

`primary_native_mask AND finite(reference) AND proposed_endpoint_valid AND fit_first_endpoint_valid AND finite(proposed) AND finite(fit_first)`.

It reports `sax_observed`, `lax_observed`, `lax_heldout`, `nonheldout_all`, `all_lax`, and `all_source`. For L50/L25, retained and held-out LAX identities must exactly partition each R0 cache; SAX is always observed. For L100 every source group is observed and `lax_heldout` is `NOT_APPLICABLE_L100` rather than an invented empty comparison.

This output is separate from `final_pose_native_metrics.py` (L100 final-pose endpoint) and `final_sparse_native_metrics.py` (L50/L25 held-out-only original-native endpoint). Results from these families must retain their distinct provenance and are never silently pooled.
