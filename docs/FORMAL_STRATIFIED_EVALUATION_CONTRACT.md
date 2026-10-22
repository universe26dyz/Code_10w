# Formal stratified native evaluation contract

`final_stratified_native_metrics.py` is a report-only evaluator for a named, eligible L100/L50/L25 sparse manifest or an explicit `phase4r_sparse_profile_v2` case. It samples both immutable endpoint exports on every original R0 native source plane, then applies the current-native `primary_native_mask`.

The paired support for every parameter and plane is:

`primary_native_mask AND finite(reference) AND proposed_endpoint_valid AND fit_first_endpoint_valid AND finite(proposed) AND finite(fit_first)`.

It reports `sax_observed`, `lax_observed`, `lax_heldout`, `nonheldout_all`, `all_lax`, and `all_source`. For L50/L25, retained and held-out LAX identities must exactly partition each R0 cache; SAX is always observed. For L100 every source group is observed and `lax_heldout` is `NOT_APPLICABLE_L100` rather than an invented empty comparison.

This output is separate from `final_pose_native_metrics.py` (L100 final-pose endpoint) and `final_sparse_native_metrics.py` (L50/L25 held-out-only original-native endpoint). Results from these families must retain their distinct provenance and are never silently pooled.

## Explicit custom-profile reporting

For `phase4r_sparse_profile_v2`, `profile_id` is the profile design and is not a legacy condition level. `case_id` is the formal case identity; a custom fold is an explicit within-profile case, not a subject. The stratified metric manifest carries `profile_id`, `case_id`, `fold_id`, `fold_count`, and `sparse_schema_version`, with `condition == case_id`.

The v4 subject summary retains each explicit case before producing a separate profile-level summary. Profile-level rows record `case_ids`, `case_count`, `*_mean_across_cases`, and `*_sample_sd_across_cases`. Legacy L100/L50/L25 sparsity robustness is not defined for custom profiles and is never emitted for them.

Older `final_stratified_native_bundle_metrics/v1` artifacts remain readable when their own identity fields are absent only by reading the recorded sparse manifest and verifying its SHA256, subject, custom schema, and `condition == case_id`. Directory names are never used to infer an identity.

Figure configs accept `Fit-first` plus one or more allowlisted Proposed methods. The deterministic reporting order is `Fit-first`, `Q002S640`, `Q002S640_K256`, `Q005S640_K4`, then `Q004S640`; only configured methods are rendered. `Q005S640_K4` names a frozen training configuration, while `central` and `psf128` remain distinct endpoints. No reporting command authorizes K4 execution or reconstruction.
