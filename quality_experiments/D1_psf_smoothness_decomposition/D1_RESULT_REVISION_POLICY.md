# D1 result revision policy

`D1_psf_smoothness_decomposition_baseline_v1` is a failed precompute/input-schema attempt, not a formal D1 result. Its loader incorrectly required one central-plane archive row per spatial group; the historical export has ten weight rows per group. Do not delete or overwrite that directory.

`D1_psf_smoothness_decomposition_baseline_v2` is the corrected formal rerun target. It retains the scientific result schema `code10w_d1_psf_smoothness_decomposition/v1` and records the corrected implementation explicitly as `implementation_revision: D1_psf_smoothness_decomposition_baseline_v2`, along with the exact Git commit.
