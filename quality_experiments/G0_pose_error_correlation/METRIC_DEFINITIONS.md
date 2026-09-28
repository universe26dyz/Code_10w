# Metric definitions

Each map record is retained only if its official correspondence is the same group: `content_matched_group == expected_group` and `best_orientation == IDENTITY`. The error endpoints are the central/no-PSF `relative_rmse` and `one_minus_correlation = 1 - correlation`; no PSF or reconstruction output is created.

Pose metrics are copied from `G0_pose_drift.csv`: `translation_magnitude_mm` and `rotation_magnitude_deg`. Their provenance is the pose row matched by `stack + expected_group == group_idx`; `global_group_idx` is emitted for traceability only and is never a join key.

For each of SAX, 2CH, and 4CH and each of T1 and T2, the four two-sided associations are translation/rotation against relative RMSE and `1-correlation`. Both Pearson `r` and Spearman `rho` plus p-values are reported; Spearman is the primary rank-based statistic. Benjamini-Hochberg q-values adjust all 24 primary Spearman p-values together.

Leave-one-group-out sensitivity recomputes Spearman `rho` after removing each local group once. It reports the full rho, the LOO minimum, maximum, median, and the local and global identifiers whose removal produces the largest absolute change from full rho.
