# G0 pose-error correlation completion report

- Status: `CODE_READY_SERVER_RUN_PENDING`.
- Implementation commit: `d491c96fa235d9269bc338ff484285c459be03fb` — `Add G0 pose-error correlation diagnostic`.
- Scope: independent, read-only downstream analysis of corrected G0 v2 outputs. No reconstruction, map, baseline source, historical result, correspondence, Hungarian assignment, D1/D2/D3, or `next_phase: D2` was changed.
- Provenance contract: accept only a CYJ `G0_geometry_correspondence_baseline_v2` bundle with matching manifest revision, `metrics/G0_per_slice.csv`, and `metrics/G0_pose_drift.csv`; require `stack + expected_group == group_idx`, diagonal content matches, and `IDENTITY` orientation.
- Statistics: six fixed stack/parameter pairs × translation/rotation against relative RMSE and `1-correlation`; report Pearson and Spearman two-sided tests, BH-FDR over all 24 Spearman p-values, and leave-one-group-out rank sensitivity.
- Validation: 11 targeted synthetic tests passed in explicit `knesvr_torch`; package compileall and direct CLI help passed. A scoped search under `/home/universe/SVR/multimap_postprogramming` ran the CLI and returned `FORMAL_LOCAL_ANALYSIS = NOT_RUN_MISSING_G0_V2_ARTIFACTS`.
- Formal CYJ analysis: NOT RUN / NOT AVAILABLE. The required local v2 bundle was not present; missing paths are `metrics/G0_per_slice.csv` and `metrics/G0_pose_drift.csv` beneath the verified v2 root.
- Reproduction after a verified bundle is copied locally:

  `conda run --no-capture-output -n knesvr_torch python quality_experiments/G0_pose_error_correlation/run_g0_pose_error_correlation.py --input-root /absolute/path/G0_geometry_correspondence_baseline_v2`

- Output root for an actual local run: `/home/universe/SVR/multimap_postprogramming/Code_10w_local_results/G0_pose_error_correlation_CYJ_v1/`.
- Git status at implementation commit: the unrelated untracked files `et --hard HEAD@{0}` and `git reset --hard HEAD@{1}` were deliberately not staged, changed, or deleted.
- PUSH PERFORMED: NO.
