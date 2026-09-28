# G0 pose drift vs central-map error correlation

`G0_POSE_ERROR_CORRELATION_CYJ_V1` is a read-only post-hoc analysis of the corrected `G0_geometry_correspondence_baseline_v2` artifact bundle. It does not rerun reconstruction, change central maps, alter Hungarian correspondence, modify D1/D2/D3, or overwrite any results.

The analysis is intentionally strict. It accepts only a CYJ bundle with `manifest.json` declaring `implementation_revision: G0_geometry_correspondence_baseline_v2`, and both `metrics/G0_per_slice.csv` and `metrics/G0_pose_drift.csv`. It joins `stack + expected_group == group_idx`, never `global_group_idx`, and rejects duplicate or missing rows, nonfinite values, non-diagonal content matches, and non-identity orientations. T1 and T2 must use the same pose values for each stack/local group.

Run a known copied local bundle with:

```bash
conda run --no-capture-output -n knesvr_torch python quality_experiments/G0_pose_error_correlation/run_g0_pose_error_correlation.py \
  --input-root /absolute/path/G0_geometry_correspondence_baseline_v2
```

Without `--input-root`, `--search-root` may name one or more local directories. The command reports `FORMAL_LOCAL_ANALYSIS = NOT_RUN_MISSING_G0_V2_ARTIFACTS` when none contains a complete corrected bundle. Formal outputs, when an input is present, are written only to `/home/universe/SVR/multimap_postprogramming/Code_10w_local_results/G0_pose_error_correlation_CYJ_v1/` by default.

The inference is exploratory, single-subject, and small-N. Association does not establish that a pose change is harmful or that it represents correct motion correction rather than objective-driven drift.
