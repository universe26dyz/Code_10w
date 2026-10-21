# Multi-subject statistics plan

The independent statistical unit is subject. A sparse fold is a within-subject sampling replicate and must never be pooled as an independent subject.

For each method, endpoint (`central`, `psf128`), parameter (`t1`, `t2`) and metric (`rmse_ms`, `mae_ms`, `bias_ms`), retain:

1. `per_fold`: subject, condition, phase, formal case ID, paired support voxel count and metric values.
2. `within_subject`: L100 one value; L50/L25 arithmetic mean and sample SD across observed folds, preserving missing-fold status.
3. `per_subject`: one row per subject/method/endpoint/condition/parameter, carrying the source summaries and protocol/dependency identity.
4. `across_subject`: summary over per-subject values only, with subject count, mean, sample SD and confidence interval method declared before use.

Paired proposed-vs-Fit-first comparisons require the same endpoint and an explicitly audited common support. If the methods have different paired support, write `PAIRED_SUPPORT_MISMATCH` and do not calculate a paired comparison. Report central and PSF128 separately; do not present training K1/K8 as an endpoint dimension.

```json
{
  "schema": "multi_subject_native_metrics/v1",
  "subject": "DYL_2",
  "formal_case": "L50_fold1",
  "condition": "L50",
  "phase": 1,
  "endpoint": "central",
  "parameter": "t1",
  "method": "Q004S640",
  "metric": {"rmse_ms": 0.0, "mae_ms": 0.0, "bias_ms": 0.0},
  "paired_support": {"status": "VERIFIED", "voxel_count": 0},
  "provenance": {"metric_manifest_sha256": "required", "mask_manifest_sha256": "required"}
}
```

The numeric zeroes in the schema example are placeholders only and are not measurements. The enforceable machine-readable definition is `docs/MULTI_SUBJECT_STATISTICS_SCHEMA.json`.
