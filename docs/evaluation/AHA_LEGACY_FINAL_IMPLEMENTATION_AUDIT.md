# Legacy AHA Final Implementation Audit

## Scope

This audit identifies the historical SCMR/2D-fit-first AHA implementation
that may be migrated as algorithmic provenance only.  Historical
reconstructions, AHA CSV/JSON values, and rendered figures are not valid
inputs to the Final5 calculation.

## Final historical call chain

The documented batch entrypoint is
`MultiMapCode/method_repositories/2D_fit_first/python/evaluation/batch/run_dual_version_cohort_evaluation.py`.
`EVALUATION_PROGRAMS_USAGE_ORDER.md` records the required subject-level order:

1. `evaluation.roi.build_myocardium_masks`
2. `evaluation.roi.map_native_roi_to_final_pose`
3. `evaluation.aha16.build_aha16_masks`
4. `evaluation.aha16.compute_aha16_stats`
5. `evaluation.aha16.validate_aha16`
6. `visualization.scmr.make_figure3_bullseye`

The batch entrypoint invokes the subject evaluator and then the cohort Figure3
entrypoint.  `make_all_scmr_figures.py` invokes the subject Figure3 entrypoint.
The reusable single-bullseye primitive is `visualization.plotting.bullseye`;
the historical final Figure3 entrypoint itself composes a 2x2 montage, so only
the primitive's geometry, labels, LUT handling, and limits are transferable to
the requested one-file-per-method output.

## Algorithm actually implemented

- **AHA model:** AHA16 only: basal segments 1-6, mid 7-12, apical 13-16.  No
  17th apical segment is implemented.
- **Level selection:** an `aha_selection.yaml` supplies one native SAX slice
  for each basal/mid/apical level.  The dual-version batch derives this from
  the six-point label map; it does not infer levels from reconstruction data.
- **Myocardium:** filled outer mask minus filled inner mask, with a historical
  1.0 mm distance-transform erosion for `myocardium_core`.
- **Orientation:** points labelled 1/3/5 are anterior and 2/4/6 inferior.
  `sectorize()` starts at the anterior ray and chooses its traversal sign by
  placing the inferior ray closest to the opposite half-circle.
- **Statistics:** for each parameter/level/segment, the historical stats
  function intersects segment labels with its evaluation support and writes
  mean and SD for both maps, plus signed and absolute mean differences.
  Empty/insufficient segments fail later validation; the configured minimum
  is 10 pixels.
- **Display:** `bullseye()` draws three rings with a top/anterior start,
  segment number and rounded value labels, white boundaries, `lightgray` for
  missing values, and equal physical aspect.  Historical map ranges are T1
  0-2500 ms and T2 0-200 ms.

## Source provenance

| File | SHA256 | Role |
|---|---|---|
| `python/evaluation/roi/build_myocardium_masks.py` | `66737d85727d817ffb8a5a303fd0960b415994572e525b8f7d43e553dc435fcd` | Filled ROI/core and landmark masks |
| `python/evaluation/aha16/build_aha16_masks.py` | `3511e164927db903f22d9ef7aadfb26a4a3b396fa5069eaf3028a309aa81ba4a` | AHA16 sector assignment |
| `python/evaluation/aha16/compute_aha16_stats.py` | `0daa5351ded9f53fa3a2f8002ec4ebe416ef4a0f34888c40d2916613be362456` | Segment mean/SD statistics |
| `python/evaluation/aha16/validate_aha16.py` | `21a24b9146e11ef8e15570126f5698399a365cbbbb5ac7af5056d77cb38491c8` | AHA16 presence/support validation |
| `python/visualization/plotting.py` | `a43b18820d7bde357514de3aada3bfd7444154246e67934c3e1ffa1993b01e93` | Reusable bullseye primitive |
| `python/visualization/scmr/make_figure3_bullseye.py` | `35df248a49b83e138697b4268611d6b5e02483eaff1d392395179be34d8179da` | Historical final Figure3 composition |
| `python/evaluation/batch/run_dual_version_cohort_evaluation.py` | `9cb14707587b22fdd8f39e3a444c28bd502e610e3d71080f1a8fa3a8183c326d` | Documented batch entrypoint |

## Final5 migration boundary

The migrated implementation must retain the pure ROI/AHA16/mean-SD/bullseye
logic above.  It must replace all historical reconstruction, slice
correspondence, common-mask, and result-loading paths with a Final5 adapter:
verified central endpoints are reprojected with the current native SAX affine
onto the same retained native pixels; the same manual AHA labels are then
applied to Proposed and Fit-first independently.  The adapter must record
paired-valid coverage separately and must not alter anatomical membership.

## Current local readiness, checked 2026-10-25

- Manual ROI triplets plus `sax_composite_reference.nii.gz` exist for CYJ,
  DYZ, HHZ, and HJL at the versions supplied by the task.
- No DYL_2 ROI directory/files exist.  DYL ROI is not a substitute; DYL_2 is
  `ROI_PENDING` until its own triplet and composite reference are supplied.
- The local Final5 root
  `/home/universe/SVR/data/Code_10w_v1/Code_10w_runs/phase4r_final_five_subject_k8_central_20261023`
  is absent.  Current endpoint maps cannot be evaluated locally until the
  minimum explicit Final5 endpoint/native-dictionary inputs are synchronized.
