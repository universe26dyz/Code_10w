# HHZ Fit-first-only Final-pose Downstream Design

## Goal

Provide an immutable, single-method downstream route that validates each corrected HHZ reconstruction before its paired method has completed. It must use the method's current final-pose native-plane bundle, not the old endpoint-to-native sampling route.

## Inputs

- Corrected-run native cache (`native_maps.npz`, `geometry.npz`, and `manifest.json`) and matching prepared SAX observations, both supplied explicitly by the caller. No `ismrm_five_subject_analysis_v1` path is built into the code.
- Either a Fit-first native bundle (`fit_first_native_prediction_bundle/v2` or later), or a Proposed direct native bundle (`direct_native_prediction_bundle/v1`, `direct_sparse_native_prediction_bundle/v1`, or `cyj11_r3_direct_final_pose_native_bundle/v1`).
- Existing SCMR HHZ ROI bundle and case mapping only after the current prepared SAX observations SHA256 and native group/shape identities are proven to match.  A reused historical mapping must never be silently admitted.

## Processing

The new module reads native reference T1/T2 and one method's final-pose planes. For each native SAX group it intersects manual anatomical ROI/mask support with finite reference, final-pose validity, and finite prediction. It then produces per-plane and pooled myocardial metrics, plus AHA16 statistics/bullseyes from exactly that support.

It may reuse the established AHA labels, ROI loader, statistics, and rendering primitives, but will not call paired-only functions or fabricate a Proposed result.

## Outputs and provenance

Outputs live in a new diagnostic result root, never under the paired results.  Every manifest is immutable and records the corrected native cache, sparse manifest, final-pose bundle, reused ROI assets, and the exact support equation.  It is explicitly marked:

```text
diagnostic_only: true
comparison: native_reference_vs_single_method
not_a_proposed_vs_fit_first_result: true
sampling: final_pose
```

The historical HHZ AHA figures remain reference/QC assets only: their v1 manifests show endpoint sampling and lack `current_final_pose_bundles`, so they must not be reported as the corrected result.

## Failure behavior

The route fails before writing output if final-pose schema/arrays are absent, output already exists, case/group/shape differs, or the mapping cannot prove the supplied prepared SAX / ROI derivation identities. It never substitutes an endpoint volume or historical map.
