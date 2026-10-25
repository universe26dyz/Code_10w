# Final5 AHA16 local runbook

This is a read-only downstream reporting module.  It does not run a
reconstruction, fitting, formal metric, or sparse-manifest operation, and it
never writes below the Final5 root.

## Inputs

Use only the following explicit artifacts:

- `05_formal_endpoints/central/proposed/<SUBJECT>/<CASE>/` and
  `05_formal_endpoints/central/fit_first/<SUBJECT>/<CASE>/` below the Final5
  root.  Their existing formal manifests are revalidated with the repository's
  endpoint validator.
- One verified standardized current native SAX folder containing
  `native_maps.npz`, `geometry.npz`, and `manifest.json`.
- One subject-specific audited ROI.  `--roi-root` may be the exact ROI folder,
  or the established `MultiMapCode/subjects` root: CYJ/HHZ/HJL use
  `20260821_corrected_sequence_v1`; DYZ uses `20260823_v5_newdata_v1`.
  `DYL_2` deliberately has no mapping and is always `ROI_PENDING` until its
  own new ROI is supplied.
- A reviewed `final5_aha_native_roi_mapping/v1` JSON.  For every basal, mid,
  and apical plane it records `native_group_idx`, its matching
  `original_group_idx`, `roi_slice_index`, `roi_transform`, and reviewed
  `anterior_xy`/`inferior_xy`.  The reviewed coordinates must match the
  transformed manual labels `{1,3,5}` (anterior) and `{2,4,6}` (inferior).
  It also SHA-binds the actual standardization manifest.  The adapter verifies the `derived_group_idx` →
  `original_group_idx` relationship; no filename order is used for CYJ's
  14→12 selection.

The mapping's `roi_transform` is one of `identity`, `transpose`, `flipud`,
`fliplr`, `transpose_flipud`, or `transpose_fliplr`.  It is a discrete audited
array-orientation declaration only; manual labels are never interpolated.

## Readiness

With the local Final5 root still unsynchronized, use this CPU-only command to
obtain explicit states rather than synthetic measurements:

```bash
python -m paper_experiments.joint_vs_2dfit.aha.run_final5_aha16 \
  --final-root /data/dengyz/dataset/Code_10w_v1/Code_10w_runs/phase4r_final_five_subject_k8_central_20261023 \
  --subject all --case L100 \
  --roi-root /home/universe/SVR/multimap_postprogramming/MultiMapCode/subjects
```

`ENDPOINTS_PENDING` is expected on this local machine until the central
endpoint root and its manifests are synchronized.  Do not run this command on
the server to generate formal outputs without a separate confirmation.

## Immutable execution after inputs are available locally

```bash
python -m paper_experiments.joint_vs_2dfit.aha.run_final5_aha16 \
  --final-root <FINAL5_ROOT> --subject CYJ --case L100 \
  --roi-root <SUBJECTS_ROOT_OR_EXACT_CYJ_ROI> \
  --mapping <CYJ_L100_AHA_MAPPING.json> \
  --native-sax-root <VERIFIED_STANDARDIZED_NATIVE_SAX_DIR> \
  --output <NEW_DOWNSTREAM_AHA_OUTPUT_DIR>
```

The output must not exist.  It contains one PNG and one PDF for each
`Proposed|Fit-first × T1|T2`, plus CSV/JSON provenance.  Both methods are
reprojected onto exactly the same native SAX lattice and use the same AHA
labels and paired-valid support:

`manual_myocardium_core AND finite(reference) AND proposed_reprojection_valid AND fit_first_reprojection_valid AND finite(proposed, fit_first)`.

Each segment's mean and SD use the historical sample-SD definition.  AHA is
basal 1–6, mid 7–12, apical 13–16 only; there is no segment 17.
