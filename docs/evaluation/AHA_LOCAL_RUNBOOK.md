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
- One subject-specific Final5-ready ROI.  For CYJ, DYZ, HHZ, and HJL this is
  the derived `20261025_final5_cropped_roi_v1` bundle, never the historical
  full-FOV folder.  `--roi-root` may be that exact bundle or the established
  `MultiMapCode/subjects` root.  `DYL_2` uses its own audited
  `20261025_final5_aha_v1` bundle and never falls back to DYL.
- A reviewed `final5_aha_native_roi_mapping/v1` JSON.  For every basal, mid,
  and apical plane it records `native_group_idx`, its matching
  `original_group_idx`, `roi_slice_index`, `roi_transform`, and reviewed
  `anterior_xy`/`inferior_xy`.  The reviewed coordinates must match the
  transformed manual labels `{1,3,5}` (anterior) and `{2,4,6}` (inferior).
  It also SHA-binds the actual standardization manifest and, for a derived
  legacy-subject ROI, the ROI derivation manifest.  The adapter verifies the
  `derived_group_idx` → `original_group_idx` relationship; no filename order
  is used for CYJ's 14→12 selection.

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
## Final5 legacy full-FOV ROI adaptation

For CYJ, DYZ, HHZ, and HJL, never pass the historical full-FOV ROI folder
directly to Final5 AHA.  First create one immutable
`20261025_final5_cropped_roi_v1` bundle using the explicit Final5 SAX
reference and standardization manifest from the completed formal run.

The derivation config is JSON with this required structure:

```json
{
  "schema": "final5_legacy_roi_derivation_config/v1",
  "subject": "CYJ",
  "legacy_roi": {"path": "/.../CYJ/data/roi/20260821_corrected_sequence_v1"},
  "final5_sax_reference": {"path": "/.../current_cropped_sax_composite.nii.gz", "sha256": "..."},
  "standardization_manifest": {"path": "/.../standardization_manifest.json", "sha256": "..."},
  "planes": [
    {"native_group_idx": 0, "original_group_idx": 4, "legacy_roi_slice_index": 0, "target_reference_slice_index": 0}
  ]
}
```

There must be exactly one `planes` record per reference z slice.  Each
`native_group_idx -> original_group_idx` pair must match the formal
standardization manifest; `legacy_roi_slice_index` and
`target_reference_slice_index` are explicit, zero-based NIfTI indices.
The tool proves that each target pixel maps to an integer old-grid pixel and
rejects cropped myocardium or landmarks.  It never resamples.

Before formal inputs are locally available, only run the non-writing
readiness check:

```bash
python -m paper_experiments.joint_vs_2dfit.aha.run_final5_legacy_roi \
  --config /absolute/path/CYJ_final5_roi_derivation.json
```

After checking `READY`, create a new bundle (never reuse an existing output):

```bash
python -m paper_experiments.joint_vs_2dfit.aha.run_final5_legacy_roi \
  --config /absolute/path/CYJ_final5_roi_derivation.json \
  --output /home/universe/SVR/multimap_postprogramming/MultiMapCode/subjects/CYJ/data/roi/20261025_final5_cropped_roi_v1
```

Inspect `roi_qc_montage.png` in the derived bundle; it overlays the directly
copied mask on the current Final5 reference.  The bundle records its exact
mask/reference geometry and hashes in `roi_derivation_manifest.json`.  Pass that derived directory as the AHA
`--roi-root`.  Proposed and Fit-first then share this one ROI and the same
paired support; no method-specific ROI is created.
