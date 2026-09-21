# Legacy Myocardium Mask Audit

Audit date: 2026-09-21  
Scope: read-only inventory of `MultiMapCode/subjects/{CYJ,DYL,DYZ,HHZ,HJL}/data/roi`; no legacy file was modified.

## Result

**CYJ integration status: STOP-2 — legacy manual mask geometry is not provenance-compatible with its segmentation reference.** The same problem occurs for DYL, DYZ, and HJL. Only HHZ's manual segmentation headers match its composite reference. Consequently no legacy mask has been applied to Code_10w/Trad, no myocardium-only metric or overlay has been generated, and no 3-D propagation has been attempted.

This status is deliberately strict: equal array shape is not accepted as evidence of equal physical geometry.

## Legacy semantics established from source

The actual old implementation is `MultiMapCode/method_repositories/2D_fit_first/python/evaluation/roi/build_myocardium_masks.py`.

- `Segmentation_1-out-label.nii.gz`: manually drawn epicardium-enclosed **filled** region.
- `Segmentation_2-in-label.nii.gz`: manually drawn endocardium-enclosed **filled** region; the blood-pool candidate.
- `Segmentation_3-label_point.nii.gz`: landmark labels 1–6. Labels `[1,3,5]` are anterior and `[2,4,6]` inferior.
- `blood_pool = inner & outer`; `epi_enclosed = outer`; `myocardium_full = outer & ~blood_pool`.
- `myocardium_core` is derived, not independently hand-drawn: per-slice Euclidean distance transform of `myocardium_full`, retained where distance is `> 1.0 mm`.

The old formal evaluation config selects `myocardium_core` for AHA but retains full/core masks for myocardial metrics. Its native-to-final-pose routine uses nearest-neighbour resampling followed by a correspondence-derived orientation mapping. This audit does **not** treat that as evidence that the current Code_10w geometry is valid.

## Inventory and geometry result

All manual files are gzip-compressed NIfTI, and each triad shares shape, dtype, spacing, and labels within a subject. They are SAX-only; no 2CH/4CH manual ROI was found. The nonempty axial index ranges below are zero-based NIfTI indices.

| Subject | Manual triad shape | outer/inner labels | landmark labels | Nonempty outer slices | Geometry status |
| --- | --- | --- | --- | --- | --- |
| CYJ | 256×288×14, int16, 1.25×1.25×8.000001 mm | 0,1 | 0–6 | 1–10 | STOP |
| DYL | 256×288×14, int16, 1.25×1.25×8.0 mm | 0,1 | 0–6 | 2–10 | STOP |
| DYZ | 288×256×12, int16, 1.25×1.25×7.999998 mm | 0,1 | 0–6 | 2–9 | STOP |
| HHZ | 256×288×12, int16, 1.25×1.25×8.000005 mm | 0,1 | 0–6 | 1–9 | PASS against its legacy composite only |
| HJL | 288×256×12, int16, 1.25×1.25×8.000001 mm | 0,1 | 0–6 | 2–10 | STOP |

The discovery tool records path, type, shape, dtype, exact label values, nonempty slices, spacing, affine, mtime, byte size, and SHA-256 for every source file. The resulting hashes for CYJ are:

```text
Segmentation_1-out-label.nii.gz   8e4e592ccce175622019ed605ca524354e1b3540b24343590e65059ab2c83ca3
Segmentation_2-in-label.nii.gz    a51e5fcf660d5a42ab237a487810f160ec971b6dac736488e1b4980fb0a5b055
Segmentation_3-label_point.nii.gz 997e92bd57661f533c89b03dc8d5ba00d4464bd5bbceccd15a9a74d7ad6c3a11
sax_composite_reference.nii.gz    4378e3f3b9a0a8a33f83d10ef6f58dc01470f8342659d2010f6cb0eb8c0d6d4f
```

## CYJ correspondence audit

The current Code_10w prepared SAX manifest has 14 contiguous groups, each with 10 observations, 145×129 cropped pixels, and `affine_lps_rc`. Its `full_affine_lps_rc` agrees exactly with the geometry represented by the legacy `sax_composite_reference.nii.gz` after the explicit NIfTI representation transform:

```text
legacy composite voxel [x, y, z] -> current full voxel [row, col, slice]
                                      = [y, x, 13 - z]
```

This is a deterministic format/axis relation, not an inferred visual correction. It confirms that the legacy composite and current full SAX geometry describe the same native acquisition before the current prepared crop.

The three CYJ manual segmentation headers instead have this source-to-composite voxel transform (minor floating-point terms omitted):

```text
[[ 1, 0,  0,  0],
 [ 0, 1,  0,  0],
 [ 0, 0, -1, 13],
 [ 0, 0,  0,  1]]
```

That is a z reflection and a 13-slice translation, so it is **not** the identity mapping required for a manual label map and the reference volume it purportedly segments. Their identical 256×288×14 shape does not repair this conflict. There is no documented geometry-repair provenance for the CYJ segmentation triad. DYL, DYZ, and HJL have the same z reflection/translation pattern; HHZ alone has identity-equivalent headers.

The old DYL `geometry_repair_audit.json` applies only to older DYL files in `data/roi/`, not to the later `20260821_corrected_sequence_v1/Segmentation_*` triad. It cannot justify an automatic repair for CYJ or the other affected subjects.

## Existing old geometry logic and Code_10w adaptation boundary

Useful, source-proven logic:

- `evaluation/geometry/build_slice_correspondence.py`: matches native maps and registered output independently using pixel matching plus a discrete orientation audit; it does not infer IDs from file order.
- `evaluation/roi/map_native_roi_to_final_pose.py`: maps a validated native ROI into a final registered plane using nearest-neighbour resampling and the correspondence orientation mapping.
- `evaluation/reprojection/generate_psf_matched_reprojections.py`: uses final registered NeSVoR geometry for its legacy route.

Required Code_10w adaptation, which is not yet allowed:

```text
manual mask on a verified native full grid
-> explicit, provenance-recorded crop to current native group grid
-> direct native-grid use for synthetic apparent-map metrics
-> optional final-pose 3-D occupancy/nearest-neighbour propagation
```

The initial arrow cannot currently be established for CYJ because the source label header conflicts with the reference header. Applying a transpose, flip, `rot90`, or z reversal merely because it looks plausible would violate provenance.

## Reusable diagnostic

`evaluation_common/myocardium/legacy_masks.py` is decoder-agnostic and read-only. It discovers exactly one segmentation triad per subject, fingerprints it, compares every mask affine to its sibling composite reference, and returns `STOP` for missing/ambiguous files, shape mismatch, or non-identity geometry. It does not resample or repair masks.

Example:

```bash
python -m evaluation_common.myocardium.legacy_masks \
  --subjects-root /home/universe/SVR/multimap_postprogramming/MultiMapCode/subjects \
  --subject-id CYJ \
  --output /tmp/cyj_legacy_mask_audit.json
```

## Minimal evidence needed to resume

Provide a provenance record or a verified exported labelmap that establishes the exact physical transform from each `Segmentation_*` image to `sax_composite_reference`. It must be based on the manual-segmentation source/scene or DICOM geometry, not a visual fit. Once that is available, rerun the audit and only proceed if it reports `PASS`; then derive a new run-local mask bundle with source SHA-256 and explicit transform manifest.
