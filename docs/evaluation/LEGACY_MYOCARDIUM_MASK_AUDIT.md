# Legacy Myocardium Mask Audit

Audit date: 2026-09-21
Legacy root: `/home/universe/SVR/multimap_postprogramming/MultiMapCode/subjects`

## Revised result

The previous audit used the conservative rule `non-identity affine = STOP`. That historical STOP-2 was appropriate before an exact-lattice proof existed, but it was too restrictive as a permanent classifier.

The revised rule accepts a non-identity relation only as `EXACT_REORIENTABLE_GRID` when all of these pass: signed-permutation linear component, integer translation, shape agreement under permutation, complete corner/extent agreement, algebraic integer-lattice mapping, and physical-coordinate preservation within NIfTI/DICOM header precision. The implementation then permits only transpose/flip operations; it never resamples labels.

| Subject | Result | Manual triad → composite |
| --- | --- | --- |
| CYJ | PASS | `EXACT_REORIENTABLE_GRID` |
| DYL | PASS | `EXACT_REORIENTABLE_GRID` |
| DYZ | PASS | `EXACT_REORIENTABLE_GRID` |
| HHZ | PASS | `IDENTICAL_GRID` |
| HJL | PASS | `EXACT_REORIENTABLE_GRID` |

The machine-readable record is `docs/evaluation/legacy_myocardium_mask_audit.json` (`legacy_myocardium_mask_audit/v2`). It includes every source path/SHA-256, affine, shape, signed-permutation checks, corner/lattice checks, inferred axes/signs, and maximum physical-coordinate error.

## CYJ proof

For all three CYJ labelmaps,

```text
mask voxel [x,y,z] -> composite voxel [x,y,13-z]
```

up to NIfTI header quantization (maximum corner physical error `2.24e-5 mm`). Label values and nonzero voxel counts are preserved exactly, and applying the inverse discrete transform restores the original array exactly.

The current prepared/native group order was rechecked from the translations of all 14 current `full_affine_lps_rc` matrices and from `observations.npz`, rather than from the nominal third affine column. This corrects one statement in the previous narrative: the verified composite-to-current ordered-volume relation is

```text
legacy composite [x,y,z] -> current full prepared [row,col,group] = [y,x,z]
```

not `[y,x,13-z]`. The latter follows only if the single-slice DICOM normal is incorrectly treated as the stored group-order direction. Direct physical origins prove composite z=0 equals current group 0; the current verified native-reference NIfTI uses the same group-to-group direction. The complete label-to-current relation therefore remains `[x,y,z] -> [y,x,13-z]`.

The current prepared crop is an integer offset `[row,col]=[71,63]`, yielding `[group,row,col]=[14,145,129]`. Its affine matches the verified native T1/T2 SAX stack within header precision.

## Derived CYJ bundle

No legacy source was changed. A new run-local bundle was generated at:

```text
/home/universe/SVR/data/Code_10w_v1/Code_10w_runs/trad_v2/CYJ_stackinit_TV_baseline/evaluation_scmr/myocardium_masks
```

It contains exact-grid outer/inner/landmark labels, blood pool, `myocardium_full`, `myocardium_core`, the cropped native composite, compressed NPZ, manifest, and 14 all-slice composite/T1/T2 contour QC images.

The legacy definitions are unchanged:

- `blood_pool = inner & outer`
- `myocardium_full = outer & ~blood_pool`
- `myocardium_core = distance_transform_edt(myocardium_full, sampling=(1.25,1.25)) > 1.0 mm`

At 1.25-mm in-plane sampling, the historical strict `>1.0 mm` rule retains every full-myocardium pixel, so CYJ full/core masks are identical. This is an observed consequence of the legacy definition, not a redefinition.

## Tests

Targeted tests cover identity, z reversal, x/y permutation, FOV-breaking integer shift, half-voxel shift, scale, shear, arbitrary rotation, exact label/count preservation, and inverse restoration. The signed-permutation plus exact-extents proof establishes the complete voxel lattice algebraically rather than enumerating every voxel.
