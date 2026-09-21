# PSF K=8 and Legacy Myocardium Mask Integration Report

## Status

**STOP — scientific integration intentionally not run.** The required CYJ legacy-manual-mask provenance check triggered STOP-2 before any evaluation integration. See [LEGACY_MYOCARDIUM_MASK_AUDIT.md](LEGACY_MYOCARDIUM_MASK_AUDIT.md).

## Git

- Starting HEAD: `e9b9293220a3cbbd1678d29ca8912fe029e16973`
- Ending HEAD: this report is committed with the diagnostic implementation; see the containing commit.
- Legacy data: read-only; no source legacy mask has been modified, renamed, moved, or overwritten.

## What was verified

- Current Trad reprojection uses `GroupRigidPSF.sample_local_then_transform` and `TradQuantitativeForward`: local PSF sample → final learned `axisangle` pose → per-sample INR fields → per-sample Bloch fingerprint → amplitude → signal mean. It does not average T1/T2/B1 before Bloch.
- Current baseline export receives `psf.export.enabled: false`; `export_native_plane_reprojections` therefore sets effective `n_samples = 1`. The existing `signal_reprojection_*.npz` files are legacy K=1 outputs.
- Current checkpoint, reconstruction, Module 01/02 data, Bloch physics, PSF training configuration, TV, and Figure 1/2 science were not modified or rerun.
- Native reference dictionary matching is already based on the original MultiMap MATLAB matcher (`function_T1T2_10HB_bssfp.m` / `sim_T1T2_10HB_bssfp.m`), but no synthetic K=8 signals/maps were generated in this STOP state.

## Deliberately not generated

- No K=8 PSF signal reprojection.
- No synthetic apparent T1/T2 dictionary maps.
- No native-grid derived mask bundle, mask overlay, myocardium-only metrics, or 3-D propagated mask.
- No Figure 3/AHA integration.

Generating any of these from the mismatched CYJ mask header would create geometry claims that cannot be reproduced from the source data. The only implemented addition is a shared read-only diagnostic that prevents such a silent integration.

## Targeted test

`trad/tests/test_legacy_myocardium_mask_audit.py` asserts that a z-shifted source affine is `STOP` even if its image shape is identical to the reference. The test was run in the existing `knesvr_torch` environment.

## Resume condition

Resume only after an explicit, source-backed transform between CYJ `Segmentation_*` and `sax_composite_reference` is supplied or regenerated from the original segmentation scene. At that point, the audit must pass before implementing the PSF K=8 export and mask-dependent metrics.
