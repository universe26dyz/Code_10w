# Final5 Legacy ROI Adaptation Design

The historical CYJ, DYZ, HHZ, and HJL ROI bundles remain immutable.  A
downstream-only adapter will derive a new Final5 bundle from each historical
bundle only when an explicit Final5 SAX geometry/group manifest proves an
integer physical crop and a declared group correspondence.  It shall not
resample a mask, search for inputs, or use method-specific masks.

The derived bundle contains the existing four AHA ROI files on the current
cropped SAX grid and an immutable derivation manifest binding legacy inputs,
current geometry, group mapping, and output hashes.  The AHA evaluator will
accept only a derived bundle for automatic Final5 subject-root resolution;
callers may still pass one exact legacy folder explicitly for legacy-only
workflows.

For each AHA plane, both central endpoints are already reprojected to the
same Final5 native SAX lattice.  The shared support remains:
`manual_myocardium_core AND finite(reference) AND proposed_reprojection_valid
AND fit_first_reprojection_valid AND finite(proposed, fit_first)`.

If the Final5 manifests are unavailable locally, the tool reports an explicit
input-pending condition and creates no bundle.  It never falls back to a
historical crop or reconstructs any endpoint.
