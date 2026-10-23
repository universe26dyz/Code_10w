# Intensity-preserving Fit-first

NeSVoR's stack-registration preprocessing divides each stack by its 99th
percentile.  `--no-slice-scale` does not disable that preprocessing.  Feeding
those registered output pixels directly into a later quantitative endpoint can
therefore introduce a parameter-specific multiplicative error.

The new route separates pose from intensity:

1. Stage 1 runs `nesvor register --registration stack` independently for T1
   and T2, and retains only its final slice affines.
2. The route writes individual Stage-2 slices with those affines, but uses the
   frozen native dictionary values (`T1/2500`, `T2/200`; non-finite background
   is zero).
3. A fail-closed gate checks source group identity/order, affine equality,
   values, finite background, SHA provenance, and a fresh destination.
4. Stage 2 runs `nesvor reconstruct --input-slices` without a registration
   argument.  Its input is the gated slice directory, so no second pose
   optimisation or q99-derived pixel values enter reconstruction.

The legacy `build_independent_nesvor_commands()` route remains historical and
unchanged.  The new route does not alter dictionary fitting, masks, decoder,
Proposed reconstruction, or evaluation semantics.
