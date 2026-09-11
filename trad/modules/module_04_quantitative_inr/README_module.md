# Module 04 — Quantitative INR

`QuantitativeINR` consumes RAS-mm coordinates and uses NeSVoR's vendored
`HashEmbedder`, `build_encoding`, `build_network`, and
`compute_resolution_nlevel`.  A shared latent feeds four physical fields:
`T2=5+195*sigmoid`, `T1=T2+(2500-T2)*sigmoid`,
`B1=0.1+1.1*sigmoid`, and `A=softplus+eps`.

It has no positional encoding, deformation, bias, variance, or slice-scale
parameters.
