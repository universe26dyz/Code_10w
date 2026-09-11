# Module 06 — Rigid group pose and PSF forward

`GroupRigidPSF` owns exactly one vendored-NeSVoR axis-angle rigid pose per
native spatial `group_idx`.  It starts from the cropped HB1 affine expressed
as RAS-mm local-to-world geometry.  It exposes `deformable = false` and has no
deformation network or parameters.

PSF noise is anisotropic Gaussian noise in local `[column,row,slice]` mm axes,
with sigma from NeSVoR `resolution2sigma(..., isotropic=False)`, before the
group transform.  `TradQuantitativeForward` evaluates every PSF world sample,
simulates every sample's 10-HB fingerprint, selects the observed weight,
multiplies by `A`, and only then averages samples.
