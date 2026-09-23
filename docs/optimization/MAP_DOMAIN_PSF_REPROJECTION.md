# Common map-domain PSF-matched reprojection

`trad.evaluation.scmr.map_domain_psf` is the shared fair-comparison operator
for exported `T1_3D.nii.gz` / `T2_3D.nii.gz` maps.  It is map-domain only:
it uses neither Bloch simulation nor dictionary matching.

The audited 2D-fit-first evaluation-v2 implementation is
`MultiMapCode/method_repositories/2D_fit_first/python/evaluation/reprojection/generate_psf_matched_reprojections.py`.
Its formal `volume_nesvor` path calls
`nesvor.svr.reconstruction.simulate_slices`, which obtains the Gaussian PSF
from `nesvor.utils.get_PSF`.  Native pixel centres and final poses are carried
by each `registered_slices` template; T1 and T2 templates are independently
loaded and therefore may have separate final poses.  The NeSVoR kernel uses
in-plane resolution as well as slice thickness: sigma is
`[SINC_FWHM*dx, SINC_FWHM*dy, GAUSSIAN_FWHM*dz]`.  `simulate_slices` samples
the reconstructed volume on that geometry and returns a separate acquisition
weight/support image.

The common operator calls the vendored `get_PSF` kernel, then exposes K=32/K=128
deterministic Monte-Carlo integration for the comparison path: independent
draws from that discrete NeSVoR kernel are rotated into final RAS coordinates,
trilinearly sampled from the exported isotropic volume, and averaged over
finite samples.  No central-plane resample is used.  `NativeGrid.affine_ras_rc` maps
`[row,col,slice]` to RAS-mm; its local x/y/z axes are col/row/normal.

For Trad, grids derive from prepared native geometry and the exported
`final_rigid_poses.json` `trans_first=true` matrices.  For 2D-fit-first,
grids derive separately from its final T1 and T2 registered-slice NIfTIs.

Example calls:

```bash
python -m trad.evaluation.scmr.run_map_domain_psf_comparison --method shared_svr --decoder-type Bloch \
  --subject-id SUBJECT --t1-volume T1_3D.nii.gz --t2-volume T2_3D.nii.gz \
  --prepared-root PREPARED --final-poses final_rigid_poses.json \
  --n-samples 32 --seed 20260921 --output OUTPUT

python -m trad.evaluation.scmr.run_map_domain_psf_comparison --method 2dfit \
  --subject-id SUBJECT --t1-volume T1_3D.nii.gz --t2-volume T2_3D.nii.gz \
  --t1-registered-slices T1/registered_slices --t2-registered-slices T2/registered_slices \
  --n-samples 128 --seed 20260921 --output OUTPUT
```
