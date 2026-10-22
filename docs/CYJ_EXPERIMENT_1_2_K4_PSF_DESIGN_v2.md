# CYJ Experiment 1.2 K4 PSF design

`Q005S640_K4` uses current untrimmed CYJ L100 and changes only `training_psf_samples` from Q002/K8 to 4. It therefore has 640 anchors × 4 = 2,560 INR locations. Renderer/sample-PSF controls and central/psf128 endpoint semantics remain separate. This code window creates no K4 result and must not launch it before 1.1 review.
