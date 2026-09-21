# PSF K=8 and Legacy Myocardium Mask Integration Report

Date: 2026-09-21
Starting HEAD: `26a97f5ef4fb696b52149e775df2fdf1f59a3505`

## Split status

- Track A — legacy myocardial mask geometry: **PASS**
- Derived CYJ current-native-grid mask bundle: **PASS**
- Track B — final-pose PSF K=8 reprojection: **FAIL (execution environment)**
- Same-dictionary synthetic apparent T1/T2: **FAIL (blocked by missing K=8 signals)**
- Global K=8 quantitative comparison: **FAIL (blocked by missing K=8 maps)**
- Myocardium-only comparison: **STOP (blocked by Track B only; mask provenance now passes)**

Track A and Track B were handled independently. No reconstruction, preprocessing, registration, or training was rerun.

## Track A outputs

The strict exact-reorientation classifier and zero-interpolation builder are complete. CYJ uses the proven source-label z reversal, actual ordered prepared-group geometry, integer crop `[71,63]`, and verified native-reference affine. Output shape is `[14,145,129]`; all 14 slices have composite/T1/T2 contour QC.

Final bundle:

```text
/home/universe/SVR/data/Code_10w_v1/Code_10w_runs/trad_v2/CYJ_stackinit_TV_baseline/evaluation_scmr/myocardium_masks
```

Legacy source data remained read-only. Figure 3/AHA and 3-D propagation were skipped.

## Track B implementation

The dedicated evaluator is `trad/evaluation/scmr/run_psf_k8_evaluation.py`. It is isolated from training and refuses to overwrite a non-empty output directory. It explicitly requests `PSF enabled=true`, `n_samples=8`, records a fixed seed, loads `trained_axisangle_train`, and verifies that the loaded `model.rigid_psf.axisangle` equals it.

The unchanged forward path is:

```text
local native pixel -> 8 Gaussian local PSF samples -> final Stage-B rigid pose
-> INR fields per sample -> Bloch per sample -> amplitude per sample -> mean signal
```

The deterministic test proves effective K is exactly 8 and identical seeds produce byte-identical predictions. A nonlinear regression test distinguishes the implemented mean-of-signals result from the forbidden Bloch-of-mean-parameters order.

The MATLAB wrapper `trad/evaluation/scmr/matlab/build_synthetic_apparent_maps.m` feeds predicted ten-weight tensors to the original:

```text
function_T1T2_10HB_bssfp.m
sim_T1T2_10HB_bssfp.m
```

with the exact current `preprocessed.mat` timing/protocol metadata and unchanged T1/T2/B1 dictionary lists. No Python fitter exists in this path. When run on the required CUDA backend, it will generate all SAX/2CH/4CH observed/predicted/residual archives, 41 all-group ten-weight QC panels, synthetic apparent maps, global metrics, and SAX full/core per-slice and summary metrics.

## Irreducible execution blocker

This host exposes no NVIDIA device (`torch.cuda.is_available() == false`, device count 0, and no `nvidia-smi`). The existing checkpoint was trained with tiny-cuda-nn and stores its HashGrid as `inr.encoding.params`. The vendored CPU fallback expects a structurally different set of per-level embedding tensors, so `load_state_dict` correctly fails rather than silently changing the learned field.

Checkpoint:

```text
/home/universe/SVR/data/Code_10w_v1/Code_10w_runs/trad_v2/CYJ_stackinit_TV_baseline/model.pt
SHA256 fc2ff229b0f58b975ef0dc2f544dd3dd51878b5e51993ad3dcc185e5b115e0b2
```

The checkpoint `trained_axisangle_train` was still verified read-only against `final_rigid_poses.json`: 41×6 poses, maximum physical JSON round-trip difference `3.0518e-5`, tensor SHA-256 `617ff03bce754d2b25ce6fe7ac142468732a1a928993f0b633dc6b1667a7c3c1`.

Implementing an unverified CPU reinterpretation of tiny-cuda-nn HashGrid parameters would risk changing the trained Trad field. It was deliberately rejected. Consequently no `evaluation_scmr/psf_k8_reprojection` directory or synthetic result was claimed on this host.

## Scientific freeze

Unchanged and not rerun: Module 01/02, MIND, MP-PCA, Bloch physics, INR parameterization, Stage A/B, HB1 initialization, training PSF/K=8 definition, TV, variance, amplitude guidance, existing K=1 exports, and Figure 1/2 scientific definitions.
