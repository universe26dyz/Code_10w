# PSF K=8 Split Evaluation and Myocardium Core ROI Report

Date: 2026-09-21

## Status and retained history

The earlier all-in-one K=8 evaluator correctly stopped locally: the trained checkpoint is a CUDA tiny-cuda-nn HashGrid and this host has no CUDA device. No CPU parameter reinterpretation was attempted.

The evaluator is now split, without changing any reconstruction, checkpoint, Bloch physics, or K=1 output:

- Stage 1 server GPU reprojection: `trad/evaluation/scmr/run_psf_k8_gpu_stage.py`
- Stage 2 local MATLAB/metrics: `trad/evaluation/scmr/run_psf_k8_postprocess.py`

Stage 1 performs only CUDA/tiny-cuda-nn checkpoint loading, final-pose equality verification, deterministic K=8 signal reprojection, and transferable signal provenance. It imports neither MATLAB nor myocardium masks. Stage 2 never loads a checkpoint or tiny-cuda-nn; it validates Stage-1 hashes and K=8 provenance before MATLAB matching, QC, and metrics.

No server output is claimed in this local execution.

## Myocardium core ROI v2

The actual legacy source `2D_fit_first/python/evaluation/roi/build_myocardium_masks.py` confirms the historical filled-mask implementation:

```text
myocardium_core = distance_transform_edt(myocardium_full,
                                         sampling=(row_spacing_mm, col_spacing_mm)) > 1.0
```

For CYJ's 1.25×1.25-mm grid, the first foreground pixel layer has EDT 1.25 mm, so it survives `>1.0`. This is a historical ROI-definition/design issue, not a geometry bug. The exact legacy-compatible result is preserved as `myocardium_core_legacy`; it is identical to `myocardium_full` for every annotated CYJ slice.

The new primary ROI is explicitly named `myocardium_core_1px`:

```text
binary_erosion(myocardium_full, structure=ones((3,3)), iterations=1,
               border_value=0)
```

It is applied independently to each SAX slice, uses no interpolation, and is an explicit one-pixel erosion—not a claim of 1-mm erosion. New K=8 myocardial metrics will use `myocardium_core_1px` as primary and retain `myocardium_full` and `myocardium_core_legacy` as separate secondary rows.

The non-overwriting v2 bundle is:

```text
/home/universe/SVR/data/Code_10w_v1/Code_10w_runs/trad_v2/CYJ_stackinit_TV_baseline/evaluation_scmr/myocardium_masks_v2
```

Its manifest has source hashes, exact transforms, crop, spacing, both core definitions, and per-slice counts. It includes 14 composite/T1/T2 all-slice QC panels with outer, inner, full, legacy core, and corrected 1px core contours.

## Copy-paste server Stage-1 command

Run this after the committed source is present at `/data/dengyz/code/Code_10w`:

```bash
cd /data/dengyz/code/Code_10w
conda run --no-capture-output -n cr_dreme python -m trad.evaluation.scmr.run_psf_k8_gpu_stage \
  --subject-id CYJ \
  --prepared-root /data/dengyz/dataset/Code_10w_v1/Code_10w_prepared \
  --checkpoint /data/dengyz/dataset/Code_10w_v1/Code_10w_runs/trad_v2/CYJ_stackinit_TV_baseline/model.pt \
  --config /data/dengyz/dataset/Code_10w_v1/Code_10w_runs/trad_v2/CYJ_stackinit_TV_baseline/config_resolved.yaml \
  --final-poses /data/dengyz/dataset/Code_10w_v1/Code_10w_runs/trad_v2/CYJ_stackinit_TV_baseline/final_rigid_poses.json \
  --output /data/dengyz/dataset/Code_10w_v1/Code_10w_runs/trad_v2/CYJ_stackinit_TV_baseline/evaluation_scmr/psf_k8_reprojection_stage1 \
  --device cuda:0
```

Expected transfer products are:

```text
signals/signal_reprojection_sax.npz
signals/signal_reprojection_2ch.npz
signals/signal_reprojection_4ch.npz
stage1_manifest.json
TRANSFER_SHA256SUMS.txt
```

Stage-1 provenance records checkpoint/config/prepared-input SHA-256 values, CUDA/tiny-cuda-nn confirmation, PSF `enabled=true`, `n_samples=8`, seed `20260921`, operation order, final-pose tensor hash/shape/equality, and signal SHA-256 values.

## Transfer and local Stage-2 commands

Run the transfer from a host that can access both paths:

```bash
rsync -av --checksum --partial \
  /data/dengyz/dataset/Code_10w_v1/Code_10w_runs/trad_v2/CYJ_stackinit_TV_baseline/evaluation_scmr/psf_k8_reprojection_stage1/ \
  /home/universe/SVR/data/Code_10w_v1/Code_10w_runs/trad_v2/CYJ_stackinit_TV_baseline/evaluation_scmr/psf_k8_reprojection_stage1/
```

Then run locally:

```bash
cd /home/universe/SVR/multimap_postprogramming/Code_10w
conda run --no-capture-output -n knesvr_torch python -m trad.evaluation.scmr.run_psf_k8_postprocess \
  --subject-id CYJ \
  --stage1-root /home/universe/SVR/data/Code_10w_v1/Code_10w_runs/trad_v2/CYJ_stackinit_TV_baseline/evaluation_scmr/psf_k8_reprojection_stage1 \
  --preprocessed-root /home/universe/SVR/data/Code_10w_v1/Code_10w_preprocessed \
  --native-reference /home/universe/SVR/data/Code_10w_v1/Code_10w_runs/trad_v2/CYJ_stackinit_TV_baseline/native_reference_2d \
  --mask-bundle /home/universe/SVR/data/Code_10w_v1/Code_10w_runs/trad_v2/CYJ_stackinit_TV_baseline/evaluation_scmr/myocardium_masks_v2 \
  --output /home/universe/SVR/data/Code_10w_v1/Code_10w_runs/trad_v2/CYJ_stackinit_TV_baseline/evaluation_scmr/psf_k8_evaluation_v2
```

Stage 2 verifies every transferred signal hash and rejects any Stage-1 provenance other than K=8/seed 20260921. It then runs the original MATLAB matcher, writes apparent T1/T2, 41 signal QC panels, global SAX/2CH/4CH metrics, and distinct SAX myocardial metrics for full, legacy core, and corrected core.
