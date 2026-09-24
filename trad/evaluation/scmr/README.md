# Trad SCMR Figure 1/2 evaluation

This package adapts the old 2D-fit-first Figure 1/2 layout, connected/common
support principle, and masked metrics to the current quantitative Trad route.
It never performs a PSF average of T1/T2: quantitative agreement compares the
existing `t1_t2_native_plane_*.npz` export with native dictionary maps, while
signal data consistency uses the existing `signal_reprojection_*.npz` export.

The required `--native-reference-root` contains `native_reference_manifest.json`:

```json
{
  "subject_id": "CYJ",
  "preprocessing_semantics": "MP-PCA(MIND_mag_reg)",
  "map_units": "ms",
  "preprocessed_mat_sha256": {"sax": "...", "2ch": "...", "4ch": "..."},
  "maps": {"sax": "sax.npz", "2ch": "2ch.npz", "4ch": "4ch.npz"},
  "figure1_native_stacks": {"t1": "sax_T1_stack_ms.nii.gz", "t2": "sax_T2_stack_ms.nii.gz"}
}
```

Each map archive contains `t1_ms`, `t2_ms`, `valid_mask`, and `group_idx`, with
shape `[group,row,column]`. The SHA256 values must match the exact current
`Code_10w_preprocessed/<subject>/<stack>/preprocessed.mat` inputs. This is a
deliberate gate: an old mapping run cannot be silently used as the reference.

Run from `Code_10w` with the environment that includes nibabel/SimpleITK:

```bash
conda run --no-capture-output -n knesvr_torch python -m trad.evaluation.scmr.run_scmr_fig12 \
  --subject-id CYJ \
  --prepared-root /home/universe/SVR/data/Code_10w_v1/Code_10w_prepared \
  --trad-run /home/universe/SVR/data/Code_10w_v1/Code_10w_runs/trad_v2/CYJ_stackinit_TV_baseline \
  --native-reference-root /path/to/verified_native_reference \
  --dry-run
```

Figure 3/AHA is intentionally not implemented (`SKIPPED / ROI_PENDING`).

## Formal decoder-pair PSF postprocessing

`run_decoder_pair_evaluation` is a read-only second-stage evaluator. It reads
two already-complete map-domain PSF evaluation directories (Bloch and
FrozenMLP), their existing 3-D run exports, the same verified native-reference
bundle, and the SAX myocardium bundle. It does not rerun reconstruction or
PSF reprojection, and refuses to write into a non-empty output directory.
It uses the repository-vendored byte-identical legacy Lipari/Navia assets by
default; `--legacy-source` remains an optional external override.

```bash
python -m trad.evaluation.scmr.run_decoder_pair_evaluation \
  --subject-id CYJ \
  --trad-eval /runs/CYJ/trad/map_domain_psf \
  --mlp-eval /runs/CYJ/frozen_mlp/map_domain_psf \
  --trad-run /runs/CYJ/trad \
  --mlp-run /runs/CYJ/frozen_mlp \
  --native-reference /references/CYJ/native_reference \
  --preprocessed-root /prepared/preprocessed \
  --mask-bundle /references/CYJ/myocardium_mask \
  --output /runs/CYJ/evaluation/decoder_pair_full_evaluation
```

It rejects mismatched decoder identities, shared-SVR map-domain PSF contracts,
native-reference manifest hashes, or reference arrays. Final pose values are
not compared because each reconstruction independently optimizes them. Figure
1 is through-plane 3-D continuity QC; Figure 2 and all-slice panels are
PSF-matched native-plane comparisons on paired common support.

Legacy-component mapping:

- `make_figure1_throughplane_svr.py` and candidate montage code → modified
  migration: shared world-plane sampling, native nearest interpolation, and
  Trad linear interpolation.
- `make_figure2_reprojection_residual.py` → modified migration: native
  dictionary map versus existing Trad quantitative native-plane export, with a
  common p99 residual scale.
- `prepare_common_evaluation_mask.py` and `metrics.py` → modified migration:
  native-valid ∩ Trad-support ∩ finite values, boolean-indexed metrics.
- PSF quantitative reprojection, myocardial/AHA masks, bullseye, and
  Bland–Altman plotting are intentionally not migrated in this phase.
