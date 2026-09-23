# Native reference and myocardium-mask migration guide

The verified CYJ native-reference and `myocardium_masks_v2` bundles have now
been copied to the configured server destinations. Do not create substitutes
or replace the copied bundle contents. The following local paths remain the
recorded source provenance.

## Recorded copy provenance

| Local source | Server destination | Purpose | Used by |
|---|---|---|---|
| `/home/universe/SVR/data/Code_10w_v1/Code_10w_runs/trad_v2/CYJ_stackinit_TV_baseline/native_reference_2d/` | `/data/dengyz/dataset/Code_10w_v1/reference_data/CYJ/native_reference_2d/` | Verified native 2-D T1/T2 maps and their provenance manifest | `trad/evaluation/scmr/reference_2d.py`, `run_psf_k8_postprocess.py`, `run_scmr_fig12.py` |
| `native_reference_2d/native_reference_manifest.json` | same relative destination | Subject/preprocessing/MS units and source-hash gate | `reference_2d.load_verified_reference` |
| `native_reference_2d/{sax,2ch,4ch}.npz` | same relative destination | Native T1/T2 maps, valid support and group indices | SCMR quantitative metrics |
| `native_reference_2d/{sax_T1_stack_ms,sax_T2_stack_ms}.nii.gz` | same relative destination | Native SAX display planes | `run_scmr_fig12.py` |
| `/home/universe/SVR/data/Code_10w_v1/Code_10w_runs/trad_v2/CYJ_stackinit_TV_baseline/evaluation_scmr/myocardium_masks_v2/` | `/data/dengyz/dataset/Code_10w_v1/reference_data/CYJ/myocardium_masks_v2/` | Validated SAX myocardium/blood-pool support bundle | `run_psf_k8_postprocess.py`, `psf_k8_postprocess_common.py` |
| `myocardium_masks_v2/manifest.json` | same relative destination | Bundle schema and PASS status | PSF-K8 stage-2 validation |
| `myocardium_masks_v2/sax_myocardium_masks.npz` | same relative destination | `myocardium_core_1px`, `myocardium_full`, and `myocardium_core_legacy` arrays | Myocardium metrics |

The bundles were copied recursively with file names preserved. Do not replace
the prepared source MAT files: the native-reference manifest intentionally
checks their hashes under the server preprocessed root.

## Configure the server

The active `config/reference_paths.yaml` is now configured as:

```yaml
reference_data:
  native_reference_root: /data/dengyz/dataset/Code_10w_v1/reference_data/CYJ/native_reference_2d
  myocardium_mask_root: /data/dengyz/dataset/Code_10w_v1/reference_data/CYJ/myocardium_masks_v2
```

Before any formal evaluation, run the runtime/preflight checks in
`SERVER_DECODER_ONLY_WORKFLOW.md`; they must confirm required files, native
reference provenance, and myocardium manifest schema/PASS status. This update
does not claim that a formal evaluation has been run. The
`trad.evaluation.scmr.reference_paths.load_reference_paths` helper validates
the configured required files. The PSF-K8 stage-2 command accepts explicit
`--native-reference` and `--mask-bundle` overrides; otherwise it uses this
configuration.  If either bundle is absent, it emits a clear warning, writes a
`PARTIAL_EVALUATION_SKIPPED` manifest, and does not fabricate myocardium or
native-reference metrics.  Reconstruction is unaffected because it does not
load this optional evaluation configuration.

The optional extra NIfTIs in the mask bundle (`myocardium_core_1px.nii.gz`,
`myocardium_full.nii.gz`, `blood_pool.nii.gz`, and QC assets) should be kept
with the bundle to retain its audit trail, though the stage-2 metric reader
uses the NPZ payload.
