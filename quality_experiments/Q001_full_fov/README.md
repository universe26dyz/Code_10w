# Q001 full-FOV preprocessing and bridge

This independent input route performs `raw full-FOV DICOM → full-FOV MIND → full-FOV MP-PCA → full-FOV prepared observations`. It does not modify `preprocess_stack_v1`, use cropped results padded back to full size, or run reconstruction.

The MATLAB entry calls the existing MIND and MP-PCA algorithms with the unchanged HHZ settings, but does not apply the baseline central crop or its divisibility-by-four condition. Its v7.3 MAT stores `spatial_mode=full_fov`, zero crop offsets, full shape, full-FOV semantics, and HB1 group geometry provenance. The formal adapter accepts only the exact `MP-PCA(full-FOV MIND_mag_reg)` semantic string and exact HB1 geometry rule; a MIND-only or otherwise incomplete MAT is intentionally rejected. With MP-PCA disabled, the MATLAB entry records `MIND_mag_reg`, which is useful diagnostic provenance but cannot enter the formal route. The adapter validates those fields before delegating SOP resolution, timing, protocol checks, affine construction, and NPZ writing to the immutable baseline bridge.

No non-example deployment-path configuration is present in this repository. Therefore the local runner intentionally requires both an explicit, new `Q001_OUTPUT_ROOT` and the existing baseline prepared root `Q001_BASELINE_PREPARED_ROOT`; it never derives or writes a baseline input path. The latter must contain `CYJ/{sax,2ch,4ch}/{observations.npz,manifest.json,timing.npy,qc_summary.json}`:

```bash
Q001_OUTPUT_ROOT=/absolute/new/q001_inputs \
Q001_BASELINE_PREPARED_ROOT=/absolute/current/baseline_prepared \
bash quality_experiments/Q001_full_fov/run_Q001_full_fov_local_CYJ.sh
```

For each stack it writes `full_fov_preprocessed/`, `full_fov_prepared/`, and a strict cropped-vs-full QC JSON under the new root. The QC requires SOP/timing/protocol correspondence, matching group/observation counts, a valid in-FOV integer crop location, and the baseline crop affine-origin relationship; it never compares processed pixel values. `q001_input_manifest.json` records hashes and provenance for all three stacks and is `READY_FOR_Q001_RECON_IMPLEMENTATION` only when every strict QC passes; otherwise it is `FAILED_INPUT_QC`. The runner refuses a non-empty output root.
