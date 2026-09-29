# Q001 full-FOV preprocessing and bridge

This independent input route performs `raw full-FOV DICOM → full-FOV MIND → full-FOV MP-PCA → full-FOV prepared observations`. It does not modify `preprocess_stack_v1`, use cropped results padded back to full size, or run reconstruction.

The MATLAB entry calls the existing MIND and MP-PCA algorithms with the unchanged HHZ settings, but does not apply the baseline central crop or its divisibility-by-four condition. Its v7.3 MAT stores `spatial_mode=full_fov`, zero crop offsets, full shape, full-FOV semantics, and HB1 group geometry provenance. The Python adapter validates those fields before delegating SOP resolution, timing, protocol checks, affine construction, and NPZ writing to the immutable baseline bridge.

No non-example deployment-path configuration is present in this repository. Therefore the local runner intentionally requires an explicit, new `Q001_OUTPUT_ROOT`; it never derives or writes a baseline input path:

```bash
Q001_OUTPUT_ROOT=/absolute/new/q001_inputs bash quality_experiments/Q001_full_fov/run_Q001_full_fov_local_CYJ.sh
```

It uses the checked-in CYJ deployment manifest for DICOM sources and writes `full_fov_preprocessed/` and `full_fov_prepared/` under that chosen root. It refuses a non-empty root.
