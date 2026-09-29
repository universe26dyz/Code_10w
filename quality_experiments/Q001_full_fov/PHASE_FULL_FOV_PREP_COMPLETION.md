# Q001 full-FOV preparation completion report

- Status: `CODE_READY_LOCAL_INPUT_ROOT_REQUIRED`.
- Implementation commit: `41331bd008e8f3d160887a7c75dca89f91d6dedb` — `Add Q001 full-FOV preprocessing and bridge`.
- Code ready: independent MATLAB full-FOV MIND/MP-PCA preprocessing and a Python adapter over the baseline SOP/timing/geometry bridge.
- Local validation: Q001 targeted regression tests (real HDF5-MAT plus generated DICOM geometry fixtures), baseline MAT/geometry/grouping regressions, Python compile/CLI checks, MATLAB `checkcode`, and local runner shell syntax checks.
- Baseline changes: none. `preprocess_stack_v1` and shared Python bridge files remain unchanged.
- Local real CYJ preprocessing: NOT RUN. MATLAB and the three manifest-resolved CYJ DICOM directories are present, but the repository contains no non-example deployment-path configuration defining a new, authorized Q001 output root. The provided runner requires the user to set a fresh `Q001_OUTPUT_ROOT` rather than guess or risk historical baseline inputs.
- Real prepared inputs available: NO.
- Exact local command: `Q001_OUTPUT_ROOT=/absolute/new/q001_inputs bash quality_experiments/Q001_full_fov/run_Q001_full_fov_local_CYJ.sh`.
- Prohibited phases: no reconstruction, Q001A/Q001B, D3, or D2 formal run was started.
- PUSH PERFORMED: NO.
