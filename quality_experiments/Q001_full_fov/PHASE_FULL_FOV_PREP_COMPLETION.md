# Q001 full-FOV preparation completion report

- Status: `CODE_READY_REAL_INPUT_RERUN_PENDING`.
- Implementation commit: `41331bd008e8f3d160887a7c75dca89f91d6dedb` — `Add Q001 full-FOV preprocessing and bridge`.
- Pre-run path hotfix: `2d59356b15caf1c1291f431cb7413d497434ab39` corrects the independent MATLAB entry's baseline-helper root resolution; no data was run before the correction.
- Code ready: independent MATLAB full-FOV MIND/MP-PCA preprocessing and a Python adapter over the baseline SOP/timing/geometry bridge.
- Local validation: Q001 targeted regression tests (real HDF5-MAT plus generated DICOM geometry fixtures), baseline MAT/geometry/grouping regressions, Python compile/CLI checks, MATLAB `checkcode`, and local runner shell syntax checks.
- Baseline changes: none. `preprocess_stack_v1` and shared Python bridge files remain unchanged.
- Pre-run acceptance hotfix: `34b1e0b` (`Harden Q001 full-FOV input acceptance`). The preprocessing metadata now truthfully records `MIND_mag_reg` if MP-PCA is disabled; formal input acceptance permits only exact `MP-PCA(full-FOV MIND_mag_reg)` plus the exact HB1 full-FOV geometry rule. Strict cropped-vs-full QC is now part of the runner and produces a hash-bearing `q001_input_manifest.json` only as reconstruction-ready after all three stacks pass.
- Failed real input-generation attempt: `/home/universe/SVR/data/Code_10w_v1/Q001_full_fov_inputs_v1` is preserved. It failed before any successful preprocessing completion because Q001's geometry validator used MATLAB-incompatible temporary-result chained indexing. Its retained manifest is `FAILED_INPUT_QC` with no stack artifacts. `49f176c` mirrors the baseline-compatible geometry helper, preserves all geometry checks, and adds a synthetic MATLAB runtime valid/mismatch regression. The next formal root is `/home/universe/SVR/data/Code_10w_v1/Q001_full_fov_inputs_v2`.
- Local real CYJ preprocessing: NOT RUN. This is not a code blocker: the user must explicitly select both a fresh `Q001_OUTPUT_ROOT` and `Q001_BASELINE_PREPARED_ROOT` with the verified `CYJ/{sax,2ch,4ch}` baseline layout. The runner will not guess or risk historical baseline inputs.
- Real prepared inputs available: NO.
- Exact local command: `Q001_OUTPUT_ROOT=/absolute/new/q001_inputs Q001_BASELINE_PREPARED_ROOT=/absolute/current/baseline_prepared bash quality_experiments/Q001_full_fov/run_Q001_full_fov_local_CYJ.sh`.
- Prohibited phases: no reconstruction, Q001A/Q001B, D3, or D2 formal run was started.
- PUSH PERFORMED: NO.
