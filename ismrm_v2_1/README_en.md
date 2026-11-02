# ismrm_v2_1

This is a standalone SCMR-v5 MultiMap execution tree. Operator-facing code is ordered in `pipeline/01_` through `pipeline/09_`; every module contains bilingual manuals and `run_stage.sh`. `common/` contains the import-compatible runtime code, `configs/` contains protocol and manifest templates, `subjects/` contains local subject inputs, and `runs/` contains generated results. Data and outputs are ignored by Git by default.

Run all new experiments from this root, with explicit `SUBJECT`, run roots and manifests. The server performs DICOM preprocessing, observation preparation, SCMR-v5 MATLAB fitting, 2D-fit-first, Proposed, final-pose and metrics. Local work is limited to manual ROI/mask drawing and copying the ROI bundle to the server.

The immutable contract is TR 2.61 ms, VPS 87, TI `[10,100]`, T2-preparation `[35,45,55]`, FA `[45,45,45]`, ramp-up 10, ten weights per group. Proposed accepts only decoder SHA `338f2c...`; the historical `[50,150]` decoder is invalid.

Start with the commands in `pipeline/01_RAW_DICOM/README_en.md`, then follow modules 02–09 in order. Every module documents inputs, outputs, execution host, SHA checks, pass markers and recovery from partial outputs.
