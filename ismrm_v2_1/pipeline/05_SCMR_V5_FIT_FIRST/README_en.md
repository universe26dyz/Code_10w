# 05 SCMR-v5 fitting and 2D-fit-first

Run the SCMR-v5 MATLAB dictionary matcher and then NeSVoR/fit-first reconstruction on the server. The native-cache manifests must bind source observations, preprocessed MAT and generated maps by SHA256.

`run_stage.sh` invokes the server native-cache/fit-first launcher and retains per-case logs.

```bash
bash "$PROJECT_ROOT/common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_scmr_v5_fit_first.sh" \
  "$RUN_ROOT" "$RUN_ROOT/preprocessed" "$SUBJECT" "$RUN_ROOT/prepared" matlab
```

The expected completion marker is `SCMR-v5 Fit-first native maps and reconstructions: PASS`. Do not accept a cache with a different timing contract.
