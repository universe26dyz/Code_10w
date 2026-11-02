# 04 Sparse profiles

Derive L100, L25, L50 and five cross-stack profiles from one correct prepared subject root. This stage selects groups/layers and writes immutable sparse manifests; it does not fit maps.

`run_stage.sh` invokes the sparse profile builder and creates the eight case manifests.

```bash
bash "$PROJECT_ROOT/common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/build_sparse_profiles.sh" \
  "$RUN_ROOT/prepared" "$SUBJECT" "$RUN_ROOT/sparse_inputs" "$PYTHON"
```

Check prepared-root identity, source-observation SHA, generator git SHA and selected groups in every sparse manifest. Existing sparse subject directories are never overwritten.
