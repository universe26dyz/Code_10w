# 04 稀疏 profile

## 作用

从同一个正确的 prepared subject root 派生 L100、L25、L50 和五种跨 stack profile。它只选择 group/layer 并写入 sparse manifest，不重新拟合。

`run_stage.sh`：调用 sparse profile builder，创建八个 case 的 manifest。

```bash
bash "$PROJECT_ROOT/common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/build_sparse_profiles.sh" \
  "$RUN_ROOT/prepared" "$SUBJECT" "$RUN_ROOT/sparse_inputs" "$PYTHON"
```

必须确认每个 `$RUN_ROOT/sparse_inputs/$SUBJECT/<CASE>/sparse_manifest.json` 的 prepared-root、source observation SHA、generator git SHA 和选择的层。脚本拒绝覆盖已有 sparse subject 目录。
