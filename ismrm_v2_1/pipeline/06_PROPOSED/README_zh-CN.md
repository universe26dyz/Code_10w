# 06 Proposed

Proposed 使用原始 sparse observations、正确的 `[10,100]` decoder、B6、protocol 和 sparse manifest；它不读取 fitting map 作为输入。L100 必须使用 `$RUN_ROOT/prepared/$SUBJECT`，其他 case 使用对应 sparse case root。输出 root 必须不存在，避免 formal sparse execution 的 `fresh output root` 错误。

`run_stage.sh`：每次只执行一个 case，并在调用前检查 manifest 和 fresh output root。

```bash
DECODER="$PROJECT_ROOT/runtime/model_assets/decoder_scmr_v5_10_100.pth"
B6="$PROJECT_ROOT/runtime/model_assets/b6_model.pt"
for CASE in L100_fold0 L25_fold0 L50_fold0; do
  # 按项目脚本/README 中的显式 expected SHA 执行；每个 case 单独记录日志。
  echo "START Proposed $SUBJECT $CASE"
done
```

必须传入 decoder、B6、protocol、sparse manifest 和四个 expected SHA。完成后每 case 应有 T1/T2 NIfTI、experiment manifest 和 run provenance。
