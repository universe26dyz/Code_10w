# 05 SCMR-v5 fitting 与 2D-fit-first

## 作用

MATLAB 使用 SCMR-v5 dictionary matcher 生成 native mapping/cache；随后 NeSVoR 与 fit-first 入口生成 L100 和 sparse case 的 T1/T2 volume。此步骤必须在服务器运行，MATLAB matcher、simulator、prepared observations 和 protocol 都要能追溯。

`run_stage.sh`：调用 native-cache/fitting 服务器脚本，并保留每个 case 的日志。

```bash
bash "$PROJECT_ROOT/common/paper_experiments/joint_vs_2dfit/phase2_clean/server_commands/run_scmr_v5_fit_first.sh" \
  "$RUN_ROOT" "$RUN_ROOT/preprocessed" "$SUBJECT" "$RUN_ROOT/prepared" matlab
```

检查 `$RUN_ROOT/native_cache/${SUBJECT}_scmr_v5` 的 manifest：source observations、source preprocessed、mapping/native maps SHA 必须绑定；成功标志是 `SCMR-v5 Fit-first native maps and reconstructions: PASS`。
