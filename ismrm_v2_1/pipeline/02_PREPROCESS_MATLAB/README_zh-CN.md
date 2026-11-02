# 02 MATLAB 预处理

## 作用

服务器上的 MATLAB 将十个权重的 DICOM 转换为每个 stack 的 `preprocessed.mat` 和 `preprocess_qc.json`。这是 fitting 前的输入生成步骤；不能复用含有错误 timing 的旧 prepared 结果，除非重新由正确 protocol 生成 observations。

`run_stage.sh` 是本模块唯一入口：调用 `local_preprocess_all`，随后执行 Python 预处理验证。

## 命令

```bash
RUN_ROOT=/data/.../runs/$SUBJECT
MATLAB=matlab
PYTHON=/path/to/python
bash "$PROJECT_ROOT/pipeline/02_PREPROCESS_MATLAB/run_stage.sh" \
  "$RUN_ROOT" "$SUBJECT" "$PROJECT_ROOT/subjects/$SUBJECT/manifest.json" \
  "$PROJECT_ROOT/configs/protocol_scmr_v5_10_100.yaml" "$MATLAB" "$PYTHON"
```

## 检查

输出在 `$RUN_ROOT/preprocessed/$SUBJECT/{sax,2ch,4ch}`。检查 `Mag_crop` 第三维为 10、MP-PCA enabled、`center_data=true`。脚本拒绝覆盖已有目录；已有半成品时先确认后清理该 subject 的新 run 根。
