# 03 生成 observations

## 作用

读取 02 的 MAT、DICOM 身份和 `protocol_scmr_v5_10_100.yaml`，生成 `observations.npz`、stack manifest 及 protocol 记录。`observations.npz` 是桥接产物，不是原始输入，也不能手工替代。

`run_stage.sh`：按显式 manifest/protocol 调用 `prepare_all_observations.py`。

## 输出

`$RUN_ROOT/prepared/$SUBJECT/{sax,2ch,4ch}/observations.npz`。manifest 必须记录 TR=2.61、VPS=87、TI=[10,100]、T2-prep=[35,45,55]、weight_idx=0..9 及源文件 SHA。

## 命令

```bash
python "$PROJECT_ROOT/common/trad/scripts/prepare_all_observations.py" \
  --manifest "$PROJECT_ROOT/subjects/$SUBJECT/manifest.json" \
  --preprocessed-root "$RUN_ROOT/preprocessed" \
  --prepared-root "$RUN_ROOT/prepared" \
  --protocol "$PROJECT_ROOT/configs/protocol_scmr_v5_10_100.yaml"
```

若出现 source observation SHA mismatch，先检查 prepared root、subject 和 manifest 是否来自同一次运行，不要覆盖修补。
